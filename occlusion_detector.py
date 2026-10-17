"""
occlusion_detector.py
ตรวจจับ "การบังกล้อง/บังคน" สำหรับระบบคุมสอบ ใช้คู่กับ main.py

เหตุผลดิบที่คืนให้ (ส่งต่อเข้า ViolationFilter เดิม ซึ่งจะกรองความต่อเนื่องให้เอง):
    "กล้องถูกบัง"   ภาพส่วนใหญ่เรียบผิดปกติ/มืดลงกะทันหัน (เอามือ กระดาษ หนังสือ มาบังเลนส์)
    "มือใกล้กล้อง"  มือใหญ่ผิดปกติในภาพ (ยกมือมาใกล้เลนส์)
    "มือบังหน้า"    มือทับใบหน้า หรือ หน้าหายไปขณะที่มีมืออยู่ในภาพ (บังคนข้างหลัง)

ไม่ต้องติดตั้งอะไรเพิ่ม ใช้ mp.solutions.hands ของ mediapipe 0.10.21
"""
import collections
import cv2
import numpy as np

# ---------------- ค่าเกณฑ์ (ปรับตามกล้อง/แสงจริงได้) ----------------
HAND_TOO_CLOSE_AREA = 0.25     # กรอบมือใหญ่เกินสัดส่วนนี้ของเฟรม
HAND_OVER_FACE_RATIO = 0.35    # มือทับหน้าเกินสัดส่วนนี้ของกรอบหน้า
FACE_MEMORY_SEC = 4.0          # จำจำนวนหน้าสูงสุดย้อนหลังกี่วินาที (ไว้จับหน้าหายตอนมีมือ)
HAND_EVERY_N = 2               # ตรวจมือทุก N เฟรม (เพิ่มถ้าเครื่องช้า)
BLOCK_STD = 4.0                # บล็อกภาพที่ std ต่ำกว่านี้ถือว่า "เรียบ"
UNIFORM_JUMP = 0.35            # สัดส่วนบล็อกเรียบเพิ่มจากปกติเท่านี้ = ถูกบัง
UNIFORM_ABS = 0.92             # เรียบเกินนี้ = ถูกบังแน่ (แม้ไม่มีค่าปกติให้เทียบ)
BRIGHT_DROP = 0.5              # ความสว่างตกเหลือต่ำกว่า 50% ของปกติ + เรียบพอสมควร
WARMUP_FRAMES = 30             # เฟรมแรกที่ใช้เรียนรู้ค่าปกติ ยังไม่แจ้ง


def _overlap_ratio(a, b):
    """สัดส่วนของกล่อง b ที่ถูกกล่อง a ทับ (x1,y1,x2,y2)"""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return inter / max(1.0, (b[2] - b[0]) * (b[3] - b[1]))


class OcclusionAnalyzer:
    def __init__(self, hands=None):
        self.hands = hands             # mp.solutions.hands.Hands (สร้างใน load_ai_models)
        self.hand_boxes = []           # กรอบมือล่าสุด (x1,y1,x2,y2) ไว้วาดบนภาพ
        self.reset()

    def set_hands(self, hands):
        self.hands = hands

    def reset(self):
        self._n = 0
        self._hand_cache = []
        self.hand_boxes = []
        self._face_hist = collections.deque()   # (เวลา, จำนวนหน้า)
        self._base_uniform = None               # ค่าปกติของสัดส่วนบล็อกเรียบ
        self._base_bright = None
        self._blocked = False
        self._warm = 0

    # ------------------------------------------------------------
    def _frame_stats(self, frame):
        small = cv2.resize(frame, (160, 120))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # แบ่งเป็นบล็อก 10x10 → 16x12 บล็อก แล้วดู std ของแต่ละบล็อก
        blocks = gray.reshape(12, 10, 16, 10).transpose(0, 2, 1, 3).reshape(12, 16, 100)
        uniform = float((blocks.std(axis=2) < BLOCK_STD).mean())
        return uniform, float(gray.mean())

    def _check_blocked(self, frame):
        uniform, bright = self._frame_stats(frame)
        if self._base_uniform is None:
            self._base_uniform, self._base_bright = uniform, bright
            self._warm = 1
            return False
        if self._warm < WARMUP_FRAMES:      # ช่วงเรียนรู้ค่าปกติ
            self._warm += 1
            a = 1.0 / self._warm
            self._base_uniform += a * (uniform - self._base_uniform)
            self._base_bright += a * (bright - self._base_bright)
            return uniform > UNIFORM_ABS
        jump = uniform - self._base_uniform
        blocked = (
            uniform > UNIFORM_ABS
            or jump > UNIFORM_JUMP
            or (bright < self._base_bright * BRIGHT_DROP and uniform > 0.5)
        )
        if not blocked:   # อัปเดตค่าปกติช้าๆ เฉพาะตอนไม่ถูกบัง (แสงห้องเปลี่ยนตามเวลา)
            self._base_uniform += 0.02 * (uniform - self._base_uniform)
            self._base_bright += 0.02 * (bright - self._base_bright)
        return blocked

    # ------------------------------------------------------------
    def _detect_hands(self, rgb, w, h):
        boxes = []
        if self.hands is None:
            return boxes
        res = self.hands.process(rgb)
        for hand in (res.multi_hand_landmarks or []):
            xs = [p.x * w for p in hand.landmark]
            ys = [p.y * h for p in hand.landmark]
            boxes.append((min(xs), min(ys), max(xs), max(ys)))
        return boxes

    # ------------------------------------------------------------
    def analyze(self, frame, rgb, face_boxes, n_faces, now):
        """
        frame = ภาพ BGR, rgb = ภาพ RGB (เฟรมเดียวกัน)
        face_boxes = [(x1,y1,x2,y2), ...] พิกเซล, n_faces = จำนวนหน้าที่นับได้
        คืน set ของเหตุผลดิบในเฟรมนี้
        """
        reasons = set()
        h, w = frame.shape[:2]

        # 1) กล้องถูกบัง
        self._blocked = self._check_blocked(frame)
        if self._blocked:
            reasons.add("กล้องถูกบัง")

        # 2) มือ (ข้ามเฟรมเพื่อลดภาระ ใช้ผลล่าสุดแทน)
        self._n += 1
        if self._n % HAND_EVERY_N == 0 or not self._hand_cache:
            try:
                self._hand_cache = self._detect_hands(rgb, w, h)
            except Exception as ex:
                print(f"⚠️ hand detect error: {ex}")
                self._hand_cache = []
        self.hand_boxes = self._hand_cache

        # จำจำนวนหน้าสูงสุดในช่วงหลัง
        self._face_hist.append((now, n_faces))
        while self._face_hist and now - self._face_hist[0][0] > FACE_MEMORY_SEC:
            self._face_hist.popleft()
        recent_max = max(n for _, n in self._face_hist)

        if self.hand_boxes:
            for hb in self.hand_boxes:
                if (hb[2] - hb[0]) * (hb[3] - hb[1]) / float(w * h) > HAND_TOO_CLOSE_AREA:
                    reasons.add("มือใกล้กล้อง")
                for fb in face_boxes:
                    if _overlap_ratio(hb, fb) > HAND_OVER_FACE_RATIO:
                        reasons.add("มือบังหน้า")
            # มีมืออยู่ในภาพ แต่หน้าที่เคยเห็นหายไป → มือ/วัตถุอาจบังคนข้างหลัง
            if n_faces < recent_max and not self._blocked:
                reasons.add("มือบังหน้า")
        return reasons

    # ------------------------------------------------------------
    def draw(self, frame):
        """วาดกรอบมือสีเหลืองลงภาพ (เห็นในจอและคลิปหลักฐาน)"""
        for x1, y1, x2, y2 in self.hand_boxes:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 220, 255), 2)
            cv2.putText(frame, "hand", (int(x1), max(15, int(y1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)
        return frame