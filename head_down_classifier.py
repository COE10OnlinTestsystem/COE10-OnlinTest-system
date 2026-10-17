"""
head_down_classifier.py
แยกประเภทการก้มหน้าของนักเรียน ว่าแบบไหนปกติ แบบไหนเสี่ยงทุจริต
ใช้คู่กับ FaceAnalyzer ใน main.py (รับ "องศาที่ก้ม/หันเทียบท่าปกติ" ที่ calibrate ไว้แล้ว)

ประเภทที่แยกได้
  ปกติ (ไม่เตือน แค่แสดงสถานะ)
    - ก้มชั่วครู่ (ปกติ)        ก้มไม่ถึง 2.5 วินาที เช่น มองคีย์บอร์ด เกาหัว บิดขี้เกียจ
    - ก้มเขียน/คิด (ปกติ)       ก้มระดับปานกลาง ค้างนานกว่านั้น เหมือนเขียนกระดาษทด/คิดเลข
  เสี่ยง (ส่งเหตุผลให้ ViolationFilter)
    - ก้มดูใต้โต๊ะ      ก้มลึกมากและค้าง (มองตัก/ลิ้นชัก/มือถือ/โพยใต้โต๊ะ)
    - ก้มมองด้านข้าง    ก้มพร้อมเบนหัวไปข้าง (มองกระเป๋า/ข้างตัว/กระดาษของเพื่อน)
    - ก้มหน้าซ้ำๆ       ก้ม-เงยซ้ำหลายรอบในเวลาสั้น (รูปแบบแอบดูเป็นจังหวะ)
    - ก้มหลบกล้อง       ก้มลึกแล้วใบหน้าหายไปจากภาพ
    - ก้มนานผิดปกติ     ก้มระดับปานกลางแต่นานเกินที่การเขียนควรใช้
"""
import collections


class HeadDownClassifier:
    # ---- เกณฑ์เวลา (วินาที) ----
    BRIEF_SEC = 2.0          # ก้มสั้นกว่านี้ = ชั่วครู่ ไม่เตือน
    DEEP_HOLD_SEC = 1.2      # ก้มลึกค้างเกินนี้ = ก้มดูใต้โต๊ะ
    SIDE_HOLD_SEC = 1.0      # ก้มเฉียงค้างเกินนี้ = ก้มมองด้านข้าง
    WRITE_MAX_SEC = 10.0     # ก้มเขียน/คิดได้นานสุด ก่อนนับว่านานผิดปกติ
    EPISODE_MIN_SEC = 0.8    # ก้มสั้นกว่านี้ไม่นับเป็น "รอบ" ของการก้มซ้ำ
    REPEAT_COUNT = 3         # ก้มกี่รอบ
    REPEAT_WINDOW = 40.0     # ภายในกี่วินาที ถึงนับว่า "ซ้ำๆ"
    REPEAT_HOLD = 8.0        # แจ้งต่อหลังครบเงื่อนไขกี่วินาที
    HIDE_MEMORY = 1.2        # หน้าหายภายในกี่วินาทีหลังก้มลึก = ก้มหลบกล้อง
    HIDE_MAX = 10.0          # แจ้ง "ก้มหลบกล้อง" นานสุดกี่วินาทีต่อครั้ง

    # ---- เกณฑ์มุม เทียบกับ limit (องศาที่ถือว่า "เริ่มก้ม") ----
    DEEP_MULT = 1.3          # ก้มลึก = limit × 1.3 (limit 15° → 19.5°)
    EXIT_MULT = 0.75         # เงยกลับต่ำกว่า limit × 0.75 ถือว่าจบการก้ม (กันกะพริบ)
    SIDE_YAW_MULT = 0.6      # เบนหัวเกิน limit × 0.6 (limit 15° → 9°) ตอนก้ม = มองด้านข้าง

    def __init__(self, limit=20.0):
        self.limit = float(limit)
        self.reset()

    def set_limit(self, limit):
        self.limit = float(limit)

    def reset(self):
        self._down_since = None
        self._deep_since = None
        self._side_since = None
        self._episodes = collections.deque()    # เวลาที่จบการก้มแต่ละรอบ
        self._repeat_until = 0.0
        self._last_deep_t = -1e9
        self._hiding = False
        self._hide_since = 0.0
        self.label = ""

    # ------------------------------------------------------------
    def update(self, d_pitch, d_yaw, now):
        """
        d_pitch = ก้มกี่องศาจากท่าปกติ (+ = ก้ม), d_yaw = หันกี่องศาจากท่าปกติ (ซ้าย/ขวา)
        คืน (set เหตุผลเสี่ยง, ข้อความสถานะ) เรียกทุกเฟรมที่เห็นหน้า
        """
        L = self.limit
        reasons = set()
        is_down = (d_pitch > L) if self._down_since is None else (d_pitch > L * self.EXIT_MULT)

        if not is_down:
            if self._down_since is not None:          # จบการก้ม 1 รอบ
                if now - self._down_since >= self.EPISODE_MIN_SEC:
                    self._episodes.append(now)
                self._down_since = self._deep_since = self._side_since = None
                self._prune(now)
                if len(self._episodes) >= self.REPEAT_COUNT:
                    self._repeat_until = now + self.REPEAT_HOLD
            if now < self._repeat_until:
                reasons.add("ก้มหน้าซ้ำๆ")
            self.label = "ก้มหน้าซ้ำๆ (เสี่ยง)" if reasons else ""
            return reasons, self.label

        # ---- กำลังก้มอยู่ ----
        if self._down_since is None:
            self._down_since = now
        dur = now - self._down_since

        deep = d_pitch > L * self.DEEP_MULT
        if deep:
            self._last_deep_t = now
            if self._deep_since is None: self._deep_since = now
        else:
            self._deep_since = None

        side = abs(d_yaw) > L * self.SIDE_YAW_MULT
        if side:
            if self._side_since is None: self._side_since = now
        else:
            self._side_since = None

        risky = None
        if self._deep_since is not None and now - self._deep_since >= self.DEEP_HOLD_SEC:
            risky = "ก้มดูใต้โต๊ะ"
        elif self._side_since is not None and now - self._side_since >= self.SIDE_HOLD_SEC:
            risky = "ก้มมองด้านข้าง"
        elif dur > self.WRITE_MAX_SEC:
            risky = "ก้มนานผิดปกติ"
        if risky:
            reasons.add(risky)

        # รอบนี้จะเป็นรอบที่ 3 ในเวลาสั้น ๆ แล้ว
        self._prune(now)
        if (len(self._episodes) + 1 >= self.REPEAT_COUNT and dur >= self.EPISODE_MIN_SEC) or now < self._repeat_until:
            reasons.add("ก้มหน้าซ้ำๆ")

        if reasons:
            self.label = f"{sorted(reasons)[0]} (เสี่ยง)"
        elif dur < self.BRIEF_SEC:
            self.label = "ก้มชั่วครู่ (ปกติ)"
        else:
            self.label = "ก้มเขียน/คิด (ปกติ)"
        return reasons, self.label

    # ------------------------------------------------------------
    def face_lost(self, lost, now):
        """
        เรียกทุกเฟรมพร้อมสถานะว่าหน้าหายหรือไม่ คืน True ถ้าควรเตือน "ก้มหลบกล้อง"
        (ใบหน้าหายทันทีหลังก้มลึก ต่างจากหน้าหายเพราะลุกไปหรือหันหลัง)
        """
        if not lost:
            self._hiding = False
            return False
        if not self._hiding and now - self._last_deep_t <= self.HIDE_MEMORY:
            self._hiding = True; self._hide_since = now
        if self._hiding and now - self._hide_since > self.HIDE_MAX:
            self._hiding = False
        if self._hiding:
            self.label = "ก้มหลบกล้อง (เสี่ยง)"
        return self._hiding

    def _prune(self, now):
        while self._episodes and now - self._episodes[0] > self.REPEAT_WINDOW:
            self._episodes.popleft()