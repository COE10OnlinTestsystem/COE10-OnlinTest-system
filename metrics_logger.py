"""
metrics_logger.py
บันทึกค่าที่ AI วัดได้ระหว่างสอบ (ตัวเลขล้วน ไม่มีภาพ/เสียง) เป็นไฟล์ CSV เพื่อตรวจย้อนหลังว่าระบบเตือนผิด/พลาดตรงไหน
และใช้จูนเกณฑ์จากข้อมูลจริงด้วย analyze_metrics.py

ไฟล์: evidence/metrics_<ชื่อผู้ใช้>_<เวลา>.csv   (บันทึก ~2 ครั้ง/วินาที)
"""
import csv
import os
import re
import time

FIELDS = ["t", "faces", "persons", "yaw", "pitch", "gaze", "head_state", "hands", "object", "reasons"]


class MetricsLogger:
    def __init__(self, folder, username, rate_hz=2.0):
        os.makedirs(folder, exist_ok=True)
        safe = re.sub(r'[\\/:*?"<>|\s]+', "_", str(username)).strip("_") or "student"
        self.path = os.path.join(folder, f"metrics_{safe}_{int(time.time())}.csv")
        self._f = open(self.path, "w", newline="", encoding="utf-8")
        self._w = csv.writer(self._f)
        self._w.writerow(FIELDS)
        self._t0 = None
        self._last = 0.0
        self._min_dt = 1.0 / rate_hz

    def log(self, now, faces, persons, yaw, pitch, gaze, head_state, hands, obj, reasons):
        if self._f is None or now - self._last < self._min_dt:
            return
        self._last = now
        if self._t0 is None:
            self._t0 = now
        self._w.writerow([round(now - self._t0, 2), faces, persons, round(yaw, 1), round(pitch, 1),
                          round(gaze, 3), head_state, hands, obj, "|".join(sorted(reasons))])

    def close(self):
        if self._f is not None:
            try:
                self._f.close()
            finally:
                self._f = None
