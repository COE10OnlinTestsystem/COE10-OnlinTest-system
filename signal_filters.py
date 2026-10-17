"""
signal_filters.py
ตัวกรองสัญญาณสำหรับค่ามุมหัว (yaw/pitch) และสายตา ที่วัดจากกล้องเว็บแคมซึ่งมีสัญญาณรบกวน

- median3   : ตัดค่าที่พุ่งผิดปกติเฟรมเดียว (เช่น landmark กระโดดตอนภาพเบลอ)
- OneEuro   : ทำให้ค่านิ่งตอนหัวอยู่นิ่ง แต่ตามทันเมื่อหัวขยับเร็ว (ต่างจากค่าเฉลี่ยเคลื่อนที่ที่หน่วงเสมอ)
  อ้างอิง: Casiez et al., "1€ Filter", CHI 2012
"""
import collections
import math


class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.05, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)   # ยิ่งต่ำ ยิ่งนิ่งตอนอยู่กับที่
        self.beta = float(beta)               # ยิ่งสูง ยิ่งตามทันตอนขยับเร็ว
        self.d_cutoff = float(d_cutoff)
        self.reset()

    def reset(self):
        self._x = None
        self._dx = 0.0
        self._t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self._x is None or self._t is None or t <= self._t:
            self._x, self._t, self._dx = float(x), t, 0.0
            return float(x)
        dt = t - self._t
        if dt > 1.0:                           # ขาดช่วงนานเกิน (หน้าหายไป) เริ่มใหม่
            self._x, self._t, self._dx = float(x), t, 0.0
            return float(x)
        dx = (x - self._x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self._dx += a_d * (dx - self._dx)
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        a = self._alpha(cutoff, dt)
        self._x += a * (x - self._x)
        self._t = t
        return self._x


class PoseSmoother:
    """median 3 เฟรม (ตัดสไปก์) แล้วตามด้วย One Euro"""

    def __init__(self, min_cutoff=1.0, beta=0.05):
        self._buf = collections.deque(maxlen=3)
        self._f = OneEuroFilter(min_cutoff, beta)

    def reset(self):
        self._buf.clear()
        self._f.reset()

    def __call__(self, x, t):
        self._buf.append(float(x))
        med = sorted(self._buf)[len(self._buf) // 2]
        return self._f(med, t)
