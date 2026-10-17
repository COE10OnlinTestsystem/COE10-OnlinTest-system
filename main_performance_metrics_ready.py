import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox, simpledialog, filedialog
import cv2
from PIL import Image, ImageTk
import numpy as np
import time
import os
import sys
import db_helper as db 
from occlusion_detector import OcclusionAnalyzer
from head_down_classifier import HeadDownClassifier
from signal_filters import PoseSmoother
from metrics_logger import MetricsLogger
import mss
import subprocess
import requests
import threading
import socket
from flask import Flask, request, jsonify, send_from_directory, g
import logging
import uuid
import shutil
import sqlite3
import json
import contextlib
import collections
import random
import csv
import io
from werkzeug.utils import secure_filename

# ปิด Log ของ Flask
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# ================= จัดการ Path (สำหรับ .exe) =================
def get_base_path():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

BASE_DIR = get_base_path()
CONFIG_FILE = os.path.join(BASE_DIR, "server_config.txt")
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'Submitted_Exams')

if not os.path.exists(UPLOAD_FOLDER): os.makedirs(UPLOAD_FOLDER)

# ================= Media helper (รูปภาพ / เสียงประกอบข้อสอบ) — รวมไว้ในไฟล์เดียว =================
# โครงสร้าง: ตาราง question_media ใน media.db (SQLite) เก็บว่าโจทย์ข้อไหน/ช่องไหนมีไฟล์อะไร
#   slot 0 = ตัวโจทย์, slot 1..n = ช้อยส์ข้อที่ 1..n, kind = image|audio
# ไฟล์จริงเก็บใน media/ (ฝั่งอาจารย์) ฝั่งนักเรียนโหลดจาก Flask แล้วเก็บชั่วคราวใน media_cache/
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")


MEDIA_DIR = os.path.join(BASE_DIR, "media")            # ฝั่งอาจารย์: ไฟล์ต้นฉบับ
CACHE_DIR = os.path.join(BASE_DIR, "media_cache")      # ฝั่งนักเรียน: ไฟล์ที่โหลดมา
MEDIA_DB = os.path.join(BASE_DIR, "media.db")

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp")
AUDIO_EXTS = (".mp3", ".wav", ".ogg", ".flac")
MAX_IMAGE_MB = 15
MAX_AUDIO_MB = 25
MAX_IMAGE_SIDE = 1600   # ย่อรูปที่ใหญ่กว่านี้ ลดเวลาโหลดของนักเรียน


class MediaError(Exception):
    pass


# ------------------------------------------------------------------ database
@contextlib.contextmanager
def _conn():
    c = sqlite3.connect(MEDIA_DB, timeout=10)
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init_db():
    """สร้างตารางถ้ายังไม่มี (เรียกซ้ำได้ ไม่ทำลายข้อมูลเดิม)"""
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS question_media (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                question_id INTEGER NOT NULL,
                slot        INTEGER NOT NULL DEFAULT 0,
                kind        TEXT    NOT NULL CHECK (kind IN ('image', 'audio')),
                filename    TEXT    NOT NULL,
                orig_name   TEXT,
                UNIQUE (question_id, slot, kind)
            )""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_qm_question ON question_media(question_id)")


init_db()


# ------------------------------------------------------------------ files
def kind_of(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in AUDIO_EXTS:
        return "audio"
    return None


def validate(path):
    """ตรวจไฟล์ก่อนแนบ คืนค่า kind หรือ raise MediaError"""
    if not path or not os.path.isfile(path):
        raise MediaError("ไม่พบไฟล์")
    kind = kind_of(path)
    if kind is None:
        raise MediaError("ชนิดไฟล์ไม่รองรับ\nรูปภาพ: " + ", ".join(IMAGE_EXTS) +
                         "\nเสียง: " + ", ".join(AUDIO_EXTS))
    size_mb = os.path.getsize(path) / 1048576
    limit = MAX_AUDIO_MB if kind == "audio" else MAX_IMAGE_MB
    if size_mb > limit:
        raise MediaError(f"ไฟล์ใหญ่เกินไป ({size_mb:.1f} MB) จำกัดไม่เกิน {limit} MB")
    if kind == "image":
        try:
            with Image.open(path) as im:
                im.verify()
        except Exception:
            raise MediaError("ไฟล์รูปภาพเสียหายหรือเปิดไม่ได้")
    return kind


def _store_file(src, kind):
    """คัดลอกไฟล์เข้า media/ ด้วยชื่อ uuid คืนชื่อไฟล์ใหม่"""
    os.makedirs(MEDIA_DIR, exist_ok=True)
    uid = uuid.uuid4().hex
    if kind == "image":
        is_jpg = src.lower().endswith((".jpg", ".jpeg"))
        name = uid + (".jpg" if is_jpg else ".png")
        with Image.open(src) as im:
            im.load()
            if max(im.size) > MAX_IMAGE_SIDE:
                im.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.LANCZOS)
            if is_jpg:
                im.convert("RGB").save(os.path.join(MEDIA_DIR, name), quality=88)
            else:
                if im.mode not in ("RGB", "RGBA", "L", "P"):
                    im = im.convert("RGBA")
                im.save(os.path.join(MEDIA_DIR, name), "PNG")
        return name
    name = uid + os.path.splitext(src)[1].lower()
    shutil.copy2(src, os.path.join(MEDIA_DIR, name))
    return name


def _gc_file(filename):
    """ลบไฟล์ใน media/ ถ้าไม่มีโจทย์ข้อไหนอ้างอิงแล้ว"""
    with _conn() as c:
        used = c.execute("SELECT 1 FROM question_media WHERE filename=? LIMIT 1", (filename,)).fetchone()
    if not used:
        with contextlib.suppress(OSError):
            os.remove(os.path.join(MEDIA_DIR, filename))


# ------------------------------------------------------------------ CRUD
def list_media(question_id):
    with _conn() as c:
        rows = c.execute("SELECT slot, kind, filename, orig_name FROM question_media "
                         "WHERE question_id=? ORDER BY slot, kind", (question_id,)).fetchall()
    return [{"slot": s, "kind": k, "filename": f, "orig_name": o or f,
             "path": os.path.join(MEDIA_DIR, f)} for s, k, f, o in rows]


def index_for_questions(question_ids):
    """สำหรับ Flask: {"12": [{slot, kind, filename}, ...]}"""
    out = {}
    if not question_ids:
        return out
    marks = ",".join("?" * len(question_ids))
    with _conn() as c:
        rows = c.execute(f"SELECT question_id, slot, kind, filename FROM question_media "
                         f"WHERE question_id IN ({marks}) ORDER BY slot, kind", list(question_ids)).fetchall()
    for qid, slot, kind, fn in rows:
        out.setdefault(str(qid), []).append({"slot": slot, "kind": kind, "filename": fn})
    return out


def replace_all(question_id, desired):
    """
    แทนที่สื่อทั้งหมดของโจทย์ข้อนี้ด้วยรายการ desired = [(slot, kind, path), ...]
    ไฟล์ที่อยู่ใน media/ อยู่แล้ว (กรณีแก้ไขโจทย์) จะถูกใช้ซ้ำโดยไม่คัดลอกใหม่
    ถ้าไฟล์ใดผิดพลาด จะไม่แตะข้อมูลเดิมเลย
    """
    media_root = os.path.abspath(MEDIA_DIR)
    with _conn() as c:
        old_rows = c.execute("SELECT filename, orig_name FROM question_media WHERE question_id=?",
                             (question_id,)).fetchall()
    old_names = {f: o for f, o in old_rows}

    new_rows, created = [], []
    try:
        for slot, kind, path in desired:
            if os.path.dirname(os.path.abspath(path)) == media_root:
                fn = os.path.basename(path)
                orig = old_names.get(fn, fn)
            else:
                validate(path)
                fn = _store_file(path, kind)
                created.append(fn)
                orig = os.path.basename(path)
            new_rows.append((question_id, slot, kind, fn, orig))
    except Exception:
        for fn in created:
            with contextlib.suppress(OSError):
                os.remove(os.path.join(MEDIA_DIR, fn))
        raise

    with _conn() as c:
        c.execute("DELETE FROM question_media WHERE question_id=?", (question_id,))
        c.executemany("INSERT INTO question_media (question_id, slot, kind, filename, orig_name) "
                      "VALUES (?,?,?,?,?)", new_rows)
    for fn in old_names:
        _gc_file(fn)


def delete_question_media(question_id):
    with _conn() as c:
        names = [r[0] for r in c.execute("SELECT filename FROM question_media WHERE question_id=?",
                                         (question_id,)).fetchall()]
        c.execute("DELETE FROM question_media WHERE question_id=?", (question_id,))
    for fn in names:
        _gc_file(fn)


# ------------------------------------------------------------------ student side
def teacher_base_url():
    if not os.path.exists(CONFIG_FILE):
        return None
    try:
        with open(CONFIG_FILE, "r") as f:
            ip = f.read().strip()
    except OSError:
        return None
    if not ip:
        return None
    ip = ip.strip().rstrip("/")
    if ip.startswith(("http://", "https://")):   # เช่น https://xxxx.trycloudflare.com (Cloudflare Tunnel)
        return ip
    if ":" not in ip:
        ip += ":5000"
    return f"http://{ip}"


def _http():
    """
    session ที่ไม่ใช้ proxy ของระบบ: เครือข่ายโรงเรียน/Windows ที่ตั้ง proxy ไว้ จะทำให้ requests
    ส่งคำขอไปเครื่องอาจารย์ (IP ในวง LAN) ผ่าน proxy แล้วล้มเหลว
    """
    import requests
    s = requests.Session(); s.trust_env = False
    client = globals().get("api")
    if client is not None and client.token: s.headers["X-Token"] = client.token   # ยืนยันตัวตนกับเซิร์ฟเวอร์
    return s


def _candidate_bases():
    bases = []
    b = teacher_base_url()
    if b: bases.append(b)
    for local in ("http://127.0.0.1:5000",):   # เผื่อทดสอบอาจารย์+นักเรียนในเครื่องเดียวกัน
        if local not in bases: bases.append(local)
    return bases


def _local_media_fallback(question_ids):
    """ถ้าเครื่องนี้มีไฟล์สื่อของอาจารย์อยู่แล้ว (รันทั้ง 2 ฝั่งในเครื่องเดียว) ใช้ไฟล์ในเครื่องได้เลย"""
    out = {}
    try:
        for qid, items in index_for_questions(question_ids).items():
            lst = []
            for it in items:
                local = os.path.join(MEDIA_DIR, os.path.basename(it["filename"]))
                if os.path.exists(local): lst.append({**it, "local": local})
            if lst: out[int(qid)] = lst
    except Exception:
        pass
    return out


def fetch_media_for_questions(question_ids):
    """
    ดึงรายการสื่อของโจทย์หลายข้อในคำขอเดียว แล้วดาวน์โหลดไฟล์ที่ยังไม่มีลง media_cache/
    คืนค่า ({question_id: [{slot, kind, filename, local}, ...]}, error_or_None)
    """
    if not question_ids:
        return {}, None
    http = _http()
    ids = ",".join(str(i) for i in question_ids)
    index, base, errors = None, None, []
    for cand in _candidate_bases():
        try:
            r = http.get(f"{cand}/media_index", params={"ids": ids}, timeout=5)
            r.raise_for_status()
            index, base = r.json(), cand
            break
        except Exception as e:
            errors.append(f"{cand} → {type(e).__name__}")
    if index is None:
        local = _local_media_fallback(question_ids)
        if local:
            return local, None
        msg = "ติดต่อเครื่องอาจารย์ไม่ได้ (" + "; ".join(errors) + ")"
        if not teacher_base_url():
            msg = "ยังไม่ได้ตั้งค่า IP อาจารย์ (กดปุ่มตั้งค่า IP ที่หน้า Login)"
        print(f"⚠️ โหลดสื่อไม่สำเร็จ: {msg}")
        return {}, msg

    os.makedirs(CACHE_DIR, exist_ok=True)
    out, err = {}, None
    for qid, items in index.items():
        lst = []
        for it in items:
            fn = os.path.basename(it["filename"])
            local = os.path.join(CACHE_DIR, fn)
            if not os.path.exists(local):
                try:
                    with http.get(f"{base}/media/{fn}", stream=True, timeout=20) as resp:
                        resp.raise_for_status()
                        with open(local + ".part", "wb") as f:
                            for chunk in resp.iter_content(65536):
                                f.write(chunk)
                    os.replace(local + ".part", local)
                except Exception as e:
                    # ไฟล์หายจากเซิร์ฟเวอร์ แต่อาจอยู่ในเครื่องนี้
                    here = os.path.join(MEDIA_DIR, fn)
                    if os.path.exists(here):
                        lst.append({**it, "filename": fn, "local": here}); continue
                    err = f"ดาวน์โหลด {fn} ไม่ได้: {type(e).__name__}"
                    print(f"⚠️ {err} ({e})")
                    continue
            lst.append({**it, "filename": fn, "local": local})
        out[int(qid)] = lst
    return out, err


def clear_cache():
    PLAYER.stop()
    if os.path.isdir(CACHE_DIR):
        for fn in os.listdir(CACHE_DIR):
            with contextlib.suppress(OSError):
                os.remove(os.path.join(CACHE_DIR, fn))


# ------------------------------------------------------------------ audio
class AudioPlayer:
    """เล่นเสียงทีละไฟล์ผ่าน pygame.mixer (ติดตั้ง: pip install pygame)"""
    def __init__(self):
        self._ok = None
        self._pg = None
        self.error = ""

    def available(self):
        if self._ok is None:
            try:
                import pygame
                pygame.mixer.init()
                self._pg = pygame
                self._ok = True
            except Exception as e:
                self._ok = False
                self.error = str(e)
        return self._ok

    def play(self, path):
        if not self.available():
            return False
        try:
            self._pg.mixer.music.load(path)
            self._pg.mixer.music.play()
            return True
        except Exception as e:
            self.error = str(e)
            return False

    def is_busy(self):
        return bool(self._ok and self._pg.mixer.music.get_busy())

    def stop(self):
        if self._ok:
            with contextlib.suppress(Exception):
                self._pg.mixer.music.stop()
                if hasattr(self._pg.mixer.music, "unload"):
                    self._pg.mixer.music.unload()


PLAYER = AudioPlayer()

# ให้โค้ดส่วนอื่นเรียกแบบ media.xxx ได้เหมือนเดิม (ชี้มาที่โมดูลนี้เอง)
media = sys.modules[__name__]


# ================= FLASK SERVER (ฝั่งอาจารย์) =================
# เครื่องอาจารย์เป็น "เซิร์ฟเวอร์" ตัวเดียวที่ถือฐานข้อมูล นักเรียนทุกคนคุยผ่าน HTTP API นี้
# จึงสอบได้ทั้งในวงเดียวกัน (LAN) และนอกวง (ผ่าน Tailscale / Cloudflare Tunnel)
from functools import wraps

app_flask = Flask(__name__)
app_flask.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024   # คลิปหน้าจอสูงสุด 2 GB
server_started = False
SERVER_PORT = 5000


def _auth_user():
    return db.user_by_token(request.headers.get("X-Token", ""))


def require_token(fn):
    """ทุก API (ยกเว้น login/register) ต้องแนบ token ที่ได้ตอน login — กันคนนอกยิง API มาแก้คะแนน"""
    @wraps(fn)
    def wrapper(*a, **kw):
        u = _auth_user()
        if not u or u["role"] != "student" or not u["approved"]:
            return jsonify({"ok": False, "error": "unauthorized"}), 401
        g.user = u
        return fn(*a, **kw)
    return wrapper


def _student_exam(exam_id):
    """ข้อสอบต้องเป็นของอาจารย์ที่นักเรียนคนนี้สังกัดเท่านั้น"""
    exam = db.get_exam_full(exam_id)
    if not exam or exam["owner_id"] != g.user["teacher_id"]: return None
    return exam


@app_flask.route('/api/ping', methods=['GET', 'POST'])
def api_ping():
    return jsonify({"ok": True, "app": "ai-proctor", "time": time.time()})


@app_flask.route('/api/login', methods=['POST'])
def api_login():
    d = request.get_json(silent=True) or {}
    res = db.login_user(str(d.get("username", "")), str(d.get("password", "")))
    if not res: return jsonify({"ok": False, "error": "invalid"}), 200
    uid, role, approved, tid = res
    if role != "student": return jsonify({"ok": False, "error": "teacher_remote"}), 200
    if not approved: return jsonify({"ok": False, "error": "pending"}), 200
    return jsonify({"ok": True, "user_id": uid, "role": role, "teacher_id": tid, "token": db.issue_token(uid)})


@app_flask.route('/api/register', methods=['POST'])
def api_register():
    d = request.get_json(silent=True) or {}
    u, p, c = str(d.get("username", "")).strip(), str(d.get("password", "")), str(d.get("class_code", "")).strip()
    if not u or not p or not c: return jsonify({"ok": False, "result": "missing"})
    return jsonify({"ok": True, "result": db.register_student(u, p, c)})


@app_flask.route('/api/exam_by_code', methods=['POST'])
@require_token
def api_exam_by_code():
    code = str((request.get_json(silent=True) or {}).get("code", "")).strip()
    exam = db.get_exam_by_code(code)
    if not exam or exam["owner_id"] != g.user["teacher_id"]:
        return jsonify({"ok": False})
    return jsonify({"ok": True, "exam": {k: exam[k] for k in ("id", "name", "active", "duration", "sensitivity")}})


@app_flask.route('/api/questions', methods=['POST'])
@require_token
def api_questions():
    exam = _student_exam((request.get_json(silent=True) or {}).get("exam_id"))
    if not exam or not exam["active"]: return jsonify({"ok": False, "error": "closed"})
    # ไม่ส่งเฉลยไปเครื่องนักเรียน (เดิมส่งไปด้วย นักเรียนที่เก่งคอมอาจแอบดูได้)
    qs = [{"id": q["id"], "type": q["type"], "text": q["text"], "options": q["options"]}
          for q in db.get_questions_by_exam(exam["id"]) if q["type"] == "MCQ"]
    return jsonify({"ok": True, "questions": qs})


@app_flask.route('/api/submit', methods=['POST'])
@require_token
def api_submit():
    d = request.get_json(silent=True) or {}
    exam = _student_exam(d.get("exam_id"))
    if not exam: return jsonify({"ok": False, "error": "exam"}), 400
    picked = {str(k): v for k, v in (d.get("answers") or {}).items()}
    sc = 0; max_sc = 0; answers = []
    for q in db.get_questions_by_exam(exam["id"]):   # ตรวจคะแนนที่เซิร์ฟเวอร์ ใช้เฉลยในฐานข้อมูลของอาจารย์
        u = picked.get(str(q["id"]), "")
        if q["type"] != "MCQ": continue   # ข้อเขียนเก่า (เลิกใช้แล้ว) ไม่นำมาคิดคะแนน
        max_sc += 1; ok = mcq_is_correct(q, u)
        sc += 1 if ok else 0
        answers.append((q["id"], str(u) if str(u) not in ("0", "None") else "", 1 if ok else 0))
    def num(key, lo=0.0, hi=1e7):
        try: return max(lo, min(hi, float(d.get(key) or 0)))
        except (TypeError, ValueError): return 0.0
    detail = d.get("detail") if isinstance(d.get("detail"), dict) else {}
    db.save_score(exam["id"], g.user["username"], sc, max_score=max_sc, user_id=g.user["id"],
                  exam_seconds=num("exam_seconds"), flagged_seconds=num("flagged_seconds"),
                  risk_pct=num("risk_pct", 0, 100),
                  violation_detail={str(k)[:40]: round(float(v), 1) for k, v in list(detail.items())[:20]
                                    if isinstance(v, (int, float))},
                  answers=answers)
    return jsonify({"ok": True, "score": sc, "max": max_sc})


@app_flask.route('/api/heartbeat', methods=['POST'])
@require_token
def api_heartbeat():
    d = request.get_json(silent=True) or {}
    exam_id = d.get("exam_id")
    if exam_id is not None and not _student_exam(exam_id): exam_id = None
    status = d.get("live_status")
    db.update_heartbeat(g.user["id"], None if status is None else str(status)[:200], exam_id)
    return jsonify({"ok": True, "kicked": db.check_user_status(g.user["id"]) == 1})


@app_flask.route('/api/violation', methods=['POST'])
@require_token
def api_violation():
    d = request.get_json(silent=True) or {}
    if _student_exam(d.get("exam_id")):
        db.log_violation(d.get("exam_id"), g.user["id"], g.user["username"], str(d.get("reason", ""))[:60])
    return jsonify({"ok": True})


@app_flask.route('/api/change_password', methods=['POST'])
@require_token
def api_change_password():
    d = request.get_json(silent=True) or {}
    new = str(d.get("new", ""))
    if len(new) < 6: return jsonify({"ok": False, "error": "short"})
    return jsonify({"ok": bool(db.change_password(g.user["id"], str(d.get("old", "")), new))})


@app_flask.route('/api/clear_live', methods=['POST'])
@require_token
def api_clear_live():
    db.clear_live_status(g.user["id"])
    return jsonify({"ok": True})


@app_flask.route('/upload', methods=['POST'])
@require_token
def upload_file():
    if 'file' not in request.files: return jsonify({"msg": "No file"}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({"msg": "No selected file"}), 400
    fname = secure_filename(file.filename) or f"upload_{int(time.time())}.mp4"
    # บังคับให้ชื่อไฟล์ขึ้นต้นด้วยชื่อผู้ส่งจริง กันนักเรียนส่งไฟล์ทับของคนอื่น
    user_tag = secure_filename(g.user["username"]) or f"user{g.user['id']}"
    if user_tag not in fname: fname = f"{user_tag}_{fname}"
    file.save(os.path.join(UPLOAD_FOLDER, fname))
    print(f"📥 ได้รับไฟล์: {fname}")
    return jsonify({"msg": "Success"}), 200


@app_flask.route('/media_index')
@require_token
def media_index():
    # นักเรียนถามว่าโจทย์ข้อไหนมีรูป/เสียงอะไรบ้าง: /media_index?ids=1,2,3
    ids = [int(x) for x in request.args.get('ids', '').split(',') if x.strip().isdigit()]
    return jsonify(media.index_for_questions(ids))


@app_flask.route('/media/<path:filename>')
@require_token
def media_file(filename):
    return send_from_directory(media.MEDIA_DIR, filename)


def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80)); ip = s.getsockname()[0]
    except: ip = '127.0.0.1'
    finally: s.close()
    return ip


def get_tailscale_ip():
    """IP ของ Tailscale (ช่วง 100.64.0.0/10) ถ้าเครื่องนี้ต่อ Tailscale อยู่"""
    ips = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET): ips.add(info[4][0])
    except OSError:
        pass
    for ip in ips:
        a, b = (int(x) for x in ip.split(".")[:2])
        if a == 100 and 64 <= b <= 127: return ip
    return None


def start_teacher_server():
    global server_started
    if server_started: return
    def run():
        try:
            from waitress import serve   # เซิร์ฟเวอร์จริง รองรับนักเรียนพร้อมกันหลายสิบคนได้ดีกว่า
            print(f"🌐 เซิร์ฟเวอร์ข้อสอบ (waitress) พอร์ต {SERVER_PORT}")
            serve(app_flask, host="0.0.0.0", port=SERVER_PORT, threads=24, channel_timeout=300)
        except ImportError:
            print(f"🌐 เซิร์ฟเวอร์ข้อสอบ (Flask) พอร์ต {SERVER_PORT}")
            app_flask.run(host='0.0.0.0', port=SERVER_PORT, use_reloader=False, threaded=True)
        except OSError as ex:
            print(f"❌ เปิดเซิร์ฟเวอร์ไม่ได้ (พอร์ต {SERVER_PORT} ถูกใช้อยู่?): {ex}")
    threading.Thread(target=run, daemon=True).start()
    server_started = True


# ================= API Client (ฝั่งนักเรียน) =================
class ApiError(Exception):
    pass


class ApiClient:
    """ฝั่งนักเรียนเรียกเซิร์ฟเวอร์อาจารย์ผ่าน HTTP แทนการเปิดไฟล์ฐานข้อมูลเอง"""
    def __init__(self):
        self.token = None; self.base = None

    def _bases(self):
        if self.base: return [self.base] + [b for b in media._candidate_bases() if b != self.base]
        return media._candidate_bases()

    def call(self, path, payload=None, timeout=10):
        last = None
        for base in self._bases():
            try:
                r = media._http().post(f"{base}{path}", json=payload or {}, timeout=timeout,
                                       headers={"X-Token": self.token or ""})
                if r.status_code == 401: raise ApiError("session หมดอายุ กรุณา Logout แล้วเข้าใหม่")
                r.raise_for_status()
                self.base = base
                return r.json()
            except ApiError:
                raise
            except Exception as ex:
                last = ex
        if not media.teacher_base_url():
            raise ApiError("ยังไม่ได้ตั้งค่าที่อยู่เครื่องอาจารย์ (กดปุ่ม ⚙ ตั้งค่า IP ที่หน้า Login)")
        raise ApiError(f"ติดต่อเครื่องอาจารย์ไม่ได้ ({type(last).__name__}) — ตรวจ IP/อินเทอร์เน็ต และให้อาจารย์ล็อกอินค้างไว้")

    def login(self, username, password):
        d = self.call("/api/login", {"username": username, "password": password})
        if d.get("ok"): self.token = d["token"]
        return d

    def logout(self):
        self.token = None


api = ApiClient()


# ================= UI Theme: Sci-Fi HUD (Cyan / Teal on Dark Slate) =================
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

# --- Palette ---
BG_APP      = "#0f172a"   # dark metallic slate — window / base background
BG_PANEL    = "#141f38"   # panel background
BG_PANEL_2  = "#1b2a4a"   # nested / inner panel background
BG_INPUT    = "#0c1526"   # entry field background
CYAN        = "#06b6d4"   # primary accent
CYAN_DIM    = "#0e7490"   # hover / dim cyan (also used for glow borders)
TEAL        = "#14b8a6"   # secondary accent (go / positive actions)
TEAL_DIM    = "#0f766e"
GREEN_OK    = "#22c55e"   # online / safe
RED_ALERT   = "#ef4444"   # offline / warning / danger
RED_DIM     = "#7f1d1d"
AMBER_WARN  = "#f59e0b"   # pending / notice
TEXT_MAIN   = "#e2e8f0"   # primary text
TEXT_MUTED  = "#7c8ba1"   # secondary / muted text

BORDER_SOFT = "#233350"  # subtle low-contrast border for grouping (non-focal)
WHITE_TXT   = "#f8fafc"

# --- Fonts ---
FONT_H1     = ("Arial", 30, "bold")
FONT_H2     = ("Arial", 22, "bold")
FONT_H3     = ("Arial", 18, "bold")
FONT_BODY   = ("Arial", 16)
FONT_BOLD   = ("Arial", 16, "bold")
FONT_SMALL  = ("Arial", 12)
FONT_SMALL_B= ("Arial", 12, "bold")
FONT_SUB    = ("Arial", 14)
FONT_MONO   = ("Consolas", 15, "bold")   # HUD-style readouts: IP, subject code, timers
FONT_MONO_LG= ("Consolas", 26, "bold")   # large HUD readouts: countdown timer

# --- Background image (ภาพห้องเรียน/กล้องวงจรปิด) ---
BG_IMAGE_PATH = os.path.join(BASE_DIR, "assets", "classroom_bg.jpg")

def set_page_background(parent, tint_color=BG_APP, tint_alpha=0.55):
    """
    วางภาพห้องเรียนเป็นพื้นหลังของหน้าจอ (parent) พร้อมทับสีเข้มบางๆ (tint)
    เพื่อให้ตัวหนังสือ/แผงควบคุมด้านบนยังอ่านง่าย ปรับขนาดอัตโนมัติตามขนาดหน้าต่าง
    ถ้าไม่พบไฟล์ภาพ จะไม่ทำอะไร (โปรแกรมยังใช้งานได้ตามปกติ)
    """
    if not os.path.exists(BG_IMAGE_PATH):
        print(f"⚠️ ไม่พบไฟล์พื้นหลัง: {BG_IMAGE_PATH}")
        print(f"   (ตรวจสอบว่ามีไฟล์ assets/classroom_bg.jpg อยู่ข้างๆ main.py จริงหรือไม่)")
        return None
    try:
        raw = Image.open(BG_IMAGE_PATH).convert("RGB")
    except Exception as e:
        print(f"⚠️ โหลดภาพพื้นหลังไม่สำเร็จ: {e}")
        return None

    lbl = tk.Label(parent, bd=0, highlightthickness=0)
    lbl._last_size = None

    def redraw(event=None):
        try:
            w = max(parent.winfo_width(), 10); h = max(parent.winfo_height(), 10)
            print(f"[BG] redraw: parent={parent.__class__.__name__} size={w}x{h}")
            if lbl._last_size == (w, h): return
            lbl._last_size = (w, h)
            img = raw.resize((w, h), Image.LANCZOS)
            tint = Image.new("RGB", (w, h), tint_color)
            img = Image.blend(img, tint, tint_alpha)
            photo = ImageTk.PhotoImage(img)
            lbl.configure(image=photo)
            lbl.image = photo  # กันรูปโดนเก็บขยะ (garbage collected)
            print(f"[BG] วาดพื้นหลังสำเร็จ ขนาด {w}x{h}")
        except Exception as e:
            import traceback
            print(f"⚠️ วาดพื้นหลังไม่สำเร็จ: {e}")
            traceback.print_exc()

    parent.bind("<Configure>", redraw, add="+")
    lbl.place(relx=0, rely=0, relwidth=1, relheight=1)
    lbl.lower()
    parent.update_idletasks()
    redraw()
    parent.after(50, redraw)
    parent.after(300, redraw)
    return lbl

# ================= AI Setup (Proctoring) =================
# โหลดโมเดล AI ในเธรดเบื้องหลังตั้งแต่เปิดโปรแกรม หน้าต่างจึงขึ้นทันที ไม่ต้องรอ 10-30 วินาที
# (เดิมโหลดตอน import ทำให้เปิดโปรแกรมแล้วเงียบไปนานจนดูเหมือนค้าง)
# หมายเหตุ: ต้องใช้ mediapipe==0.10.21 (เวอร์ชัน 0.10.30 ขึ้นไปตัด mp.solutions ทิ้งแล้ว)
face_mesh = None
face_detection = None
hand_model = None
model = None
yolo_name = None
AI_FACE_ERROR = None
AI_READY = threading.Event()
AI_STATUS = {"step": 0, "total": 5, "text": "เตรียมระบบ AI...", "errors": []}
YOLO_CANDIDATES = ["yolo11s.pt", "yolov8s.pt", "yolo11n.pt", "yolov8n.pt"]


def _ai_step(n, text):
    AI_STATUS["step"] = n; AI_STATUS["text"] = text; print(f"⏳ [{n}/{AI_STATUS['total']}] {text}")


def load_ai_models():
    global face_mesh, face_detection, hand_model, model, yolo_name, AI_FACE_ERROR
    try:
        _ai_step(1, "กำลังโหลด MediaPipe (ตรวจใบหน้า)...")
        import mediapipe as mp
        if not hasattr(mp, "solutions"):
            raise ImportError(f"mediapipe {getattr(mp, '__version__', '?')} ไม่มี mp.solutions "
                              f"— ติดตั้งใหม่ด้วย: pip install mediapipe==0.10.21")
        _ai_step(2, "กำลังสร้างโมเดลจับจุดใบหน้าและนับจำนวนคน...")
        # ใช้ใบหน้าที่ใหญ่ที่สุด (คนสอบ) ในการคำนวณมุมหัว แต่ให้ตรวจได้ถึง 3 หน้าเพื่อนับจำนวนคน
        fm = mp.solutions.face_mesh.FaceMesh(max_num_faces=3, refine_landmarks=True,
                                              min_detection_confidence=0.5, min_tracking_confidence=0.5)
        # model_selection=1 = full-range (ถึง ~5 เมตร) จับคนที่ยืนอยู่ด้านหลังได้ดีกว่ารุ่น 0
        fd = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6)
        face_mesh, face_detection = fm, fd
        print("✅ โหลด MediaPipe สำเร็จ")
        try:   # ตรวจจับมือ (ใช้จับมือบังหน้า/บังกล้อง) ถ้าโหลดไม่ได้ ระบบอื่นยังทำงานต่อ
            hand_model = mp.solutions.hands.Hands(max_num_hands=4, model_complexity=0,
                                                   min_detection_confidence=0.5, min_tracking_confidence=0.5)
            print("✅ โหลดตัวตรวจจับมือสำเร็จ")
        except Exception as e:
            AI_STATUS["errors"].append(f"ตรวจมือ: {e}")
    except Exception as e:
        import traceback; traceback.print_exc()
        AI_FACE_ERROR = str(e); AI_STATUS["errors"].append(f"AI ใบหน้า: {e}")
    try:
        _ai_step(3, "กำลังโหลด PyTorch / YOLO (ตรวจจับมือถือ-หนังสือ)...")
        from ultralytics import YOLO
        _ai_step(4, "กำลังโหลดโมเดล YOLO...")
        m = None
        for name in YOLO_CANDIDATES:   # เลือกรุ่นที่แม่นที่สุดที่มีในโฟลเดอร์
            pth = os.path.join(BASE_DIR, name)
            if os.path.exists(pth): m = YOLO(pth); yolo_name = name; break
        if m is None:
            m = YOLO('yolov8n.pt'); yolo_name = 'yolov8n.pt'   # ultralytics จะดาวน์โหลดให้เอง (ต้องมีเน็ต)
        _ai_step(5, "กำลังอุ่นเครื่องโมเดล (รอบแรกช้าที่สุด)...")
        m.predict(np.zeros((480, 640, 3), np.uint8), verbose=False, imgsz=640)
        model = m
        print(f"✅ โหลด YOLO สำเร็จ ({yolo_name})")
    except Exception as e:
        import traceback; traceback.print_exc()
        AI_STATUS["errors"].append(f"YOLO: {e}")
    AI_STATUS["text"] = "พร้อมใช้งาน" if not AI_STATUS["errors"] else "โหลดบางส่วนไม่สำเร็จ"
    AI_READY.set()


def start_ai_loading():
    if not getattr(start_ai_loading, "_started", False):
        start_ai_loading._started = True
        threading.Thread(target=load_ai_models, daemon=True).start()

target_classes = ["cell phone", "book"]
YOLO_IMGSZ = 640          # เพิ่มเป็น 800/960 จะเห็นของชิ้นเล็กไกลๆ ดีขึ้น แต่ใช้ CPU มากขึ้น
HAND_PHONE_CONF = 0.22    # มือถือที่อยู่ในมือ (ทับกรอบมือ) ยอมรับความมั่นใจต่ำลงได้ เพราะมือบังบางส่วน


class ObjectDetector:
    """
    รัน YOLO ในเธรดแยก ไม่ให้ภาพกล้อง/หน้าจอกระตุก และตัดสินผลจากหลายเฟรมรวมกัน
    - มือถือ: ยอมรับความมั่นใจต่ำลง (0.35) เพราะเป็นของชิ้นเล็ก แต่ต้องเจอซ้ำ 2 ใน 4 รอบล่าสุด
      หรือเจอครั้งเดียวที่มั่นใจสูง (≥0.60)
    - คน (person): ใช้นับจำนวนคนเสริมกับการตรวจใบหน้า เผื่อคนที่ 2 หันหลัง/เห็นแค่ครึ่งตัว
    """
    CONF = {"cell phone": 0.35, "book": 0.45, "person": 0.50}
    STRONG = {"cell phone": 0.60, "book": 0.70}

    def __init__(self, yolo_model):
        self.model = yolo_model
        names = yolo_model.names if yolo_model else {}
        self.name_by_id = dict(names) if isinstance(names, dict) else dict(enumerate(names))
        self.conf = dict(self.CONF); self.strong = dict(self.STRONG)   # ปรับได้ตามความเข้ม
        self.hand_boxes = []; self._cmult = 1.0   # กรอบมือจากตัวตรวจมือ (พิกเซลเฟรม 640x480) ไว้ช่วยยืนยันมือถือ
        self.class_ids = [i for i, n in self.name_by_id.items() if n in self.CONF]
        self._frame = None; self._lock = threading.Lock(); self._event = threading.Event()
        self._stop = False
        self.history = collections.deque(maxlen=4)   # ผลของ 4 รอบล่าสุด
        self.found_obj = False; self.obj_label = ""; self.persons = 0; self.boxes = []
        self._thread = None

    def set_sensitivity(self, conf_mult):
        # conf_mult มาก = ต้องมั่นใจมากขึ้นถึงจะเตือน (ผ่อนปรน); น้อย = ไวขึ้น (เข้มงวด)
        self.conf = {k: min(0.9, v * conf_mult) for k, v in self.CONF.items()}
        self.strong = {k: min(0.95, v * conf_mult) for k, v in self.STRONG.items()}
        self._cmult = conf_mult

    def start(self):
        if self.model is None or (self._thread and self._thread.is_alive()): return
        self._stop = False; self.history.clear()
        self._thread = threading.Thread(target=self._run, daemon=True); self._thread.start()

    def stop(self):
        self._stop = True; self._event.set()
        self.found_obj = False; self.persons = 0; self.boxes = []

    def join(self, timeout=None):
        if self._thread and self._thread.is_alive(): self._thread.join(timeout)

    def submit(self, frame):
        if self.model is None: return
        with self._lock: self._frame = frame
        self._event.set()

    def _run(self):
        try:   # warm-up รอบแรกช้า ทำทิ้งไว้ก่อน
            self.model.predict(np.zeros((480, 640, 3), np.uint8), verbose=False, imgsz=640)
        except Exception: pass
        while not self._stop:
            self._event.wait(0.5); self._event.clear()
            with self._lock: frame, self._frame = self._frame, None
            if frame is None or self._stop: continue
            try: self._infer(frame)
            except Exception as ex: print(f"⚠️ YOLO error: {ex}")

    def _hand_phone_conf(self):
        return max(0.15, min(0.6, HAND_PHONE_CONF * self._cmult))

    def _in_hand(self, x1, y1, x2, y2):
        """กล่องวัตถุอยู่ในมือไหม: จุดกึ่งกลางอยู่ในกรอบมือที่ขยายออกไป 25% (มือมักบังมือถือบางส่วน)"""
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        for hx1, hy1, hx2, hy2 in list(self.hand_boxes):
            pw, ph = (hx2 - hx1) * 0.25, (hy2 - hy1) * 0.25
            if hx1 - pw <= cx <= hx2 + pw and hy1 - ph <= cy <= hy2 + ph: return True
        return False

    def _infer(self, frame):
        h, w = frame.shape[:2]; area = float(h * w)
        res = self.model.predict(frame, imgsz=YOLO_IMGSZ, conf=min(min(self.conf.values()), self._hand_phone_conf()), iou=0.5,
                                 classes=self.class_ids or None, verbose=False)
        hits = {}; strong = False; persons = 0; boxes = []
        for r in res:
            for b in r.boxes:
                name = self.name_by_id.get(int(b.cls[0]), ""); conf = float(b.conf[0])
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                thr = self.conf.get(name, 1.0)
                if name == "cell phone" and conf < thr and conf >= self._hand_phone_conf() and self._in_hand(x1, y1, x2, y2):
                    thr = conf   # มือถือในมือ: ผ่อนเกณฑ์ (ยังต้องเจอซ้ำ 2 ใน 4 รอบถึงจะนับ ไม่ใช่เจอครั้งเดียว)
                if conf < thr: continue
                if name == "person":
                    if (x2 - x1) * (y2 - y1) / area >= 0.03: persons += 1   # ตัดคนเล็กมากๆ (รูปบนผนัง/ไกลมาก)
                    continue
                hits[name] = max(hits.get(name, 0), conf)
                if conf >= self.strong.get(name, 1.0): strong = True
                boxes.append((name, conf, (x1, y1, x2, y2)))
        self.history.append(hits)
        votes = collections.Counter(n for hh in self.history for n in hh)
        confirmed = [n for n, c in votes.items() if c >= 2]
        self.found_obj = strong or bool(confirmed)
        labels = set(confirmed) | ({n for n, _, _ in boxes} if strong else set())
        self.obj_label = "มือถือ" if "cell phone" in labels else ("หนังสือ" if "book" in labels else "")
        self.persons = persons; self.boxes = boxes


class ViolationFilter:
    """
    กันการเตือนหลอกจากเฟรมเดียว: เหตุผลหนึ่งจะนับว่าเกิดจริงเมื่อพบต่อเนื่องเกินสัดส่วนที่กำหนดในช่วงเวลาหนึ่ง
    และมี hysteresis (เมื่อเตือนแล้วจะหายเมื่อสัดส่วนลดลงต่ำกว่าครึ่งหนึ่งของเกณฑ์) ป้องกันสถานะกะพริบ
    """
    RULES = {   # เหตุผล: (ช่วงเวลา วินาที, สัดส่วนเฟรมที่ต้องพบ)
        "ไม่พบใบหน้า": (1.0, 0.7),
        "พบหลายคน": (1.2, 0.6),
        "หันซ้าย": (1.0, 0.7), "หันขวา": (1.0, 0.7),
        "ก้มหน้า": (1.2, 0.7),
        "เหลือบตามองด้านข้าง": (1.5, 0.75),
        "คุย": (0.6, 0.5),
        "ก้มดูใต้โต๊ะ": (0.6, 0.5), "ก้มมองด้านข้าง": (0.6, 0.5),
        "ก้มหน้าซ้ำๆ": (0.5, 0.5), "ก้มหลบกล้อง": (0.6, 0.5), "ก้มนานผิดปกติ": (1.0, 0.6),
        "มือบังหน้า": (1.0, 0.6),
        "มือใกล้กล้อง": (1.0, 0.7),
        "กล้องถูกบัง": (1.0, 0.7),
    }

    def __init__(self):
        self.rules = dict(self.RULES)
        self.hist = {k: collections.deque() for k in self.RULES}; self.active = set()

    def set_sensitivity(self, ratio_mult):
        # ratio_mult มาก = ต้องพบต่อเนื่องนานขึ้นถึงเตือน (ผ่อนปรน); น้อย = เตือนไวขึ้น (เข้มงวด)
        self.rules = {k: (win, max(0.3, min(0.95, ratio * ratio_mult))) for k, (win, ratio) in self.RULES.items()}

    def reset(self):
        for d in self.hist.values(): d.clear()
        self.active.clear()

    def update(self, raw, now):
        out = []
        for k, (win, ratio) in self.rules.items():
            d = self.hist[k]; d.append((now, k in raw))
            while d and now - d[0][0] > win: d.popleft()
            if len(d) < 3: continue
            frac = sum(1 for _, v in d if v) / len(d)
            if frac >= ratio: self.active.add(k)
            elif frac < ratio / 2: self.active.discard(k)
            if k in self.active: out.append(k)
        return out


class FaceAnalyzer:
    """
    วิเคราะห์ใบหน้าจาก MediaPipe FaceMesh
    - มุมหัว (yaw/pitch) ด้วย solvePnP กับโมเดล 3 มิติของใบหน้า หน่วยเป็นองศาจริง
    - การเหลือบตา จากตำแหน่งม่านตา (iris) เทียบกับหัวตา/หางตา
    - การพูดคุย จากการที่ปากขยับเปิด-ปิดซ้ำๆ (ไม่ใช่แค่อ้าปากค้าง เช่น หาว)
    - ปรับเทียบ (calibrate) ท่าทางปกติของนักเรียนแต่ละคนก่อนเริ่มสอบ เช่น กล้องอยู่สูง/ต่ำกว่าหน้า
    """
    # MediaPipe ให้เลข landmark ตามฝั่งของภาพเสมอ: 33/61 อยู่ซ้ายภาพ, 263/291 อยู่ขวาภาพ (flip หรือไม่ก็ตาม)
    PNP_IDX = [1, 152, 33, 263, 61, 291]
    MODEL_3D = np.array([
        (0.0, 0.0, 0.0),          # ปลายจมูก
        (0.0, 330.0, 65.0),       # คาง
        (-225.0, -170.0, 135.0),  # หางตา (ซ้ายภาพ)
        (225.0, -170.0, 135.0),   # หางตา (ขวาภาพ)
        (-150.0, 150.0, 125.0),   # มุมปาก (ซ้ายภาพ)
        (150.0, 150.0, 125.0),    # มุมปาก (ขวาภาพ)
    ], dtype=np.float64)

    YAW_LIMIT = 25.0          # หันซ้าย/ขวาเกินกี่องศาจากท่าปกติ
    PITCH_DOWN_LIMIT = 15.0   # ก้มหน้าเกินกี่องศาจากท่าปกติ (เดิม 20)
    GAZE_LIMIT = 0.20         # ม่านตาเลื่อนจากตำแหน่งปกติเกินเท่าไร (0 = กลางตา, ±0.5 = ชนขอบตา)
    PRE_YAW_LIMIT = 30.0      # ก่อนเริ่มสอบ (ยังไม่ calibrate) ใช้เกณฑ์หลวมกว่า
    PRE_PITCH_LIMIT = 22.0    # ก่อนเริ่มสอบ (เดิม 35 สูงเกินไป ก้มชัดๆ ก็ไม่เตือน)
    SHOW_ANGLES = True        # แสดงมุมก้มที่วัดได้บนภาพกล้อง (ไว้ปรับเกณฑ์ ปิดได้เมื่อจูนเสร็จ)

    def __init__(self):
        self.rvec = None; self.tvec = None
        self.mar_hist = collections.deque()
        self.calib = collections.deque(maxlen=60)   # (yaw, pitch, gaze) ตอนนั่งพร้อมสอบ
        self.base = None                              # ค่าปกติที่ล็อกไว้ตอนเริ่มสอบ
        self.last = (0.0, 0.0, 0.0)
        self.head = HeadDownClassifier(self.PITCH_DOWN_LIMIT)   # แยกประเภทการก้มหน้า (ปกติ/เสี่ยง)
        self.head_label = ""
        self.sm_yaw = PoseSmoother(1.0, 0.05); self.sm_pitch = PoseSmoother(1.0, 0.05)   # กรองสัญญาณสั่น (หน่วง ~0.06 วินาที)
        self.sm_gaze = PoseSmoother(1.0, 2.0); self._prev_pose = None

    def set_sensitivity(self, angle_mult):
        # angle_mult มาก = ยอมให้ขยับได้มากขึ้นก่อนเตือน (ผ่อนปรน); น้อย = เข้มงวด
        self.YAW_LIMIT = FaceAnalyzer.YAW_LIMIT * angle_mult
        self.PITCH_DOWN_LIMIT = FaceAnalyzer.PITCH_DOWN_LIMIT * angle_mult
        self.GAZE_LIMIT = FaceAnalyzer.GAZE_LIMIT * angle_mult
        self.head.set_limit(self.PITCH_DOWN_LIMIT)

    def reset(self):
        self.rvec = self.tvec = None; self.mar_hist.clear(); self.calib.clear(); self.base = None
        self.head.reset(); self.head_label = ""
        self.sm_yaw.reset(); self.sm_pitch.reset(); self.sm_gaze.reset(); self._prev_pose = None

    def lock_baseline(self):
        if len(self.calib) >= 10:
            arr = np.array(self.calib)
            yaw, pitch, gaze = np.median(arr, axis=0)
        else:
            yaw, pitch, gaze = 0.0, 0.0, 0.0
        # ไม่ให้ calibrate ไปเป็นท่าที่ผิดปกติมากเกินไป
        self.base = (float(np.clip(yaw, -12, 12)), float(np.clip(pitch, -20, 20)), float(np.clip(gaze, -0.15, 0.15)))

    @staticmethod
    def _pt(lm, i, w, h): return np.array([lm[i].x * w, lm[i].y * h])

    def head_pose(self, lm, w, h):
        img_pts = np.array([self._pt(lm, i, w, h) for i in self.PNP_IDX], dtype=np.float64)
        cam = np.array([[w, 0, w / 2.0], [0, w, h / 2.0], [0, 0, 1]], dtype=np.float64)
        dist = np.zeros((4, 1))
        if self.rvec is None:
            rvec = np.zeros((3, 1)); tvec = np.array([[0.0], [0.0], [1000.0]])
        else:
            rvec, tvec = self.rvec.copy(), self.tvec.copy()
        ok, rvec, tvec = cv2.solvePnP(self.MODEL_3D, img_pts, cam, dist, rvec, tvec,
                                      useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok or tvec[2, 0] <= 0:
            self.rvec = self.tvec = None; return None
        self.rvec, self.tvec = rvec, tvec
        angles = cv2.RQDecomp3x3(cv2.Rodrigues(rvec)[0])[0]   # องศาอยู่แล้ว ไม่ต้องคูณ 360
        pitch = ((angles[0] + 180) % 360) - 180               # + = ก้มหน้า
        yaw = angles[1]                                       # + = หันไปทางซ้ายของนักเรียน
        return float(yaw), float(pitch)

    def gaze(self, lm, w, h):
        """คืนค่า -0.5..0.5: ลบ = ม่านตาไปทางซ้ายภาพ, บวก = ขวาภาพ, None ถ้าหลับตา/ไม่มีข้อมูลม่านตา"""
        if len(lm) < 478: return None
        vals = []
        for iris, a, b, top, bot in ((468, 33, 133, 159, 145), (473, 362, 263, 386, 374)):
            pa, pb = self._pt(lm, a, w, h), self._pt(lm, b, w, h)
            width = np.linalg.norm(pb - pa)
            if width < 1: continue
            if np.linalg.norm(self._pt(lm, top, w, h) - self._pt(lm, bot, w, h)) / width < 0.15: continue  # กะพริบตา
            pi = self._pt(lm, iris, w, h)
            left, right = (pa, pb) if pa[0] < pb[0] else (pb, pa)
            vals.append(float(np.dot(pi - left, right - left) / (width ** 2)) - 0.5)
        return float(np.mean(vals)) if vals else None

    def talking(self, lm, w, h, now):
        mouth_w = np.linalg.norm(self._pt(lm, 61, w, h) - self._pt(lm, 291, w, h))
        if mouth_w < 1: return False
        mar = np.linalg.norm(self._pt(lm, 13, w, h) - self._pt(lm, 14, w, h)) / mouth_w
        d = self.mar_hist; d.append((now, mar))
        while d and now - d[0][0] > 2.0: d.popleft()
        if len(d) < 8: return False
        vals = np.array([v for _, v in d])
        if vals.max() - vals.min() < 0.12: return False
        # นับจำนวนครั้งที่ปากเปิด→ปิด ข้ามเส้นกลาง (พูด = ขยับถี่ ๆ, หาว = เปิดค้างครั้งเดียว)
        mid = (vals.max() + vals.min()) / 2
        crossings = int(np.sum(np.diff((vals > mid).astype(int)) != 0))
        return crossings >= 4

    def analyze(self, lm, w, h, now, exam_active):
        reasons = set()
        pose = self.head_pose(lm, w, h)
        g = self.gaze(lm, w, h)
        if pose is not None:
            yaw, pitch = pose
            yaw, pitch = self.sm_yaw(yaw, now), self.sm_pitch(pitch, now)   # ลดการสั่นของ landmark
            if g is not None: g = self.sm_gaze(g, now)
            self.last = (yaw, pitch, g if g is not None else 0.0)
            if exam_active and self.base is not None:
                by, bp, bg = self.base
                if yaw - by > self.YAW_LIMIT: reasons.add("หันซ้าย")
                elif yaw - by < -self.YAW_LIMIT: reasons.add("หันขวา")
                # ก้มหน้า: แยกประเภทตามระยะเวลา/ความลึก/ทิศทาง/ความถี่ (ก้มเขียนปกติจะไม่เตือน)
                head_reasons, self.head_label = self.head.update(pitch - bp, yaw - by, now)
                reasons |= head_reasons
                # เช็กเหลือบตาเฉพาะตอนหัวยังตรงอยู่ (ถ้าหันหัวไปแล้วจะถูกจับด้วยเหตุผลอื่นอยู่แล้ว)
                if g is not None and not reasons and abs(g - bg) > self.GAZE_LIMIT:
                    reasons.add("เหลือบตามองด้านข้าง")
            else:
                self.head.reset(); self.head_label = ""
                if yaw > self.PRE_YAW_LIMIT: reasons.add("หันซ้าย")
                elif yaw < -self.PRE_YAW_LIMIT: reasons.add("หันขวา")
                if pitch > self.PRE_PITCH_LIMIT: reasons.add("ก้มหน้า")
                # เก็บท่าปกติไว้ calibrate เฉพาะตอนที่หัวนิ่ง (กำลังขยับอยู่จะได้ค่าอ้างอิงเพี้ยนไปทั้งการสอบ)
                still = self._prev_pose is not None and abs(yaw - self._prev_pose[0]) < 1.2 and abs(pitch - self._prev_pose[1]) < 1.2
                self._prev_pose = (yaw, pitch)
                if not reasons and g is not None and still:
                    self.calib.append((yaw, pitch, g))
        if self.talking(lm, w, h, now): reasons.add("คุย")
        return reasons


def enhance_low_light(frame):
    """ห้องมืด → MediaPipe/YOLO หาหน้าและวัตถุไม่เจอ: เพิ่มความสว่างเฉพาะตอนภาพมืด (CLAHE บนช่อง L)"""
    gray_mean = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
    if gray_mean >= 80: return frame
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB); l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


# ================= Helper: Upload =================
def upload_to_teacher(filepath):
    if not filepath or not os.path.exists(filepath): return
    fname = os.path.basename(filepath)
    # กรณีเครื่องนี้เป็นเซิร์ฟเวอร์เอง (สอบ+คุมในเครื่องเดียว) ก็อปเข้าโฟลเดอร์รับไฟล์ตรงๆ ไม่ต้องผ่านเน็ต
    if globals().get("server_started"):
        try:
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            dest = os.path.join(UPLOAD_FOLDER, fname)
            if os.path.abspath(dest) != os.path.abspath(filepath):
                shutil.copy2(filepath, dest)
            print(f"📥 เก็บหลักฐานในเครื่อง: {fname}")
            return
        except Exception as ex:
            print(f"⚠️ ก็อปหลักฐานไม่สำเร็จ: {ex}")
    # ที่อยู่เครื่องครู: ใช้ที่ API ล็อกอินได้จริงก่อน (รวม 127.0.0.1 ที่เป็นค่าสำรอง) แล้วค่อยลองที่อื่น
    bases = []
    client = globals().get("api")
    if client is not None and getattr(client, "base", None): bases.append(client.base)
    b = media.teacher_base_url()
    if b and b not in bases: bases.append(b)
    for extra in ("http://127.0.0.1:5000",):
        if extra not in bases: bases.append(extra)
    def send():
        for attempt in range(3):   # เน็ตนอกวงอาจหลุดง่าย ลองใหม่สูงสุด 3 ครั้ง
            last = None
            for base in bases:
                try:
                    with open(filepath, 'rb') as fh:
                        r = media._http().post(f"{base}/upload", files={'file': (fname, fh)}, timeout=(10, 1800))
                    r.raise_for_status(); print(f"📤 ส่งไฟล์สำเร็จ: {fname}"); return
                except Exception as ex:
                    last = ex
            print(f"⚠️ ส่งไฟล์ให้อาจารย์ไม่สำเร็จ (ครั้งที่ {attempt + 1}): {last}")
            time.sleep(5)
        print(f"❌ ส่งไฟล์ {fname} ไม่สำเร็จ — ยังเก็บไว้ในโฟลเดอร์ evidence ของเครื่องนักเรียน")
    threading.Thread(target=send, daemon=True).start()

# ================= Media widgets (รูปภาพ / เสียงประกอบข้อสอบ) =================
def _short(name, n=20):
    return name if len(name) <= n else name[:n - 1] + "…"

class AudioControl(ctk.CTkFrame):
    """ปุ่มเล่น/หยุดเสียง เล่นได้ทีละไฟล์ทั้งโปรแกรม"""
    _active = None

    def __init__(self, parent, path, title="เสียงประกอบ", compact=False):
        super().__init__(parent, fg_color=BG_INPUT, corner_radius=10, border_width=1, border_color=BORDER_SOFT)
        self.path = path; self.playing = False
        self.btn = ctk.CTkButton(self, text="▶", width=32 if compact else 40, height=28 if compact else 34,
                                 corner_radius=8, fg_color=CYAN_DIM, hover_color=CYAN, text_color=WHITE_TXT,
                                 command=self.toggle)
        self.btn.pack(side="left", padx=6, pady=4)
        ctk.CTkLabel(self, text=f"🎧 {title}", font=FONT_SMALL if compact else FONT_SMALL_B,
                     text_color=TEXT_MAIN).pack(side="left", padx=(0, 10))

    def toggle(self):
        if self.playing: self.stop(); return
        if AudioControl._active is not None and AudioControl._active is not self:
            AudioControl._active.stop()
        if not media.PLAYER.play(self.path):
            messagebox.showerror("เล่นเสียงไม่ได้",
                                 "ไม่สามารถเล่นไฟล์เสียงได้\nตรวจสอบว่าติดตั้ง pygame แล้ว (pip install pygame)\n\n"
                                 + media.PLAYER.error)
            return
        AudioControl._active = self; self.playing = True; self.btn.configure(text="⏹"); self._poll()

    def _poll(self):
        if not self.playing or not self.winfo_exists(): return
        if not media.PLAYER.is_busy(): self._reset(); return
        self.after(300, self._poll)

    def _reset(self):
        self.playing = False
        if AudioControl._active is self: AudioControl._active = None
        try: self.btn.configure(text="▶")
        except Exception: pass

    def stop(self):
        media.PLAYER.stop(); self._reset()

    def destroy(self):
        if self.playing: media.PLAYER.stop(); self.playing = False
        if AudioControl._active is self: AudioControl._active = None
        super().destroy()

def open_zoom(parent, path):
    """คลิกรูปเพื่อดูขยาย (คลิกอีกครั้งเพื่อปิด)"""
    try: img = Image.open(path).convert("RGBA")
    except Exception: return
    win = ctk.CTkToplevel(parent); win.title("ดูรูปภาพ"); win.configure(fg_color=BG_APP)
    w, h = img.size; sc = min(900 / w, 700 / h, 1.0)
    size = (max(1, int(w * sc)), max(1, int(h * sc)))
    cimg = ctk.CTkImage(light_image=img, dark_image=img, size=size)
    lbl = ctk.CTkLabel(win, text="", image=cimg); lbl.image = cimg; lbl.pack(padx=12, pady=12)
    lbl.bind("<Button-1>", lambda e: win.destroy())
    win.geometry(f"{size[0] + 24}x{size[1] + 24}"); win.attributes('-topmost', True); win.focus()

def make_image_label(parent, path, max_w=380, max_h=240, zoom=True):
    try: img = Image.open(path).convert("RGBA")
    except Exception: return ctk.CTkLabel(parent, text="⚠️ เปิดรูปไม่ได้", text_color=RED_ALERT)
    w, h = img.size; sc = min(max_w / w, max_h / h, 1.0)
    cimg = ctk.CTkImage(light_image=img, dark_image=img, size=(max(1, int(w * sc)), max(1, int(h * sc))))
    lbl = ctk.CTkLabel(parent, text="", image=cimg, corner_radius=8); lbl.image = cimg
    if zoom: lbl.bind("<Button-1>", lambda e, p=path: open_zoom(parent, p))
    return lbl

class MediaPicker(ctk.CTkFrame):
    """ช่องแนบไฟล์ 1 ช่อง (ฝั่งอาจารย์) kinds = ('image',) / ('audio',) / ('image','audio')"""
    _IMG = "*.png *.jpg *.jpeg *.gif *.bmp *.webp"
    _AUD = "*.mp3 *.wav *.ogg *.flac"

    def __init__(self, parent, kinds=("image", "audio"), label="แนบไฟล์", compact=False):
        super().__init__(parent, fg_color="transparent")
        self.kinds = kinds; self.path = None; self.compact = compact
        self.btn = ctk.CTkButton(self, text="📎" if compact else label, width=36 if compact else 150,
                                 height=34 if compact else 32, corner_radius=8, fg_color=CYAN_DIM,
                                 hover_color=CYAN, text_color=TEXT_MAIN, font=FONT_SMALL_B, command=self.pick)
        self.btn.pack(side="left")
        # ต้องกำหนด width/height เล็กๆ ไม่งั้น CTkFrame ว่างจะสูง 200px (ค่าเริ่มต้น) ทำให้ช้อยส์แต่ละข้อห่างกันมาก
        self.holder = ctk.CTkFrame(self, fg_color="transparent", width=1, height=1); self.holder.pack(side="left", padx=(8, 0))
        self.clear_btn = ctk.CTkButton(self, text="✕", width=26, height=26, corner_radius=6, fg_color="transparent",
                                       hover_color=RED_DIM, text_color=RED_ALERT, command=self.clear)
        self._render()

    def pick(self):
        ft = []
        if len(self.kinds) > 1: ft.append(("รูปภาพและเสียง", f"{self._IMG} {self._AUD}"))
        if "image" in self.kinds: ft.append(("รูปภาพ", self._IMG))
        if "audio" in self.kinds: ft.append(("ไฟล์เสียง", self._AUD))
        p = filedialog.askopenfilename(title="เลือกไฟล์", filetypes=ft)
        if p: self.set(p)

    def set(self, path):
        try: kind = media.validate(path)
        except media.MediaError as e: messagebox.showerror("แนบไฟล์ไม่ได้", str(e)); return False
        if kind not in self.kinds:
            messagebox.showerror("แนบไฟล์ไม่ได้", "ช่องนี้รองรับเฉพาะ " +
                                 " / ".join({"image": "รูปภาพ", "audio": "ไฟล์เสียง"}[k] for k in self.kinds))
            return False
        self.path = path; self._render(); return True

    def clear(self): self.path = None; self._render()
    def get(self): return self.path

    def _render(self):
        for w in self.holder.winfo_children(): w.destroy()
        if not self.path:
            self.clear_btn.pack_forget()
            if not self.compact:
                ctk.CTkLabel(self.holder, text="ยังไม่ได้แนบ", font=FONT_SMALL, text_color=TEXT_MUTED).pack()
            return
        name = os.path.basename(self.path)
        if media.kind_of(self.path) == "image":
            try:
                im = Image.open(self.path).convert("RGBA"); im.thumbnail((34, 34))
                c = ctk.CTkImage(light_image=im, dark_image=im, size=im.size)
                t = ctk.CTkLabel(self.holder, text="", image=c); t.image = c; t.pack(side="left")
            except Exception: pass
            if not self.compact:
                ctk.CTkLabel(self.holder, text=_short(name), font=FONT_SMALL, text_color=TEXT_MAIN).pack(side="left", padx=6)
        else:
            AudioControl(self.holder, self.path, title=_short(name, 16), compact=True).pack(side="left")
        self.clear_btn.pack(side="left", padx=6)

# ================= Hero panel (แผงแนะนำระบบ ฝั่งซ้ายของหน้า Login/Register) =================
HERO_BG = "#0a111d"

def build_hero(parent):
    """แผงแนะนำระบบ กินพื้นที่ครึ่งซ้ายของหน้า ไอคอนทุกตัวอยู่ในช่องขนาดเท่ากัน ข้อความชิดซ้ายเส้นเดียวกัน"""
    hero = ctk.CTkFrame(parent, corner_radius=0, fg_color=HERO_BG)
    hero.place(relx=0, rely=0, relwidth=0.5, relheight=1)
    box = ctk.CTkFrame(hero, fg_color="transparent")
    box.place(relx=0.5, rely=0.5, anchor="center")

    # โลโก้: ไอคอนอยู่กึ่งกลางกรอบเสมอ
    badge = ctk.CTkFrame(box, width=96, height=96, corner_radius=24, fg_color=BG_PANEL_2,
                         border_width=2, border_color=CYAN)
    badge.pack(pady=(0, 18)); badge.pack_propagate(False)
    ctk.CTkLabel(badge, text="🛡️", font=("Arial", 38)).place(relx=0.5, rely=0.5, anchor="center")

    ctk.CTkLabel(box, text="AI PROCTOR", font=("Arial", 44, "bold"), text_color=WHITE_TXT).pack()
    ctk.CTkLabel(box, text="ระบบคุมสอบออนไลน์อัจฉริยะด้วย AI", font=FONT_BODY,
                 text_color=TEXT_MUTED).pack(pady=(6, 34))

    # รายการฟีเจอร์: คอลัมน์ไอคอนกว้างเท่ากัน + ข้อความชิดซ้ายเหมือนกันทุกแถว
    features = [
        ("👁", "ตรวจจับใบหน้าและพฤติกรรมแบบเรียลไทม์"),
        ("🎥", "บันทึกหลักฐานหน้าจอและกล้องอัตโนมัติ"),
        ("📊", "รายงานผลคะแนนและสถานะนักเรียนทันที"),
    ]
    feat = ctk.CTkFrame(box, fg_color="transparent"); feat.pack()
    feat.grid_columnconfigure(0, minsize=44)
    for r, (ico, txt) in enumerate(features):
        chip = ctk.CTkFrame(feat, width=40, height=40, corner_radius=12, fg_color=BG_PANEL_2,
                            border_width=1, border_color=BORDER_SOFT)
        chip.grid(row=r, column=0, pady=9, sticky="w"); chip.grid_propagate(False)
        ctk.CTkLabel(chip, text=ico, font=("Arial", 18)).place(relx=0.5, rely=0.5, anchor="center")
        ctk.CTkLabel(feat, text=txt, font=FONT_BODY, text_color="#cbd5e1", anchor="w",
                     justify="left").grid(row=r, column=1, padx=(14, 0), pady=9, sticky="w")
    return hero

# ================= Login & Register =================
class LoginPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=BG_APP); self.controller = controller
        set_page_background(self)
        build_hero(self)

        # --- outer glow wrapper gives the card a soft, layered depth ---
        glow = ctk.CTkFrame(self, width=460, height=600, corner_radius=24, fg_color=CYAN_DIM)
        glow.place(relx=0.75, rely=0.5, anchor="center")
        box = ctk.CTkFrame(glow, width=452, height=592, corner_radius=20, fg_color=BG_PANEL,
                            border_width=1, border_color=BORDER_SOFT)
        box.place(relx=0.5, rely=0.5, anchor="center")
        box.pack_propagate(False)

        # --- brand / header ---
        badge = ctk.CTkFrame(box, width=64, height=64, corner_radius=32, fg_color=BG_PANEL_2,
                              border_width=2, border_color=CYAN)
        badge.pack(pady=(38, 12)); badge.pack_propagate(False)
        ctk.CTkLabel(badge, text="🛡️", font=("Arial", 26)).place(relx=0.5, rely=0.5, anchor="center")
        ctk.CTkLabel(box, text="เข้าสู่ระบบ", font=FONT_H1, text_color=WHITE_TXT).pack()
        ctk.CTkLabel(box, text="AI Proctored Exam System", font=FONT_SUB, text_color=TEXT_MUTED).pack(pady=(2, 26))

        field_w = 360
        ctk.CTkLabel(box, text="ชื่อผู้ใช้", font=FONT_SMALL_B, text_color=TEXT_MUTED,
                     anchor="w").pack(fill="x", padx=(46, 46))
        self.u_e = ctk.CTkEntry(box, placeholder_text="Username", font=FONT_BODY, height=46, width=field_w,
                                 corner_radius=10, fg_color=BG_INPUT, border_color=BORDER_SOFT,
                                 border_width=1, text_color=TEXT_MAIN)
        self.u_e.pack(pady=(4, 14))
        ctk.CTkLabel(box, text="รหัสผ่าน", font=FONT_SMALL_B, text_color=TEXT_MUTED,
                     anchor="w").pack(fill="x", padx=(46, 46))
        self.p_e = ctk.CTkEntry(box, placeholder_text="Password", show="•", font=FONT_BODY, height=46, width=field_w,
                                 corner_radius=10, fg_color=BG_INPUT, border_color=BORDER_SOFT,
                                 border_width=1, text_color=TEXT_MAIN)
        self.p_e.pack(pady=(4, 4))
        self.p_e.bind("<Return>", lambda e: self.login())

        ctk.CTkButton(box, text="เข้าสู่ระบบ", command=self.login, font=FONT_BOLD, height=48, width=field_w,
                      corner_radius=10, fg_color=CYAN, hover_color=CYAN_DIM, text_color="#04121a").pack(pady=(22, 14))

        sep = ctk.CTkFrame(box, width=field_w, height=1, fg_color=BORDER_SOFT); sep.pack(pady=(0, 14))

        ctk.CTkButton(box, text="สมัครสมาชิกใหม่", command=lambda: controller.show_frame("RegisterPage"), font=FONT_BOLD,
                      corner_radius=10, fg_color="transparent", hover_color=BG_PANEL_2, border_width=2, border_color=TEAL,
                      text_color=TEAL, height=42, width=field_w).pack()
        ctk.CTkButton(self, text="⚙️  ตั้งค่า IP อาจารย์", command=self.set_server_ip, fg_color=BG_PANEL_2,
                      corner_radius=10, hover_color=BG_PANEL, border_width=1, border_color=BORDER_SOFT,
                      text_color=TEXT_MUTED, font=FONT_SMALL, height=32,
                      width=170).place(relx=0.97, rely=0.96, anchor="se")

    def set_server_ip(self):
        curr = ""; 
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as f: curr = f.read()
        ip = ctk.CTkInputDialog(text=f"ที่อยู่เครื่องอาจารย์ (ปัจจุบัน: {curr.strip() or '-'})\n"
                                     f"• ในวงเดียวกัน: IP เช่น 172.16.0.15\n"
                                     f"• นอกวง (Tailscale): IP 100.x.x.x\n"
                                     f"• นอกวง (Cloudflare): https://xxxx.trycloudflare.com",
                                title="ตั้งค่าเครื่องอาจารย์").get_input()
        if ip:
            ip = ip.strip()
            with open(CONFIG_FILE, "w") as f: f.write(ip)
            api.base = None
            try:
                r = media._http().get(f"{media.teacher_base_url()}/api/ping", timeout=5); r.raise_for_status()
                messagebox.showinfo("เชื่อมต่อสำเร็จ", f"บันทึก {ip} แล้ว และติดต่อเครื่องอาจารย์ได้ ✅")
            except Exception as ex:
                messagebox.showwarning("บันทึกแล้ว แต่ยังติดต่อไม่ได้",
                                       f"บันทึก {ip} แล้ว แต่ติดต่อเครื่องอาจารย์ไม่ได้ ({type(ex).__name__})\n"
                                       "ตรวจว่าอาจารย์ล็อกอินค้างไว้, IP ถูกต้อง และไม่โดนไฟร์วอลล์บล็อก")

    def login(self):
        user, pwd = self.u_e.get().strip(), self.p_e.get()
        if not user or not pwd: messagebox.showerror("ผิดพลาด", "กรอกชื่อผู้ใช้และรหัสผ่าน"); return
        # 1) อาจารย์: บัญชีอยู่ในฐานข้อมูลของเครื่องนี้ (เครื่องอาจารย์ = เซิร์ฟเวอร์)
        res = db.login_user(user, pwd)
        if res and res[1] == 'teacher':
            uid, role, app, tid = res
            self.controller.user_id = uid; self.controller.username = user; self.controller.role = role
            self.controller.teacher_id = uid
            self.controller.show_frame("TeacherDashboard"); return
        # 2) นักเรียน: ล็อกอินผ่านเซิร์ฟเวอร์ของอาจารย์เสมอ (ในวงหรือนอกวงก็ได้)
        try:
            d = api.login(user, pwd)
        except ApiError as ex:
            messagebox.showerror("เชื่อมต่อไม่ได้", str(ex)); return
        if not d.get("ok"):
            msg = {"pending": ("รออนุมัติ", "รออาจารย์อนุมัติ"),
                   "teacher_remote": ("ผิดพลาด", "บัญชีอาจารย์ต้องล็อกอินที่เครื่องอาจารย์เท่านั้น")}.get(
                       d.get("error"), ("ผิดพลาด", "ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง"))
            messagebox.showwarning(*msg); return
        self.controller.user_id = d["user_id"]; self.controller.username = user; self.controller.role = "student"
        self.controller.teacher_id = d["teacher_id"]
        self.controller.show_frame("StudentExamPage")

class RegisterPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=BG_APP); self.controller = controller
        set_page_background(self)
        build_hero(self)
        glow = ctk.CTkFrame(self, width=540, height=700, corner_radius=24, fg_color=TEAL_DIM)
        glow.place(relx=0.75, rely=0.5, anchor="center")
        box = ctk.CTkFrame(glow, width=532, height=692, corner_radius=20, fg_color=BG_PANEL,
                            border_width=1, border_color=BORDER_SOFT)
        box.place(relx=0.5, rely=0.5, anchor="center")

        field_w = 400
        ctk.CTkLabel(box, text="✨ สมัครสมาชิก", font=FONT_H1, text_color=WHITE_TXT).pack(pady=(28, 2))
        ctk.CTkLabel(box, text="สร้างบัญชีเพื่อเริ่มใช้งานระบบ", font=FONT_SUB, text_color=TEXT_MUTED).pack(pady=(0, 20))

        self.role = ctk.StringVar(value="student")
        seg = ctk.CTkSegmentedButton(box, values=["นักเรียน", "อาจารย์"], width=field_w, height=42,
                                      corner_radius=10, fg_color=BG_INPUT, selected_color=CYAN,
                                      selected_hover_color=CYAN_DIM, unselected_color=BG_INPUT,
                                      unselected_hover_color=BG_PANEL_2, text_color=TEXT_MAIN,
                                      command=self.on_role_change)
        seg.set("นักเรียน"); seg.pack(pady=(0, 18)); self.seg = seg

        def field(label, ph, show=None):
            ctk.CTkLabel(box, text=label, font=FONT_SMALL_B, text_color=TEXT_MUTED, anchor="w").pack(
                fill="x", padx=(66, 66))
            e = ctk.CTkEntry(box, placeholder_text=ph, show=show, width=field_w, height=42, corner_radius=10,
                              fg_color=BG_INPUT, border_color=BORDER_SOFT, border_width=1, text_color=TEXT_MAIN)
            e.pack(pady=(4, 12))
            return e

        self.u = field("ชื่อผู้ใช้", "Username")
        self.p = field("รหัสผ่าน", "Password", show="•")
        self.code_lbl = ctk.CTkLabel(box, text="รหัสห้องเรียน", font=FONT_SMALL_B, text_color=TEXT_MUTED, anchor="w")
        self.code_lbl.pack(fill="x", padx=(66, 66))
        self.code = ctk.CTkEntry(box, width=field_w, height=42, corner_radius=10, fg_color=BG_INPUT,
                                  border_color=BORDER_SOFT, border_width=1, text_color=TEXT_MAIN)
        self.code.pack(pady=(4, 12))
        self.sys_lbl = ctk.CTkLabel(box, text="รหัสผ่านผู้ดูแลระบบ", font=FONT_SMALL_B, text_color=TEXT_MUTED, anchor="w")
        self.sys = ctk.CTkEntry(box, placeholder_text="Admin Pass", show="•", width=field_w, height=42, corner_radius=10,
                                fg_color=BG_INPUT, border_color=BORDER_SOFT, border_width=1, text_color=TEXT_MAIN)

        ctk.CTkButton(box, text="สมัครสมาชิก", command=self.reg, height=46, width=field_w, corner_radius=10,
                      font=FONT_BOLD, fg_color=TEAL, hover_color=TEAL_DIM, text_color="#04121a").pack(pady=(20, 10))
        ctk.CTkButton(box, text="← กลับไปเข้าสู่ระบบ", command=lambda: controller.show_frame("LoginPage"),
                      corner_radius=10, fg_color="transparent", hover_color=BG_PANEL_2, text_color=TEXT_MUTED,
                      height=36, width=field_w).pack()
        self.upd()

    def on_role_change(self, value):
        self.role.set("student" if value == "นักเรียน" else "teacher"); self.upd()

    def upd(self):
        if self.role.get() == "student":
            self.code_lbl.configure(text="รหัสห้องเรียน"); self.sys_lbl.pack_forget(); self.sys.pack_forget()
        else:
            self.code_lbl.configure(text="ตั้งรหัสห้องเรียน")
            self.sys_lbl.pack(fill="x", padx=(66, 66), after=self.code)
            self.sys.pack(pady=(4, 12), after=self.sys_lbl)
    def reg(self):
        u, p, c, r = self.u.get(), self.p.get(), self.code.get(), self.role.get()
        if not u or not p or not c: messagebox.showerror("ผิดพลาด", "กรอกไม่ครบ"); return
        if r == "teacher":
            if self.sys.get() != "ADMIN_PASS": messagebox.showerror("ผิดพลาด", "รหัสระบบผิด"); return
            res = db.register_teacher(u, p, c)
        else:
            try: res = api.call("/api/register", {"username": u, "password": p, "class_code": c}).get("result")
            except ApiError as ex: messagebox.showerror("เชื่อมต่อไม่ได้", str(ex)); return
        if res == "success": messagebox.showinfo("สำเร็จ", "สมัครสำเร็จ"); self.controller.show_frame("LoginPage")
        else: messagebox.showerror("ผิดพลาด", "ข้อมูลซ้ำหรือผิด")

# ================= Student Exam Page =================
# ================= ระบบล็อกเครื่องระหว่างสอบ (Exam Lockdown) =================
# ฟีเจอร์กันทุจริตฝั่งนักเรียน: บล็อกคีย์ลัด, ตรวจจอแยก, ตรวจโปรแกรมรีโมท, ล้างคลิปบอร์ด
# ทั้งหมดทำงานเฉพาะ "ระหว่างสอบ" และถูกยกเลิกทันทีเมื่อส่งข้อสอบ/ออกจากการสอบ
# ใช้ได้เต็มที่บน Windows; บนระบบอื่นจะข้ามเฉพาะส่วนที่ทำไม่ได้ โปรแกรมยังทำงานปกติ

# คีย์ลัดที่บล็อกระหว่างสอบ (สลับหน้าจอ/คัดลอก-วาง/แคปหน้าจอ/เปิดเมนู)
BLOCKED_HOTKEYS = [
    "alt+tab", "alt+shift+tab", "alt+esc", "ctrl+esc",
    "left windows", "right windows", "windows",
    "ctrl+c", "ctrl+v", "ctrl+x", "ctrl+insert", "shift+insert",
    "print screen", "alt+print screen", "ctrl+p",
    "alt+f4", "ctrl+w", "f11", "f12",
    "ctrl+shift+esc",          # Task Manager (บล็อกได้บางเครื่องเท่านั้น)
    "ctrl+t", "ctrl+n", "ctrl+tab",
]

# ชื่อโปรแกรมรีโมท/สื่อสารที่ไม่ควรเปิดค้างระหว่างสอบ (ตรวจจากชื่อ process)
REMOTE_APPS = {
    "anydesk": "AnyDesk", "teamviewer": "TeamViewer", "rustdesk": "RustDesk",
    "ultraviewer": "UltraViewer", "ammyy": "Ammyy Admin", "vnc": "VNC",
    "tightvnc": "TightVNC", "realvnc": "RealVNC", "vncserver": "VNC",
    "chrome_remote_desktop": "Chrome Remote Desktop", "remotepc": "RemotePC",
    "discord": "Discord", "zoom": "Zoom", "obs64": "OBS", "obs32": "OBS", "obs": "OBS",
    "msteams": "Microsoft Teams", "teams": "Microsoft Teams",
    "skype": "Skype", "parsecd": "Parsec", "parsec": "Parsec",
    "gotomeeting": "GoToMeeting", "quickassist": "Windows Quick Assist",
}


class ExamLockdown:
    """
    รวมมาตรการล็อกเครื่องระหว่างสอบไว้ในที่เดียว
    on_alert(reason) จะถูกเรียกเมื่อพบเรื่องผิดปกติ (เช่น เจอจอที่ 2, เปิดโปรแกรมรีโมท)
    เพื่อให้หน้าสอบบันทึกเป็นพฤติกรรมเสี่ยงและส่งให้อาจารย์เหมือนกับการหันหน้า/เจอมือถือ
    """
    def __init__(self, on_alert=None):
        self.on_alert = on_alert or (lambda reason: None)
        self.active = False
        self._kb = None
        self._kb_hooked = False
        self.available_kb = False
        self.notes = []          # สรุปว่ามาตรการไหนเปิดได้/ไม่ได้ (ไว้โชว์ให้นักเรียนทราบ)
        try:
            import keyboard
            self._kb = keyboard
            self.available_kb = True
        except Exception as ex:
            self.notes.append(f"บล็อกคีย์ลัดใช้ไม่ได้ ({type(ex).__name__}) — ติดตั้ง: pip install keyboard และรันด้วยสิทธิ์ Administrator")

    # ---------- คลิปบอร์ด ----------
    def clear_clipboard(self):
        """ล้างข้อความที่เตรียมไว้ในคลิปบอร์ด กันเพสต์คำตอบที่เตรียมมา"""
        try:
            import tkinter as _tk
            r = _tk._default_root
            if r is not None:
                r.clipboard_clear(); r.clipboard_append(""); r.update_idletasks()
                return True
        except Exception:
            pass
        try:
            if sys.platform.startswith("win"):
                subprocess.run("cmd /c echo off | clip", shell=True, timeout=5)
                return True
        except Exception:
            pass
        return False

    # ---------- คีย์ลัด ----------
    def _install_hotkeys(self):
        if not self.available_kb or self._kb_hooked:
            return
        blocked = 0
        for combo in BLOCKED_HOTKEYS:
            try:
                # suppress=True = กลืนคีย์ไม่ให้ส่งต่อไปที่ Windows/โปรแกรมอื่น
                self._kb.add_hotkey(combo, lambda c=combo: self._on_blocked_key(c), suppress=True)
                blocked += 1
            except Exception:
                pass  # บางคีย์ (เช่น Ctrl+Alt+Del) ระบบปฏิบัติการไม่ยอมให้บล็อก — ข้ามไป
        self._kb_hooked = blocked > 0
        if self._kb_hooked:
            self.notes.append(f"บล็อกคีย์ลัดแล้ว {blocked} ชุด")
        else:
            self.notes.append("บล็อกคีย์ลัดไม่สำเร็จ — ลองรันโปรแกรมด้วยสิทธิ์ Administrator")

    def _on_blocked_key(self, combo):
        self.on_alert(f"กดปุ่มต้องห้าม ({combo})")

    def _remove_hotkeys(self):
        if self._kb is None:
            return
        try:
            self._kb.unhook_all_hotkeys()
        except Exception:
            try: self._kb.unhook_all()
            except Exception: pass
        self._kb_hooked = False

    # ---------- จอแยก ----------
    def monitor_count(self):
        """จำนวนจอจริง (ไม่นับจอรวมเสมือนของ mss ที่ index 0)"""
        try:
            with mss.mss() as sct:
                return max(0, len(sct.monitors) - 1)
        except Exception:
            return 1

    # ---------- โปรแกรมรีโมท ----------
    def scan_remote_apps(self):
        """คืนรายชื่อโปรแกรมรีโมท/สื่อสารที่กำลังเปิดอยู่ (ชื่อสวยๆ ไม่ซ้ำ)"""
        found = {}
        names = self._process_names()
        for proc in names:
            base = proc.lower().rsplit(".exe", 1)[0]
            for key, nice in REMOTE_APPS.items():
                if key == base or key in base:
                    found[nice] = True
        return list(found.keys())

    def _process_names(self):
        # ใช้ psutil ถ้ามี (แม่นกว่า) ไม่งั้นใช้ tasklist ของ Windows
        try:
            import psutil
            return [p.info["name"] or "" for p in psutil.process_iter(["name"])]
        except Exception:
            pass
        if sys.platform.startswith("win"):
            try:
                out = subprocess.run("tasklist /fo csv /nh", shell=True, capture_output=True,
                                     text=True, timeout=8, errors="ignore").stdout
                names = []
                for line in out.splitlines():
                    parts = line.split('","')
                    if parts: names.append(parts[0].strip('"'))
                return names
            except Exception:
                pass
        return []

    # ---------- เปิด/ปิดระบบล็อก ----------
    def start(self):
        """เรียกตอนเริ่มสอบ: ล้างคลิปบอร์ด + ติดตั้งบล็อกคีย์ลัด"""
        if self.active:
            return
        self.active = True
        self.clear_clipboard()
        self._install_hotkeys()

    def stop(self):
        """เรียกตอนออกจากการสอบ: ปลดบล็อกคีย์ลัดทั้งหมด (คืนเครื่องให้ปกติ)"""
        if not self.active:
            return
        self.active = False
        self._remove_hotkeys()

    def precheck(self):
        """ตรวจก่อนเริ่มสอบ: คืน (ผ่านไหม, ข้อความปัญหาที่ต้องแก้)"""
        problems = []
        n = self.monitor_count()
        if n > 1:
            problems.append(f"ตรวจพบจอภาพ {n} จอ — ถอดจอที่สองออกก่อนเริ่มสอบ")
        remote = self.scan_remote_apps()
        if remote:
            problems.append("ปิดโปรแกรมเหล่านี้ก่อนสอบ: " + ", ".join(remote))
        return (not problems), problems


# ================= คะแนนความเสี่ยงทุจริต =================
# น้ำหนักของพฤติกรรมแต่ละแบบ (1.0 = ร้ายแรงที่สุด) ใช้คิด "เปอร์เซ็นต์ความเสี่ยง" = เวลาที่ทำผิด×น้ำหนัก / เวลาสอบทั้งหมด
SENSITIVITY = {   # ความเข้มของ AI คุมสอบต่อชุดข้อสอบ (ครูเลือกได้)
    "low":    {"label": "ผ่อนปรน", "angle": 1.4, "ratio": 1.2, "conf": 1.3},
    "medium": {"label": "ปกติ",    "angle": 1.0, "ratio": 1.0, "conf": 1.0},
    "high":   {"label": "เข้มงวด",  "angle": 0.7, "ratio": 0.75, "conf": 0.8},
}

RISK_WEIGHTS = {
    "พบมือถือ/หนังสือ": 1.0, "พบหลายคน": 1.0, "ไม่พบใบหน้า": 0.8,
    "หันซ้าย": 0.6, "หันขวา": 0.6, "คุย": 0.6, "ก้มหน้า": 0.5, "เหลือบตามองด้านข้าง": 0.4,
    "มือบังหน้า": 0.7, "มือใกล้กล้อง": 0.7, "กล้องถูกบัง": 1.0,
    "ก้มดูใต้โต๊ะ": 0.8, "ก้มมองด้านข้าง": 0.8, "ก้มหน้าซ้ำๆ": 0.7, "ก้มหลบกล้อง": 0.9, "ก้มนานผิดปกติ": 0.4,
    "สลับหน้าจอ/คีย์ลัด": 0.9, "เปิดจอที่สอง": 1.0, "เปิดโปรแกรมรีโมท": 1.0,
}

def risk_key(reason):
    if reason.startswith(("พบมือถือ", "พบหนังสือ")): return "พบมือถือ/หนังสือ"
    if reason.startswith("กดปุ่มต้องห้าม"): return "สลับหน้าจอ/คีย์ลัด"
    if reason.startswith("เปิดจอ"): return "เปิดจอที่สอง"
    if reason.startswith("เปิดโปรแกรม") or reason.startswith("ปิดโปรแกรม"): return "เปิดโปรแกรมรีโมท"
    return reason

def risk_level(pct):
    """คืน (ระดับ, สี) จากเปอร์เซ็นต์ความเสี่ยง"""
    if pct is None: return "ไม่มีข้อมูล", TEXT_MUTED
    if pct < 5: return "ต่ำ", GREEN_OK
    if pct < 15: return "ปานกลาง", AMBER_WARN
    if pct < 30: return "สูง", RED_ALERT
    return "สูงมาก", RED_ALERT

def open_video_writer(base_path, fps, size):
    """เปิดไฟล์วิดีโอ .mp4 (เล็กกว่า .avi หลายเท่า) ถ้าเครื่องไม่รองรับจะใช้ .avi แทน คืน (writer, path)"""
    for ext, codec in ((".mp4", "mp4v"), (".avi", "MJPG")):
        path = base_path + ext
        w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), fps, size)
        if w.isOpened(): return w, path
        w.release()
    return None, None

def is_black_frame(frame):
    """ภาพดำสนิท/เกือบดำ (กล้องถูกปิดฝา, ถูกโปรแกรมอื่นแย่ง, ไดรเวอร์ส่งภาพว่าง)"""
    small = cv2.resize(frame, (64, 48))
    return float(small.mean()) < 8 and float(small.std()) < 6

def open_camera(preferred=None, strict=False):
    """
    ลองเปิดกล้องหลายแบบ แล้วเลือกตัวที่ "ส่งภาพจริง" (ไม่ใช่ภาพดำ)
    บางเครื่อง DSHOW เปิดได้แต่ได้ภาพดำ ต้องใช้ MSMF แทน หรือกล้องจริงอยู่ index 1
    คืน (cap, คำอธิบาย) หรือ (None, None)
    """
    tries = [(0, "DSHOW", cv2.CAP_DSHOW), (0, "MSMF", cv2.CAP_MSMF), (0, "ANY", cv2.CAP_ANY),
             (1, "DSHOW", cv2.CAP_DSHOW), (1, "MSMF", cv2.CAP_MSMF), (2, "DSHOW", cv2.CAP_DSHOW)]
    if preferred is None:
        saved = load_settings().get("camera_index", "auto")
        preferred = saved if isinstance(saved, int) else None
    if preferred is not None:   # กล้องที่ผู้ใช้เลือกไว้ในหน้าตั้งค่า ลองก่อน
        mine = [(preferred, n, b) for n, b in (("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF), ("ANY", cv2.CAP_ANY))]
        tries = mine if strict else mine + [t for t in tries if t[0] != preferred]
    fallback = None
    for idx, name, backend in tries:
        try:
            cap = cv2.VideoCapture(idx, backend)
        except Exception:
            continue
        if not cap.isOpened():
            cap.release(); continue
        got_frame = False; good = False
        for _ in range(25):   # กล้องหลายรุ่นส่งภาพดำ 5-15 เฟรมแรกระหว่างปรับแสง
            ret, fr = cap.read()
            if not ret or fr is None: time.sleep(0.03); continue
            got_frame = True
            if not is_black_frame(fr): good = True; break
        desc = f"index={idx} backend={name}"
        if good:
            if fallback: fallback[0].release()
            return cap, desc
        if got_frame and fallback is None:
            fallback = (cap, desc + " (ภาพดำ)")   # เก็บไว้เผื่อไม่มีตัวอื่นดีกว่า (เช่น ปิดฝากล้องอยู่)
        else:
            cap.release()
    return fallback if fallback else (None, None)

def normalize_mcq_answer(answer, options):
    """แปลงเฉลยให้เป็นเลขข้อ (1..n) รับได้ทั้ง '2', ' 2 ', 'ข้อ 2' หรือข้อความของช้อยส์ตรงๆ; ไม่ถูกต้องคืน None"""
    a = (answer or "").strip()
    if not a: return None
    digits = "".join(ch for ch in a if ch.isdigit())
    if a.replace("ข้อ", "").strip().isdigit() and digits and 1 <= int(digits) <= len(options):
        return int(digits)
    for i, op in enumerate(options, start=1):
        if op.strip() == a: return i
    return None

def mcq_is_correct(q, picked):
    """ตรวจคำตอบปรนัย รองรับข้อสอบเก่าที่เคยบันทึกเฉลยเป็นข้อความของช้อยส์แทนเลขข้อ"""
    try: picked = int(picked)
    except (TypeError, ValueError): return False
    if picked <= 0: return False
    options = [o for o in (q.get('options') or "").split('|') if o.strip()]
    idx = normalize_mcq_answer(str(q.get('correct') or ""), options)
    return idx is not None and idx == picked

# ================= ค่าตั้งค่าในเครื่อง (กล้อง/ไมค์ที่เลือกไว้) =================
SETTINGS_FILE = os.path.join(BASE_DIR, "user_settings.json")

def load_settings():
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f: return json.load(f)
    except (OSError, ValueError):
        return {}

def save_settings(**kw):
    data = load_settings(); data.update(kw)
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f: json.dump(data, f, ensure_ascii=False, indent=2)
    except OSError as ex:
        print(f"⚠️ บันทึกการตั้งค่าไม่ได้: {ex}")


def _tone_wav(path, freq=660, secs=1.2, rate=22050):
    """สร้างไฟล์เสียงทดสอบลำโพง (เสียงติ๊ด) ด้วยไลบรารีมาตรฐาน"""
    import wave, math, struct
    with wave.open(path, "w") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        frames = bytearray()
        for i in range(int(rate * secs)):
            env = min(1.0, i / 800, (rate * secs - i) / 800)
            frames += struct.pack("<h", int(12000 * env * math.sin(2 * math.pi * freq * i / rate)))
        w.writeframes(bytes(frames))


class SettingsWindow(ctk.CTkToplevel):
    """
    โปรไฟล์ & ตั้งค่า: เปลี่ยนรหัสผ่าน (ทุกคน) + ตรวจอุปกรณ์ก่อนสอบ (นักเรียน): กล้อง / ไมค์ / ลำโพง / เครือข่าย / AI
    """
    def __init__(self, master, controller, on_close=None, start_tab=None):
        super().__init__(master)
        self.controller = controller; self.on_close_cb = on_close
        self.title("โปรไฟล์ & ตั้งค่า"); self.geometry("780x730"); self.configure(fg_color=BG_APP)
        self.protocol("WM_DELETE_WINDOW", self.close); self.after(200, lambda: self.alive and self.lift())
        self.is_student = controller.role == "student"
        self.cap = None; self.preview_on = False; self.mic_stream = None; self.mic_level = 0.0
        self.checks = {}; self.alive = True

        head = ctk.CTkFrame(self, fg_color="transparent"); head.pack(fill="x", padx=18, pady=(14, 4))
        ctk.CTkLabel(head, text="⚙ โปรไฟล์ & ตั้งค่า", font=FONT_H2, text_color=WHITE_TXT).pack(side="left")
        role_txt = "นักเรียน" if self.is_student else "อาจารย์"
        ctk.CTkLabel(head, text=f"👤 {controller.username} · {role_txt}", font=FONT_SMALL_B, text_color=TEAL).pack(side="right")

        self.tabs = ctk.CTkTabview(self, fg_color=BG_PANEL, segmented_button_selected_color=CYAN,
                                   segmented_button_fg_color=BG_PANEL_2, segmented_button_unselected_color=BG_PANEL_2,
                                   text_color=TEXT_MAIN, command=self._on_tab)
        self.tabs.pack(fill="both", expand=True, padx=14, pady=(4, 14))
        if self.is_student:
            self._build_precheck(self.tabs.add("✅ ตรวจอุปกรณ์ก่อนสอบ"))
            self._build_camera(self.tabs.add("📷 กล้อง"))
            self._build_audio(self.tabs.add("🎤 ไมค์ & ลำโพง"))
        self._build_account(self.tabs.add("🔑 เปลี่ยนรหัสผ่าน"))
        if start_tab:
            try: self.tabs.set(start_tab)
            except Exception: pass
        self._on_tab()

    # ---------- บัญชี ----------
    def _build_account(self, t):
        box = ctk.CTkFrame(t, fg_color=BG_PANEL_2, corner_radius=12); box.pack(fill="x", padx=40, pady=30)
        ctk.CTkLabel(box, text="เปลี่ยนรหัสผ่าน", font=FONT_H3, text_color=WHITE_TXT).pack(anchor="w", padx=20, pady=(16, 8))
        self.pw_old = self._entry(box, "รหัสผ่านเดิม"); self.pw_new = self._entry(box, "รหัสผ่านใหม่ (อย่างน้อย 6 ตัว)")
        self.pw_new2 = self._entry(box, "ยืนยันรหัสผ่านใหม่")
        self.pw_msg = ctk.CTkLabel(box, text="", font=FONT_SMALL_B, text_color=TEXT_MUTED); self.pw_msg.pack(pady=(4, 0))
        ctk.CTkButton(box, text="💾 บันทึกรหัสผ่านใหม่", height=38, corner_radius=8, fg_color=TEAL, hover_color=TEAL_DIM,
                      text_color="#04121a", font=FONT_BOLD, command=self.change_password).pack(fill="x", padx=20, pady=(8, 18))

    def _entry(self, parent, ph):
        e = ctk.CTkEntry(parent, placeholder_text=ph, show="•", height=38, corner_radius=8, fg_color=BG_INPUT,
                         border_color=BORDER_SOFT, text_color=TEXT_MAIN)
        e.pack(fill="x", padx=20, pady=5); return e

    def change_password(self):
        old, new, new2 = self.pw_old.get(), self.pw_new.get(), self.pw_new2.get()
        if not old or not new: return self.pw_msg.configure(text="กรอกให้ครบ", text_color=AMBER_WARN)
        if len(new) < 6: return self.pw_msg.configure(text="รหัสใหม่ต้องยาวอย่างน้อย 6 ตัวอักษร", text_color=AMBER_WARN)
        if new != new2: return self.pw_msg.configure(text="ยืนยันรหัสผ่านไม่ตรงกัน", text_color=AMBER_WARN)
        if new == old: return self.pw_msg.configure(text="รหัสใหม่ต้องไม่ซ้ำรหัสเดิม", text_color=AMBER_WARN)
        try:
            if self.is_student:
                ok = api.call("/api/change_password", {"old": old, "new": new}).get("ok")
            else:
                ok = db.change_password(self.controller.user_id, old, new)
        except ApiError as ex:
            return self.pw_msg.configure(text=str(ex), text_color=RED_ALERT)
        if ok:
            for e in (self.pw_old, self.pw_new, self.pw_new2): e.delete(0, "end")
            self.pw_msg.configure(text="✅ เปลี่ยนรหัสผ่านเรียบร้อย", text_color=GREEN_OK)
        else:
            self.pw_msg.configure(text="❌ รหัสผ่านเดิมไม่ถูกต้อง", text_color=RED_ALERT)

    # ---------- ตรวจอุปกรณ์ก่อนสอบ ----------
    def _build_precheck(self, t):
        ctk.CTkLabel(t, text="กด \"เริ่มตรวจ\" แล้วทำตามขั้นตอน ระบบจะเช็กว่าอุปกรณ์พร้อมสอบหรือไม่",
                     font=FONT_SMALL, text_color=TEXT_MUTED).pack(anchor="w", padx=12, pady=(8, 6))
        lst = ctk.CTkFrame(t, fg_color=BG_PANEL_2, corner_radius=12); lst.pack(fill="x", padx=10)
        for key, label in [("ai", "ระบบ AI (MediaPipe / YOLO)"), ("net", "เชื่อมต่อเครื่องอาจารย์"),
                           ("cam", "กล้อง: เปิดได้ ภาพไม่ดำ"), ("face", "ใบหน้า: เห็นหน้าคุณชัด 1 คน"),
                           ("mic", "ไมโครโฟน: มีเสียงเข้า (ลองพูด)"), ("spk", "ลำโพง/หูฟัง: ได้ยินเสียงทดสอบ")]:
            row = ctk.CTkFrame(lst, fg_color="transparent"); row.pack(fill="x", padx=14, pady=5)
            icon = ctk.CTkLabel(row, text="⚪", width=28, font=FONT_BODY); icon.pack(side="left")
            ctk.CTkLabel(row, text=label, font=FONT_BOLD, text_color=TEXT_MAIN).pack(side="left", padx=6)
            det = ctk.CTkLabel(row, text="", font=FONT_SMALL, text_color=TEXT_MUTED); det.pack(side="right")
            self.checks[key] = (icon, det)
        btns = ctk.CTkFrame(t, fg_color="transparent"); btns.pack(fill="x", padx=10, pady=10)
        ctk.CTkButton(btns, text="▶ เริ่มตรวจ", height=40, corner_radius=8, fg_color=TEAL, hover_color=TEAL_DIM,
                      text_color="#04121a", font=FONT_BOLD, command=self.run_precheck).pack(side="left", expand=True, fill="x", padx=(0, 6))
        self.spk_yes = ctk.CTkButton(btns, text="🔊 ได้ยินเสียง", height=40, corner_radius=8, fg_color=BG_INPUT,
                                     border_width=1, border_color=BORDER_SOFT, text_color=TEXT_MAIN, state="disabled",
                                     command=lambda: self._set("spk", True, "ยืนยันแล้ว"))
        self.spk_yes.pack(side="left", expand=True, fill="x", padx=6)
        self.spk_no = ctk.CTkButton(btns, text="🔇 ไม่ได้ยิน", height=40, corner_radius=8, fg_color=BG_INPUT,
                                    border_width=1, border_color=BORDER_SOFT, text_color=TEXT_MAIN, state="disabled",
                                    command=lambda: self._set("spk", False, "ตรวจระดับเสียง/หูฟัง"))
        self.spk_no.pack(side="left", expand=True, fill="x", padx=(6, 0))
        self.pre_summary = ctk.CTkLabel(t, text="", font=FONT_H3, text_color=TEXT_MUTED); self.pre_summary.pack(pady=(4, 0))
        self.pre_view = ctk.CTkLabel(t, text="", fg_color="#050a14", corner_radius=10, width=320, height=240)
        self.pre_view.pack(pady=6)

    def _set(self, key, ok, detail=""):
        icon, det = self.checks[key]
        icon.configure(text="✅" if ok is True else ("❌" if ok is False else "⏳"))
        det.configure(text=detail, text_color=GREEN_OK if ok is True else (RED_ALERT if ok is False else TEXT_MUTED))
        self.checks_state = getattr(self, "checks_state", {}); self.checks_state[key] = ok
        done = [v for v in self.checks_state.values() if v is not None]
        if len(done) == len(self.checks):
            if all(done):
                self.pre_summary.configure(text="🎉 อุปกรณ์พร้อมสอบ", text_color=GREEN_OK)
                self.controller.precheck_passed = True
            else:
                self.pre_summary.configure(text="⚠️ ยังมีบางรายการไม่ผ่าน แก้ไขแล้วตรวจใหม่", text_color=AMBER_WARN)

    def run_precheck(self):
        self.checks_state = {}
        for k in self.checks: self._set(k, None, "กำลังตรวจ...")
        self.pre_summary.configure(text="")
        # AI
        if AI_READY.is_set():
            self._set("ai", not AI_STATUS["errors"], "พร้อม" if not AI_STATUS["errors"] else AI_STATUS["errors"][0][:40])
        else:
            self._set("ai", None, AI_STATUS["text"][:40]); self.after(1000, self._recheck_ai)
        # เครือข่าย (เธรดแยก)
        def net():
            t0 = time.time()
            try:
                api.call("/api/ping", timeout=5); ms = (time.time() - t0) * 1000
                res = (ms < 1500, f"{ms:.0f} ms" + (" (ช้า)" if ms >= 500 else ""))
            except Exception as ex:
                res = (False, str(ex)[:45])
            self.after(0, lambda: self.alive and self._set("net", *res))
        threading.Thread(target=net, daemon=True).start()
        # กล้อง + ใบหน้า
        self._open_preview(check=True)
        # ไมค์
        self._start_mic(); self.mic_peak = 0.0; self.after(4000, self._finish_mic_check)
        self._set("mic", None, "พูดอะไรก็ได้ 3-4 วินาที...")
        # ลำโพง
        self.play_tone(); self._set("spk", None, "เล่นเสียงแล้ว ได้ยินไหม?")
        self.spk_yes.configure(state="normal"); self.spk_no.configure(state="normal")

    def _recheck_ai(self):
        if not self.alive: return
        if AI_READY.is_set():
            self._set("ai", not AI_STATUS["errors"], "พร้อม" if not AI_STATUS["errors"] else AI_STATUS["errors"][0][:40])
        else:
            self._set("ai", None, AI_STATUS["text"][:40]); self.after(1000, self._recheck_ai)

    def _finish_mic_check(self):
        if not self.alive: return
        if self.mic_stream is None:
            self._set("mic", False, self.mic_error[:45] if getattr(self, "mic_error", "") else "เปิดไมค์ไม่ได้")
        elif self.mic_peak > 0.02:
            self._set("mic", True, f"ระดับเสียงสูงสุด {self.mic_peak * 100:.0f}%")
        else:
            self._set("mic", False, "ไม่มีเสียงเข้า — เช็กไมค์/สิทธิ์ไมค์ใน Windows")

    # ---------- กล้อง ----------
    def _build_camera(self, t):
        top = ctk.CTkFrame(t, fg_color="transparent"); top.pack(fill="x", padx=10, pady=(6, 4))
        ctk.CTkLabel(top, text="เลือกกล้อง:", font=FONT_BOLD, text_color=TEXT_MAIN).pack(side="left")
        cur = load_settings().get("camera_index", "auto")
        self.cam_choice = ctk.CTkOptionMenu(top, values=["อัตโนมัติ", "กล้อง 0", "กล้อง 1", "กล้อง 2", "กล้อง 3"],
                                            command=lambda v: self._open_preview(), fg_color=BG_INPUT,
                                            button_color=CYAN_DIM, text_color=TEXT_MAIN, width=160)
        self.cam_choice.set("อัตโนมัติ" if cur == "auto" else f"กล้อง {cur}"); self.cam_choice.pack(side="left", padx=8)
        ctk.CTkButton(top, text="💾 ใช้กล้องนี้", width=120, fg_color=TEAL, hover_color=TEAL_DIM, text_color="#04121a",
                      font=FONT_SMALL_B, command=self.save_camera).pack(side="right")
        self.cam_info = ctk.CTkLabel(t, text="", font=FONT_SMALL, text_color=TEXT_MUTED); self.cam_info.pack(anchor="w", padx=12)
        self.cam_view = ctk.CTkLabel(t, text="", fg_color="#050a14", corner_radius=10, width=480, height=360)
        self.cam_view.pack(pady=6)

    def _cam_index(self):
        v = self.cam_choice.get() if hasattr(self, "cam_choice") else "อัตโนมัติ"
        return "auto" if v == "อัตโนมัติ" else int(v.split()[-1])

    def save_camera(self):
        save_settings(camera_index=self._cam_index())
        self.cam_info.configure(text="✅ บันทึกแล้ว — หน้าสอบจะใช้กล้องนี้", text_color=GREEN_OK)

    def _open_preview(self, check=False):
        self._close_preview()
        self.preview_on = True; token = object(); self._prev_token = token
        if hasattr(self, "cam_info"): self.cam_info.configure(text="⏳ กำลังเปิดกล้อง...", text_color=TEXT_MUTED)
        if check: self._set("cam", None, "กำลังเปิดกล้อง..."); self._set("face", None, "รอกล้อง...")
        idx = self._cam_index()
        def worker():
            cap, info = open_camera(None if idx == "auto" else idx, strict=idx != "auto")
            try: self.after(0, lambda: self._preview_ready(token, cap, info, check))
            except Exception:
                if cap is not None: cap.release()
        threading.Thread(target=worker, daemon=True).start()

    def _preview_ready(self, token, cap, info, check):
        if token is not self._prev_token or not self.alive:
            if cap is not None: cap.release()
            return
        self.cap = cap
        if cap is None:
            if hasattr(self, "cam_info"): self.cam_info.configure(text="❌ เปิดกล้องนี้ไม่ได้", text_color=RED_ALERT)
            if check: self._set("cam", False, "เปิดกล้องไม่ได้"); self._set("face", False, "-")
            return
        black = "ภาพดำ" in (info or "")
        if hasattr(self, "cam_info"):
            self.cam_info.configure(text=f"📷 {info}", text_color=AMBER_WARN if black else GREEN_OK)
        if check: self._set("cam", not black, "ภาพดำ — เช็กฝาปิดกล้อง" if black else info)
        self.face_frames = 0; self.face_ok_frames = 0; self.check_face = check
        self._preview_loop()

    def _preview_loop(self):
        if not self.alive or not self.preview_on or self.cap is None: return
        ret, frame = self.cap.read()
        if ret and frame is not None:
            frame = cv2.resize(cv2.flip(frame, 1), (480, 360))
            if self.check_face and face_detection is not None and self.face_frames < 45:
                self.face_frames += 1
                res = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                n = len(res.detections or [])
                if n == 1: self.face_ok_frames += 1
                if self.face_frames == 45:
                    ok = self.face_ok_frames >= 30
                    self._set("face", ok, "เห็นหน้าชัด" if ok else "หาหน้าไม่เจอ/มีหลายคน/แสงน้อย")
            elif self.check_face and face_detection is None and AI_READY.is_set():
                self.check_face = False; self._set("face", False, "AI ใบหน้าใช้ไม่ได้")
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            for lbl, size in ((getattr(self, "cam_view", None), (480, 360)), (getattr(self, "pre_view", None), (320, 240))):
                if lbl is not None and lbl.winfo_ismapped():
                    ci = ctk.CTkImage(light_image=img, dark_image=img, size=size); lbl.configure(image=ci); lbl.image = ci
        self.after(33, self._preview_loop)

    def _close_preview(self):
        self.preview_on = False
        if self.cap is not None:
            try: self.cap.release()
            except Exception: pass
            self.cap = None

    # ---------- ไมค์ & ลำโพง ----------
    def _build_audio(self, t):
        box = ctk.CTkFrame(t, fg_color=BG_PANEL_2, corner_radius=12); box.pack(fill="x", padx=20, pady=(16, 8))
        ctk.CTkLabel(box, text="🎤 ไมโครโฟน", font=FONT_H3, text_color=WHITE_TXT).pack(anchor="w", padx=16, pady=(12, 4))
        row = ctk.CTkFrame(box, fg_color="transparent"); row.pack(fill="x", padx=16)
        self.mic_devices = self._list_mics()
        names = [n for _, n in self.mic_devices] or ["(ไม่พบไมค์)"]
        self.mic_choice = ctk.CTkOptionMenu(row, values=names, fg_color=BG_INPUT, button_color=CYAN_DIM,
                                            text_color=TEXT_MAIN, width=420, command=lambda v: self._start_mic())
        saved = load_settings().get("mic_name")
        self.mic_choice.set(saved if saved in names else names[0]); self.mic_choice.pack(side="left")
        ctk.CTkButton(row, text="💾 ใช้ไมค์นี้", width=110, fg_color=TEAL, hover_color=TEAL_DIM, text_color="#04121a",
                      font=FONT_SMALL_B, command=lambda: save_settings(mic_name=self.mic_choice.get())).pack(side="right")
        self.mic_bar = ctk.CTkProgressBar(box, height=16, progress_color=GREEN_OK, fg_color=BG_INPUT)
        self.mic_bar.set(0); self.mic_bar.pack(fill="x", padx=16, pady=(10, 4))
        self.mic_msg = ctk.CTkLabel(box, text="ลองพูด แถบสีเขียวควรขยับ", font=FONT_SMALL, text_color=TEXT_MUTED)
        self.mic_msg.pack(anchor="w", padx=16, pady=(0, 12))

        box2 = ctk.CTkFrame(t, fg_color=BG_PANEL_2, corner_radius=12); box2.pack(fill="x", padx=20, pady=8)
        ctk.CTkLabel(box2, text="🔊 ลำโพง / หูฟัง", font=FONT_H3, text_color=WHITE_TXT).pack(anchor="w", padx=16, pady=(12, 4))
        ctk.CTkLabel(box2, text="ใช้สำหรับข้อสอบที่มีไฟล์เสียง (Listening)", font=FONT_SMALL,
                     text_color=TEXT_MUTED).pack(anchor="w", padx=16)
        ctk.CTkButton(box2, text="▶ เล่นเสียงทดสอบ", height=36, fg_color=CYAN_DIM, hover_color=CYAN, text_color=WHITE_TXT,
                      font=FONT_SMALL_B, command=self.play_tone).pack(anchor="w", padx=16, pady=(8, 14))

    def _list_mics(self):
        try:
            import sounddevice as sd
            out, seen = [], set()
            for i, d in enumerate(sd.query_devices()):
                if d.get("max_input_channels", 0) > 0 and d["name"] not in seen:
                    seen.add(d["name"]); out.append((i, d["name"][:60]))
            return out
        except Exception as ex:
            self.mic_error = f"ยังไม่ได้ติดตั้ง sounddevice ({type(ex).__name__})"
            return []

    def _start_mic(self):
        self._stop_mic()
        try:
            import sounddevice as sd
        except ImportError:
            self.mic_error = "ยังไม่ได้ติดตั้ง sounddevice — pip install sounddevice"
            if hasattr(self, "mic_msg"): self.mic_msg.configure(text=self.mic_error, text_color=AMBER_WARN)
            return
        except Exception as ex:
            self.mic_error = f"ระบบเสียงของเครื่องใช้ไม่ได้: {ex}"
            if hasattr(self, "mic_msg"): self.mic_msg.configure(text=self.mic_error, text_color=AMBER_WARN)
            return
        dev = None
        if hasattr(self, "mic_choice"):
            dev = next((i for i, n in self.mic_devices if n == self.mic_choice.get()), None)
        def cb(indata, frames, t, status):
            lvl = float(np.sqrt(np.mean(np.square(indata)))) * 8
            self.mic_level = max(lvl, self.mic_level * 0.85)
            self.mic_peak = max(getattr(self, "mic_peak", 0.0), min(1.0, lvl))
        try:
            self.mic_stream = sd.InputStream(device=dev, channels=1, samplerate=16000, callback=cb, blocksize=800)
            self.mic_stream.start(); self._mic_ui()
        except Exception as ex:
            self.mic_stream = None; self.mic_error = f"เปิดไมค์ไม่ได้: {ex}"
            if hasattr(self, "mic_msg"): self.mic_msg.configure(text=self.mic_error[:80], text_color=RED_ALERT)

    def _mic_ui(self):
        if not self.alive or self.mic_stream is None: return
        if hasattr(self, "mic_bar"): self.mic_bar.set(min(1.0, self.mic_level))
        self.after(60, self._mic_ui)

    def _stop_mic(self):
        if self.mic_stream is not None:
            try: self.mic_stream.stop(); self.mic_stream.close()
            except Exception: pass
            self.mic_stream = None

    def play_tone(self):
        path = os.path.join(BASE_DIR, "media_cache", "_tone_test.wav")
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if not os.path.exists(path): _tone_wav(path)
            if media.PLAYER.play(path): return
            import winsound
            winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception:
            try: self.bell()
            except Exception: pass

    # ---------- ทั่วไป ----------
    def _on_tab(self):
        tab = self.tabs.get()
        if tab.startswith("📷"): self._open_preview()
        elif tab.startswith("✅"): self._stop_mic()
        elif tab.startswith("🎤"): self._close_preview(); self._start_mic()
        elif tab.startswith("🔑"): self._close_preview(); self._stop_mic()

    def close(self):
        self.alive = False
        self._close_preview(); self._stop_mic()
        try: media.PLAYER.stop()
        except Exception: pass
        cb = self.on_close_cb
        # ซ่อนก่อนแล้วค่อยทำลาย กัน customtkinter แจ้ง error "bad window path name" จากอีเวนต์ที่ค้างอยู่
        self.withdraw(); self.after(150, self.destroy)
        if cb: self.master.after(200, cb)


class StudentExamPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=BG_APP); self.controller = controller
        set_page_background(self)
        self.exam_active = False; self.current_exam_id = None; self.is_camera_ready = False
        self.found_obj = False; self.frame_count = 0; self.heartbeat_count = 0
        
        self.columnconfigure(0, weight=2); self.columnconfigure(1, weight=3); self.rowconfigure(0, weight=1)
        self.cam_display_size = (640, 480)   # ขนาดภาพกล้องบนจอ (จะเล็กลงตอนสอบ)
        self.left = ctk.CTkFrame(self, fg_color=BG_PANEL, corner_radius=16, border_width=1, border_color=BORDER_SOFT)
        self.left.grid(row=0, column=0, sticky="nsew", padx=(12, 6), pady=12)

        head = ctk.CTkFrame(self.left, fg_color="transparent"); head.pack(fill="x", padx=16, pady=(16, 6))
        ctk.CTkLabel(head, text="📄 ห้องสอบ", font=FONT_H2, text_color=WHITE_TXT).pack(side="left")
        self.timer_pill = ctk.CTkFrame(head, corner_radius=10, fg_color=BG_INPUT, border_width=1, border_color=BORDER_SOFT)
        self.timer_pill.pack(side="right")
        self.timer_lbl = ctk.CTkLabel(self.timer_pill, text="⏱ --:--", font=FONT_MONO, text_color=AMBER_WARN)
        self.timer_lbl.pack(padx=14, pady=6)

        sep = ctk.CTkFrame(self.left, height=1, fg_color=BORDER_SOFT); sep.pack(fill="x", padx=16, pady=(2, 10))

        self.center = ctk.CTkFrame(self.left, fg_color="transparent")
        self.center.pack(fill="both", expand=True)
        self.q_scroll = ctk.CTkScrollableFrame(self.center, fg_color=BG_PANEL_2, corner_radius=12)
        self.q_scroll.pack(fill="both", expand=True, padx=16, pady=(0, 6))
        # แถบแบ่งหน้า (หน้าละ 1 ข้อ) — สร้างเนื้อหาตอนเริ่มสอบใน load_q
        self.nav = ctk.CTkFrame(self.center, fg_color="transparent")
        self.nav_prev = ctk.CTkButton(self.nav, text="◀ ข้อก่อนหน้า", width=120, height=34, corner_radius=8,
                                      fg_color=BG_PANEL_2, hover_color=CYAN_DIM, border_width=1, border_color=BORDER_SOFT,
                                      text_color=TEXT_MAIN, command=lambda: self.go_page(self.page_i - 1))
        self.nav_prev.pack(side="left", padx=(16, 4))
        self.nav_lbl = ctk.CTkLabel(self.nav, text="", font=FONT_SMALL_B, text_color=TEXT_MAIN)
        self.nav_lbl.pack(side="left", expand=True)
        self.nav_next = ctk.CTkButton(self.nav, text="ข้อถัดไป ▶", width=120, height=34, corner_radius=8,
                                      fg_color=CYAN_DIM, hover_color=CYAN, text_color=WHITE_TXT,
                                      command=lambda: self.go_page(self.page_i + 1))
        self.nav_next.pack(side="right", padx=(4, 16))
        self.nav_grid = ctk.CTkFrame(self.center, fg_color="transparent")   # แถวเลขข้อไว้กดข้าม
        self.media_msg = ctk.CTkLabel(self.left, text="", font=FONT_SMALL, text_color=AMBER_WARN, wraplength=360)
        self.media_retry = ctk.CTkButton(self.left, text="🔄 โหลดรูป/เสียงอีกครั้ง", height=30, corner_radius=8,
                                         fg_color=BG_PANEL_2, border_width=1, border_color=AMBER_WARN,
                                         text_color=AMBER_WARN, hover_color=BG_INPUT, command=self.retry_media)
        self.btn = ctk.CTkButton(self.left, text="พร้อมเริ่มการสอบ", height=56, corner_radius=12, font=FONT_BOLD,
                                  fg_color=BG_PANEL_2, text_color=TEXT_MUTED, state="disabled", command=self.toggle)
        self.btn.pack(pady=(0, 10), padx=16, fill="x")
        self.btn_settings = ctk.CTkButton(self.left, text="⚙ ตรวจอุปกรณ์ / ตั้งค่า", corner_radius=10, height=34,
                                          fg_color=BG_PANEL_2, hover_color=CYAN_DIM, border_width=1, border_color=CYAN_DIM,
                                          text_color=TEXT_MAIN, command=self.open_settings)
        self.btn_settings.pack(pady=(0, 8), padx=16, fill="x")
        ctk.CTkButton(self.left, text="🚪 Logout", corner_radius=10, fg_color="transparent", border_width=1,
                      border_color=BORDER_SOFT, text_color=TEXT_MUTED, hover_color=BG_PANEL_2,
                      command=controller.logout).pack(pady=(0, 16), padx=16, fill="x")

        self.right = ctk.CTkFrame(self, fg_color="#02030a", corner_radius=16, border_width=1, border_color=CYAN_DIM)
        self.right.grid(row=0, column=1, sticky="nsew", padx=(6, 12), pady=12)
        cam_wrap = ctk.CTkFrame(self.right, fg_color="transparent")
        cam_wrap.pack(fill="both", expand=True, padx=20, pady=(20, 10))
        ctk.CTkLabel(cam_wrap, text="🎥 กล้องตรวจสอบ (Proctoring Camera)", font=FONT_SMALL_B,
                     text_color=TEXT_MUTED, anchor="w").pack(fill="x", pady=(0, 8))
        self.cam_frame = ctk.CTkFrame(cam_wrap, fg_color="#050a14", corner_radius=12,
                                       border_width=1, border_color=BORDER_SOFT)
        self.cam_frame.pack(fill="both", expand=True)
        self.cam_lbl = ctk.CTkLabel(self.cam_frame, text=""); self.cam_lbl.pack(fill="both", expand=True, padx=6, pady=6)
        self.status_lbl = ctk.CTkLabel(self.right, text="สถานะ: ...", font=FONT_H2, text_color=AMBER_WARN)
        self.status_lbl.pack(pady=(0, 8))

        # HUD ค่าวัดจริงของโปรแกรม
        self.perf_lbl = ctk.CTkLabel(
            self.right,
            text="PERFORMANCE  |  FPS --  CPU --  RAM --  NET --  FRAME --  LOG --",
            font=("Consolas", 11, "bold"),
            text_color=TEXT_MUTED,
            anchor="center",
            justify="center"
        )
        self.perf_lbl.pack(fill="x", padx=12, pady=(0, 14))

        self.cap = None; self.is_running = False; self.rec_cheat = False
        self.out_webcam = None; self.out_screen = None; self.sct = mss.mss()
        self.exam_end_time = None; self.timer_job = None; self.auto_submitting = False
        self.detector = ObjectDetector(model); self.face_ai = FaceAnalyzer(); self.vfilter = ViolationFilter()
        self.occlusion = OcclusionAnalyzer()
        self.fps = 15.0; self.last_t = time.time(); self.stop_rec_time = 0; self.cheat_file = None
        self.cam_display_size = (640, 480)
        self.exam_start_t = None; self.viol_seconds = collections.Counter(); self.flag_seconds = 0.0

        # ================= Performance Monitor =================
        # วัดเฉพาะช่วงสอบ: FPS / CPU / RAM / Network latency / frame latency / เวลาเขียน log
        self.perf_token = 0
        self.perf_job = None
        self.perf_file = None
        self.perf_fp = None
        self.perf_writer = None
        self.perf_last = {
            "fps": 0.0, "cpu": 0.0, "ram": 0.0,
            "net_latency": -1.0, "frame_latency": 0.0, "log_write_ms": 0.0
        }
        self._frame_perf_start = None
        self._proc = None
        try:
            import psutil
            self._proc = psutil.Process(os.getpid())
            self._proc.cpu_percent(None)  # warm-up
        except Exception:
            self._proc = None
        self.risk_seconds = 0.0; self.prev_reasons = set(); self.last_logged = {}; self.live_reasons = []
        self._metrics = None
        self.last_hb = 0.0; self.kicked = False; self.qs_loaded = []; self.loading_box = None; self.settings_win = None
        self.lockdown = ExamLockdown(on_alert=self._lockdown_alert); self.lockdown_alerts = collections.deque(maxlen=50)

    def on_show(self):
        for w in self.q_scroll.winfo_children(): w.destroy()
        ctk.CTkLabel(self.q_scroll, text="รอสักครู่...", font=FONT_BODY, text_color=TEXT_MUTED).pack(pady=50)
        start_ai_loading()
        if AI_READY.is_set(): self.after(300, self._ai_ready_start)
        else: self._show_loading()

    # ---- หน้าจอโหลด AI: บอกความคืบหน้า ไม่ให้ผู้ใช้คิดว่าโปรแกรมค้าง ----
    def _show_loading(self):
        if self.loading_box is None:
            box = ctk.CTkFrame(self.cam_frame, fg_color=BG_PANEL, corner_radius=14, border_width=1, border_color=CYAN_DIM)
            ctk.CTkLabel(box, text="🧠 กำลังเตรียมระบบ AI คุมสอบ", font=FONT_H3, text_color=WHITE_TXT).pack(padx=30, pady=(22, 6))
            self.loading_txt = ctk.CTkLabel(box, text="", font=FONT_SUB, text_color=TEXT_MAIN); self.loading_txt.pack(padx=30)
            self.loading_bar = ctk.CTkProgressBar(box, width=360, height=12, progress_color=CYAN, fg_color=BG_INPUT)
            self.loading_bar.pack(padx=30, pady=12)
            self.loading_hint = ctk.CTkLabel(box, text="ครั้งแรกอาจใช้เวลา 10-40 วินาที โปรแกรมไม่ได้ค้าง",
                                             font=FONT_SMALL, text_color=TEXT_MUTED)
            self.loading_hint.pack(padx=30, pady=(0, 20))
            self.loading_box = box
        self.loading_box.place(relx=0.5, rely=0.5, anchor="center"); self.loading_t0 = time.time()
        self.status_lbl.configure(text="⏳ กำลังโหลด AI...", text_color=AMBER_WARN)
        self.btn.configure(state="disabled", text="รอระบบ AI...", fg_color=BG_PANEL_2, text_color=TEXT_MUTED)
        self._poll_loading()

    def _poll_loading(self):
        if not self.winfo_ismapped() and self.controller.role != "student": return
        st = AI_STATUS
        self.loading_bar.set(min(0.98, st["step"] / float(st["total"])))
        self.loading_txt.configure(text=f"ขั้นที่ {st['step']}/{st['total']} · {st['text']}")
        self.loading_hint.configure(text=f"ผ่านไป {int(time.time() - self.loading_t0)} วินาที · ครั้งแรกอาจใช้เวลา 10-40 วินาที โปรแกรมไม่ได้ค้าง")
        if AI_READY.is_set():
            self.loading_bar.set(1.0); self.loading_txt.configure(text="✅ พร้อมแล้ว")
            self.after(400, self._ai_ready_start); return
        self.after(200, self._poll_loading)

    def _ai_ready_start(self):
        if self.loading_box is not None: self.loading_box.place_forget()
        if self.detector.model is None and model is not None:
            self.detector = ObjectDetector(model)
        if AI_STATUS["errors"]:
            print("⚠️ " + " | ".join(AI_STATUS["errors"]))
        self.btn.configure(text="พร้อมเริ่มการสอบ")
        if self.controller.role == "student": self.start_cam()

    def _blank_cam(self, text):
        """ล้างภาพกล้องค้าง แล้วแสดงข้อความแทน"""
        if getattr(self, "_blank_img", None) is None:
            im = Image.new("RGB", (640, 480), "#050a14")
            self._blank_img = ctk.CTkImage(light_image=im, dark_image=im, size=(640, 480))
        self.cam_lbl.configure(image=self._blank_img, text=text, text_color=TEXT_MUTED, compound="center")
        self.cam_lbl.image = self._blank_img

    def open_settings(self, start_tab=None):
        if self.exam_active:
            messagebox.showwarning("กำลังสอบ", "ตั้งค่าอุปกรณ์ได้ก่อนเริ่มสอบเท่านั้น"); return
        if getattr(self, "settings_win", None) is not None and self.settings_win.winfo_exists():
            self.settings_win.lift(); return
        self.stop_cam()   # ปล่อยกล้องให้หน้าตั้งค่าใช้ (กล้องเปิดได้ทีละโปรแกรม)
        self._blank_cam("⚙ กำลังตั้งค่าอุปกรณ์...")
        def closed():
            self.settings_win = None
            if self.controller.role == "student" and AI_READY.is_set(): self.after(300, self.start_cam)
        self.settings_win = SettingsWindow(self, self.controller, on_close=closed, start_tab=start_tab)

    def toggle(self):
        if not self.exam_active:
            if not self.is_camera_ready: messagebox.showerror("Error", "กล้องไม่พร้อม"); return
            if not getattr(self.controller, "precheck_passed", False):
                if messagebox.askyesno("ตรวจอุปกรณ์ก่อนสอบ",
                                       "ยังไม่ได้ตรวจกล้อง/ไมค์/ลำโพง/อินเทอร์เน็ต\nต้องการตรวจก่อนเริ่มสอบไหม? (แนะนำ)"):
                    self.open_settings("✅ ตรวจอุปกรณ์ก่อนสอบ"); return
                self.controller.precheck_passed = True   # ถามครั้งเดียวต่อการล็อกอิน
            code = ctk.CTkInputDialog(text="รหัสวิชาสอบ:", title="Start").get_input()
            if not code: return
            try:
                d = api.call("/api/exam_by_code", {"code": code.strip()})
                exam = d.get("exam") if d.get("ok") else None
                if not exam or exam['active'] == 0: messagebox.showerror("ผิดพลาด", "รหัสผิดหรือยังไม่เปิดสอบ"); return
                qd = api.call("/api/questions", {"exam_id": exam['id']}, timeout=20)
                if not qd.get("ok"): messagebox.showerror("ผิดพลาด", "ข้อสอบชุดนี้ปิดแล้ว"); return
            except ApiError as ex:
                messagebox.showerror("เชื่อมต่อไม่ได้", str(ex)); return
            self.qs_loaded = qd["questions"]
            ok_sec, sec_problems = self.lockdown.precheck()
            if not ok_sec:
                if not messagebox.askyesno("ตรวจพบความเสี่ยง",
                        "ระบบตรวจพบสิ่งต่อไปนี้:\n\n• " + "\n• ".join(sec_problems) +
                        "\n\nแนะนำให้แก้ไขก่อน หากยืนยันเริ่มสอบ ระบบจะบันทึกเป็นพฤติกรรมเสี่ยงให้อาจารย์เห็น\nต้องการเริ่มสอบต่อไหม?"):
                    return
            if messagebox.askyesno("Confirm", "เริ่มสอบและอัดหน้าจอ?\n\n(ระหว่างสอบระบบจะบล็อกคีย์ลัด เช่น Alt+Tab และล้างคลิปบอร์ด)"):
                self._apply_sensitivity(exam.get("sensitivity", "medium"))
                # ระหว่างสอบ: ขยายหน้าข้อสอบ (70%) ย่อกล้อง (30%)
                self.columnconfigure(0, weight=7); self.columnconfigure(1, weight=3)
                self.cam_display_size = (320, 240)
                self.face_ai.lock_baseline(); self.vfilter.reset()   # ล็อกท่านั่งปกติของนักเรียนคนนี้
                self.exam_start_t = time.time(); self.viol_seconds = collections.Counter()
                self.flag_seconds = 0.0; self.risk_seconds = 0.0; self.prev_reasons = set(); self.last_logged = {}
                self.live_reasons = []
                self.exam_active = True; self.current_exam_id = exam['id']
                self.btn.configure(text="ส่งข้อสอบ", fg_color=RED_ALERT, text_color=TEXT_MAIN, state="normal")
                self.load_q(); self.lock_screen(); self.start_screen_record(); self._start_heartbeat()
                self._start_performance_monitor()
                self.lockdown.start(); self.lockdown_alerts.clear()
                for p in sec_problems: self._lockdown_alert(p)   # บันทึกปัญหาที่ยืนยันเริ่มสอบทั้งที่พบ
                self._start_security_monitor()
                duration = exam.get('duration', 0)
                if duration and duration > 0:
                    self.exam_end_time = time.time() + duration * 60
                else:
                    self.exam_end_time = None
                    self.timer_lbl.configure(text="⏱ ไม่จำกัดเวลา")
                self.tick_timer()
        else: self.submit()

    # ================= Performance Monitor =================
    def _start_performance_monitor(self):
        """เริ่มวัด Performance จริงระหว่างสอบ และบันทึก CSV ทุก ~1 วินาที"""
        self._stop_performance_monitor()

        evidence_dir = os.path.join(BASE_DIR, "evidence")
        os.makedirs(evidence_dir, exist_ok=True)

        stamp = time.strftime("%Y%m%d_%H%M%S")
        safe_user = secure_filename(str(self.controller.username)) or "student"
        self.perf_file = os.path.join(
            evidence_dir, f"PERFORMANCE_{safe_user}_{stamp}.csv"
        )

        try:
            self.perf_fp = open(self.perf_file, "w", newline="", encoding="utf-8-sig")
            self.perf_writer = csv.writer(self.perf_fp)
            self.perf_writer.writerow([
                "timestamp", "elapsed_sec", "fps", "cpu_percent", "ram_mb",
                "network_latency_ms", "frame_latency_ms", "log_write_ms"
            ])
            self.perf_fp.flush()
        except Exception as ex:
            print(f"⚠️ เปิด Performance log ไม่ได้: {ex}")
            self.perf_fp = None
            self.perf_writer = None

        self.perf_token += 1
        token = self.perf_token

        def loop():
            # CPU percent ของ process: เรียกครั้งแรกเป็น baseline
            if self._proc is not None:
                try:
                    self._proc.cpu_percent(None)
                except Exception:
                    pass

            while self.exam_active and token == self.perf_token:
                sample_t = time.perf_counter()

                # CPU/RAM ของโปรแกรม
                cpu = 0.0
                ram = 0.0
                try:
                    if self._proc is not None:
                        cpu = float(self._proc.cpu_percent(None))
                        ram = float(self._proc.memory_info().rss / (1024 * 1024))
                    else:
                        import psutil
                        proc = psutil.Process(os.getpid())
                        cpu = float(proc.cpu_percent(None))
                        ram = float(proc.memory_info().rss / (1024 * 1024))
                except Exception:
                    pass

                # Network latency ไปเครื่องอาจารย์
                net_latency = -1.0
                try:
                    base = api.base or media.teacher_base_url()
                    if base:
                        t0 = time.perf_counter()
                        r = media._http().get(
                            f"{base}/api/ping",
                            timeout=3,
                            headers={"X-Token": api.token or ""}
                        )
                        r.raise_for_status()
                        net_latency = (time.perf_counter() - t0) * 1000.0
                except Exception:
                    net_latency = -1.0

                fps = float(getattr(self, "fps", 0.0))
                frame_latency = float(getattr(self, "_last_frame_latency_ms", 0.0))
                elapsed = max(0.0, time.time() - (self.exam_start_t or time.time()))

                self.perf_last.update({
                    "fps": fps,
                    "cpu": cpu,
                    "ram": ram,
                    "net_latency": net_latency,
                    "frame_latency": frame_latency,
                })

                # เขียน log และวัด "เวลาที่ใช้เขียน log" จริง
                log_write_ms = 0.0
                if self.perf_writer is not None and self.perf_fp is not None:
                    try:
                        # สร้าง CSV 1 แถวในหน่วยความจำก่อน แล้ววัดเฉพาะเวลาที่เขียนลงดิสก์ + flush
                        buf = io.StringIO()
                        row_writer = csv.writer(buf)
                        row_writer.writerow([
                            time.strftime("%Y-%m-%d %H:%M:%S"),
                            round(elapsed, 3),
                            round(fps, 2),
                            round(cpu, 2),
                            round(ram, 2),
                            round(net_latency, 2) if net_latency >= 0 else "",
                            round(frame_latency, 2),
                            0.0
                        ])
                        row_text = buf.getvalue()

                        tlog = time.perf_counter()
                        self.perf_fp.write(row_text)
                        self.perf_fp.flush()
                        log_write_ms = (time.perf_counter() - tlog) * 1000.0

                        # เขียนค่าจริงทับในแถวล่าสุดไม่ได้โดยไม่ทำให้ไฟล์ซับซ้อน
                        # ดังนั้นเก็บค่า log_write_ms ของรอบปัจจุบันไว้ใน HUD
                    except Exception as ex:
                        print(f"⚠️ performance log error: {ex}")

                self.perf_last["log_write_ms"] = log_write_ms

                # อัปเดต HUD ผ่าน Tk main thread
                try:
                    self.after(0, self._update_perf_hud)
                except Exception:
                    pass

                # คุม sampling ประมาณ 1 Hz
                spent = time.perf_counter() - sample_t
                time.sleep(max(0.05, 1.0 - spent))

        threading.Thread(target=loop, daemon=True).start()

    def _update_perf_hud(self):
        if not hasattr(self, "perf_lbl") or not self.perf_lbl.winfo_exists():
            return
        p = self.perf_last
        net = f"{p['net_latency']:.0f}ms" if p["net_latency"] >= 0 else "OFF"
        self.perf_lbl.configure(
            text=(
                f"PERFORMANCE  |  "
                f"FPS {p['fps']:.1f}  |  "
                f"CPU {p['cpu']:.1f}%  |  "
                f"RAM {p['ram']:.0f}MB  |  "
                f"NET {net}  |  "
                f"FRAME {p['frame_latency']:.1f}ms  |  "
                f"LOG {p['log_write_ms']:.2f}ms"
            )
        )

    def _stop_performance_monitor(self):
        """หยุด Performance Monitor และปิด CSV อย่างปลอดภัย"""
        self.perf_token = getattr(self, "perf_token", 0) + 1
        if self.perf_job:
            try:
                self.after_cancel(self.perf_job)
            except Exception:
                pass
            self.perf_job = None

        if self.perf_fp is not None:
            try:
                self.perf_fp.flush()
                self.perf_fp.close()
            except Exception:
                pass

        self.perf_fp = None
        self.perf_writer = None
        self.perf_file = None

        SCREEN_FPS = 2.0   # อัดหน้าจอ 2 ภาพ/วินาที พอสำหรับดูหลักฐาน และไฟล์เล็กพอส่งผ่านเน็ตนอกวง

    def start_screen_record(self):
        try:
            evidence_dir = os.path.join(BASE_DIR, "evidence"); os.makedirs(evidence_dir, exist_ok=True)
            m = self.sct.monitors[1]
            scale = min(1.0, 1280.0 / m['width'])
            self.screen_size = (int(m['width'] * scale) // 2 * 2, int(m['height'] * scale) // 2 * 2)
            base = os.path.join(evidence_dir, f"FULL_{self.controller.username}_{int(time.time())}")
            self.out_screen, self.screen_file = open_video_writer(base, self.SCREEN_FPS, self.screen_size)
            self.next_screen_t = time.time()
        except Exception as ex:
            print(f"⚠️ อัดหน้าจอไม่ได้: {ex}"); self.out_screen = None

    def stop_screen_record(self):
        if self.out_screen: self.out_screen.release(); self.out_screen = None; upload_to_teacher(self.screen_file)

    def tick_timer(self):
        if not self.exam_active or self.exam_end_time is None:
            return
        remaining = int(self.exam_end_time - time.time())
        if remaining <= 0:
            self.timer_lbl.configure(text="⏱ 00:00", text_color=RED_ALERT)
            if not self.auto_submitting:
                self.auto_submitting = True
                messagebox.showwarning("หมดเวลา", "หมดเวลาสอบแล้ว ระบบจะส่งข้อสอบให้อัตโนมัติ")
                self.submit()
            return
        mm, ss = divmod(remaining, 60)
        color = RED_ALERT if remaining <= 60 else (AMBER_WARN if remaining <= 300 else TEAL)
        self.timer_lbl.configure(text=f"⏱ {mm:02d}:{ss:02d}", text_color=color)
        self.timer_job = self.after(1000, self.tick_timer)

    def stop_timer(self):
        if self.timer_job:
            try: self.after_cancel(self.timer_job)
            except: pass
            self.timer_job = None
        self.exam_end_time = None; self.auto_submitting = False
        self.timer_lbl.configure(text="")

    def load_q(self):
        media.PLAYER.stop()
        self.qs = getattr(self, "qs_loaded", []) or []
        self.ans = {}; self.opt_order = {}; self.media_cache = {}; self.page_i = 0
        # --- สุ่มลำดับข้อ (ต่อคน) กันลอกกัน — เฉลยผูกกับ id ข้อ ไม่ขึ้นกับลำดับ ---
        random.shuffle(self.qs)
        for q in self.qs:
            if q['type'] == 'MCQ':
                opts = [o for o in q['options'].split('|') if o.strip()]
                order = list(range(1, len(opts) + 1))   # เลขช้อยส์เดิม (1-based) = ค่าที่ใช้ตรวจคะแนน
                random.shuffle(order)                    # --- สลับตำแหน่งช้อยส์ (ต่อคน) ---
                self.opt_order[q['id']] = order
                self.ans[q['id']] = ctk.IntVar(value=0)  # เก็บ "เลขช้อยส์เดิม" ที่เลือก ไม่ต้องแปลงตอนส่ง
        self._build_nav_grid()
        self.render_page(0)
        self._load_media_async()

    def _build_nav_grid(self):
        for w in self.nav_grid.winfo_children(): w.destroy()
        self.nav_chips = {}
        per_row = 12
        for idx, q in enumerate(self.qs):
            b = ctk.CTkButton(self.nav_grid, text=str(idx + 1), width=34, height=28, corner_radius=6,
                              font=FONT_SMALL_B, fg_color=BG_PANEL_2, hover_color=CYAN_DIM, text_color=TEXT_MUTED,
                              border_width=1, border_color=BORDER_SOFT, command=lambda i=idx: self.go_page(i))
            b.grid(row=idx // per_row, column=idx % per_row, padx=2, pady=2)
            self.nav_chips[idx] = b

    def _answered(self, q):
        v = self.ans.get(q['id'])
        return bool(v and v.get())

    def _refresh_nav(self):
        n = len(self.qs); answered = sum(1 for q in self.qs if self._answered(q))
        self.nav_lbl.configure(text=f"ข้อ {self.page_i + 1} / {n}   (ตอบแล้ว {answered}/{n})")
        self.nav_prev.configure(state="normal" if self.page_i > 0 else "disabled")
        self.nav_next.configure(state="normal" if self.page_i < n - 1 else "disabled")
        for idx, chip in getattr(self, "nav_chips", {}).items():
            q = self.qs[idx]
            if idx == self.page_i:
                chip.configure(fg_color=CYAN, text_color="#04121a", border_color=CYAN)
            elif self._answered(q):
                chip.configure(fg_color=TEAL_DIM, text_color=WHITE_TXT, border_color=TEAL)
            else:
                chip.configure(fg_color=BG_PANEL_2, text_color=TEXT_MUTED, border_color=BORDER_SOFT)

    def go_page(self, i):
        i = max(0, min(len(self.qs) - 1, i))
        self.render_page(i)

    def render_page(self, i):
        # แสดงทีละ 1 ข้อ — โหลดเฉพาะรูป/เสียงของข้อนี้ ลดภาระเครื่อง
        media.PLAYER.stop()
        self.page_i = i
        for w in self.q_scroll.winfo_children(): w.destroy()
        self.media_slots = {}
        if not self.qs:
            ctk.CTkLabel(self.q_scroll, text="ไม่มีข้อสอบในชุดนี้", font=FONT_BODY, text_color=TEXT_MUTED).pack(pady=40)
            self.nav.pack_forget(); self.nav_grid.pack_forget(); return
        if not self.nav.winfo_ismapped():
            self.nav.pack(fill="x", pady=(0, 4))
            self.nav_grid.pack(pady=(0, 8))
        q = self.qs[i]
        f = ctk.CTkFrame(self.q_scroll, fg_color=BG_PANEL, corner_radius=10, border_width=1, border_color=BORDER_SOFT)
        f.pack(fill="both", expand=True, pady=6, padx=4)
        head = ctk.CTkFrame(f, fg_color="transparent"); head.pack(fill="x", padx=14, pady=(12, 4))
        badge = ctk.CTkFrame(head, width=26, height=26, corner_radius=13, fg_color=CYAN_DIM)
        badge.pack(side="left", padx=(0, 8)); badge.pack_propagate(False)
        ctk.CTkLabel(badge, text=str(i + 1), font=FONT_SMALL_B, text_color=WHITE_TXT).place(relx=0.5, rely=0.5, anchor="center")
        ctk.CTkLabel(head, text=q['text'], font=FONT_BOLD, text_color=TEXT_MAIN, wraplength=380,
                     justify="left").pack(side="left", anchor="w")
        qm = ctk.CTkFrame(f, fg_color="transparent", height=1); qm.pack(fill="x", padx=(38, 14))
        self.media_slots[(q['id'], 0)] = qm
        if q['type'] == 'MCQ':
            opts = [o for o in q['options'].split('|') if o.strip()]
            v = self.ans[q['id']]
            for orig in self.opt_order.get(q['id'], list(range(1, len(opts) + 1))):
                op = opts[orig - 1]
                row = ctk.CTkFrame(f, fg_color="transparent"); row.pack(fill="x", padx=(38, 14), pady=2)
                # value = เลขช้อยส์เดิม ทำให้ตรวจคะแนนได้เหมือนเดิมแม้ตำแหน่งจะสลับ
                ctk.CTkRadioButton(row, text=op, variable=v, value=orig, fg_color=CYAN, hover_color=CYAN_DIM,
                                   text_color=TEXT_MAIN, command=self._refresh_nav).pack(anchor="w")
                holder = ctk.CTkFrame(row, fg_color="transparent", width=1, height=1); holder.pack(anchor="w", padx=(28, 0))
                self.media_slots[(q['id'], orig)] = holder
        self._render_cached_media(q['id'])
        self._refresh_nav()

    def _render_cached_media(self, qid):
        for it in self.media_cache.get(qid, []):
            holder = self.media_slots.get((qid, it['slot']))
            if holder is None or not holder.winfo_exists(): continue
            if it['kind'] == 'image':
                big = it['slot'] == 0
                make_image_label(holder, it['local'], 380 if big else 170, 240 if big else 120).pack(anchor="w", pady=4)
            else:
                AudioControl(holder, it['local'], title="ฟังเสียงประกอบ" if it['slot'] == 0 else "ฟังเสียง",
                             compact=it['slot'] != 0).pack(anchor="w", pady=4)

    # ---- โหลดรูป/เสียงจากเครื่องอาจารย์ในเธรดแยก ไม่ให้หน้าจอค้าง ----
    def _load_media_async(self):
        exam_id = self.current_exam_id; qids = [q['id'] for q in self.qs]
        self.media_retry.pack_forget(); self.media_msg.pack_forget()
        if not qids: return
        self.media_msg.configure(text="⏳ กำลังโหลดรูปภาพ/เสียงประกอบข้อสอบ...", text_color=TEXT_MUTED)
        self.media_msg.pack(before=self.btn, pady=(0, 6))
        def worker():
            result, err = media.fetch_media_for_questions(qids)
            try: self.after(0, lambda: self._render_media(exam_id, result, err))
            except Exception: pass
        threading.Thread(target=worker, daemon=True).start()

    def retry_media(self):
        if self.current_exam_id: self._load_media_async()

    def _render_media(self, exam_id, result, err):
        if self.current_exam_id != exam_id: return   # ส่งข้อสอบ/เปลี่ยนชุดไปแล้ว
        self.media_cache = result or {}
        if self.qs:   # วางรูป/เสียงของข้อที่กำลังแสดงอยู่
            self._render_cached_media(self.qs[self.page_i]['id'])
        if err:
            self.media_msg.configure(text=f"⚠️ โหลดรูป/เสียงไม่สำเร็จ: {err}",
                                     text_color=AMBER_WARN)
            self.media_msg.pack(before=self.btn, pady=(0, 4)); self.media_retry.pack(before=self.btn, padx=16, pady=(0, 8))
        else:
            self.media_msg.pack_forget(); self.media_retry.pack_forget()

    def submit(self):
        if self.btn.cget("state") == "disabled": return  # กันส่งซ้ำซ้อน
        # ถ้ายังตอบไม่ครบ ห้ามส่ง (ยกเว้นหมดเวลาแล้วระบบส่งให้อัตโนมัติ)
        if not self.auto_submitting:
            unanswered = [i + 1 for i, q in enumerate(self.qs) if not self._answered(q)]
            if unanswered:
                first = unanswered[0]
                show = ", ".join(map(str, unanswered[:20])) + (" ..." if len(unanswered) > 20 else "")
                messagebox.showwarning("ยังตอบไม่ครบ",
                    f"ยังไม่ได้ตอบ {len(unanswered)} ข้อ: ข้อ {show}\n\nกรุณาตอบให้ครบทุกข้อก่อนส่ง")
                self.go_page(first - 1)   # พาไปข้อแรกที่ยังไม่ตอบ
                return
        # ล็อกปุ่มกันกดซ้ำระหว่างบันทึกคะแนน
        self.btn.configure(state="disabled", text="⏳ กำลังส่งข้อสอบ...")
        self.update_idletasks()
        # อ่านค่าจากหน้าจอในเธรดหลัก (tkinter ไม่ปลอดภัยถ้าอ่านจากเธรดอื่น) แล้วค่อยบันทึกในเธรดแยก
        picked = {q['id']: self.ans[q['id']].get() for q in self.qs}
        exam_seconds = max(1.0, time.time() - (self.exam_start_t or time.time()))
        risk_pct = round(min(100.0, 100.0 * self.risk_seconds / exam_seconds), 1)
        detail = {k: round(v, 1) for k, v in self.viol_seconds.items() if v >= 0.5}
        info = dict(exam_id=self.current_exam_id, exam_seconds=round(exam_seconds, 1),
                    flagged_seconds=round(self.flag_seconds, 1), risk_pct=risk_pct, detail=detail)
        threading.Thread(target=self._submit_worker, args=(picked, info), daemon=True).start()

    def _submit_worker(self, picked, info):
        payload = {"exam_id": info['exam_id'], "answers": {str(k): v for k, v in picked.items()},
                   "exam_seconds": info['exam_seconds'], "flagged_seconds": info['flagged_seconds'],
                   "risk_pct": info['risk_pct'], "detail": info['detail']}
        # บันทึกคำตอบไว้ในเครื่องก่อน เผื่อเน็ตหลุดตอนส่ง จะได้ไม่เสียคำตอบ
        backup = os.path.join(BASE_DIR, "evidence", f"ANSWERS_{self.controller.username}_{int(time.time())}.json")
        try:
            os.makedirs(os.path.dirname(backup), exist_ok=True)
            with open(backup, "w", encoding="utf-8") as f: json.dump(payload, f, ensure_ascii=False)
        except OSError: pass
        err = None
        for attempt in range(3):
            try:
                d = api.call("/api/submit", payload, timeout=30)
                if d.get("ok"): break
                err = d.get("error", "unknown")
            except ApiError as ex:
                err = str(ex)
            time.sleep(2)
        else:
            print(f"❌ submit error: {err}")
            self.after(0, lambda: messagebox.showerror("ส่งข้อสอบไม่สำเร็จ",
                       f"{err}\n\nคำตอบถูกเก็บไว้ในเครื่องแล้ว ตรวจการเชื่อมต่อแล้วกด 'ส่งข้อสอบ' อีกครั้ง"))
            self.after(0, lambda: self.btn.configure(state="normal", text="ส่งข้อสอบ"))
            return
        sc, mx = d["score"], d.get("max", 0)
        self.after(0, lambda: self._finish_submit(sc, mx))

    def _finish_submit(self, sc, mx):
        messagebox.showinfo("เสร็จสิ้น", f"คะแนนรวม: {sc:g} / {mx:g}" if mx else f"คะแนนรวม: {sc:g}")
        self.force_stop_exam()

    def force_stop_exam(self):
        self.exam_active = False
        self._stop_performance_monitor()
        self.current_exam_id = None; self.stop_screen_record(); self.stop_timer()
        self.columnconfigure(0, weight=2); self.columnconfigure(1, weight=3)   # คืนสัดส่วนเดิม
        self.cam_display_size = (640, 480)
        try: self.lockdown.stop()
        except Exception as ex: print(f"⚠️ ปลดล็อกเครื่องไม่สำเร็จ: {ex}")
        if self.rec_cheat: self._finish_cheat_clip()
        self.face_ai.base = None; self.live_reasons = []; self.exam_start_t = None
        self.hb_token = getattr(self, "hb_token", 0) + 1   # หยุดเธรด heartbeat
        if self.controller.role == 'student' and api.token:
            threading.Thread(target=lambda: self._quiet_call("/api/clear_live"), daemon=True).start()
        media.clear_cache(); self.media_msg.pack_forget(); self.media_retry.pack_forget()
        for w in (getattr(self, "nav", None), getattr(self, "nav_grid", None)):
            if w is not None: w.pack_forget()
        self.btn.configure(text="พร้อมเริ่มการสอบ", fg_color=BG_PANEL_2, text_color=TEXT_MUTED, state="disabled")
        self.unlock_screen()

    def lock_screen(self):
        r = self.controller.winfo_toplevel()
        self._locked_root = r
        # กัน Esc หลุด fullscreen และกันปิดหน้าต่างระหว่างสอบ
        try:
            r.bind("<Escape>", lambda e: "break")
            r.bind("<F11>", lambda e: "break")
            r.protocol("WM_DELETE_WINDOW", lambda: messagebox.showwarning("กำลังสอบ", "กรุณาส่งข้อสอบก่อนปิดโปรแกรม"))
        except Exception: pass
        # ตั้งเต็มจอหลังหน้าต่างยืนยันปิดสนิทแล้ว (หน่วงสั้นๆ) ไม่งั้นบางเครื่องจะไม่เต็มจอ
        self.after(200, self._enter_fullscreen)
        self._keep_fullscreen()

    def _enter_fullscreen(self):
        r = getattr(self, "_locked_root", None)
        if r is None or not self.exam_active: return
        try: r.state('zoomed')          # ขยายเต็มจอ (สำรอง เผื่อ -fullscreen ไม่ทำงาน)
        except Exception: pass
        for attr in ('-fullscreen', '-topmost'):
            try: r.attributes(attr, True)
            except Exception: pass
        try: r.lift(); r.focus_force()
        except Exception: pass
        ok = False
        try: ok = bool(r.attributes('-fullscreen'))
        except Exception: pass
        print(f"🖥 เข้าโหมดเต็มจอ: {'สำเร็จ' if ok else 'ใช้โหมดขยายเต็มจอแทน'}")

    def _keep_fullscreen(self):
        # ย้ำเต็มจอทุก 1 วินาที ถ้าถูกทำให้หลุด (ย่อ/สลับหน้าต่าง) จะเด้งกลับมาเสมอ
        if not self.exam_active: return
        r = getattr(self, "_locked_root", None)
        if r is not None:
            try:
                fs = bool(r.attributes('-fullscreen'))
            except Exception:
                fs = True
            if not fs:
                try: r.state('zoomed')
                except Exception: pass
                for attr in ('-fullscreen', '-topmost'):
                    try: r.attributes(attr, True)
                    except Exception: pass
                try: r.lift()
                except Exception: pass
        self.after(1000, self._keep_fullscreen)

    def unlock_screen(self):
        r = getattr(self, "_locked_root", None) or self.controller.winfo_toplevel()
        try:
            r.attributes('-fullscreen', False); r.attributes('-topmost', False)
            try: r.state('normal')
            except Exception: pass
            r.unbind("<Escape>"); r.unbind("<F11>")
            r.protocol("WM_DELETE_WINDOW", self.controller.on_close)
        except Exception: pass

    def start_cam(self):
        if self.is_running or getattr(self, "_cam_opening", False): return
        if self.cap is not None:
            try: self.cap.release()
            except Exception: pass
            self.cap = None
        self._cam_opening = True; self._cam_token = getattr(self, "_cam_token", 0) + 1; token = self._cam_token
        self._blank_cam("⏳ กำลังเปิดกล้อง...")
        def worker():
            cap, info = open_camera()
            try: self.after(0, lambda: self._cam_opened(token, cap, info))
            except Exception:
                if cap is not None: cap.release()
        threading.Thread(target=worker, daemon=True).start()

    def _cam_opened(self, token, cap, info):
        self._cam_opening = False
        if token != self._cam_token:          # ออกจากหน้านี้ไปก่อนกล้องเปิดเสร็จ
            if cap is not None: cap.release()
            return
        if cap is None:
            self.cam_lbl.configure(text="❌ ไม่พบกล้อง / เปิดกล้องไม่ได้\nปิดโปรแกรมอื่นที่ใช้กล้อง (Zoom, Teams, LINE, Camera) แล้วลองใหม่",
                                   text_color=RED_ALERT)
            self.status_lbl.configure(text="⚠️ กล้องไม่พร้อม", text_color=RED_ALERT)
            return
        print(f"📷 ใช้กล้อง: {info}")
        self.cap = cap; self.cam_lbl.configure(text="")
        self.is_running = True; self.black_since = None
        self.face_ai.reset(); self.vfilter.reset(); self.occlusion.reset(); self.detector.start()
        self.last_t = time.time(); self.update_cam()

    def stop_cam(self):
        """ปิดกล้องจริงๆ เมื่อออกจากหน้าสอบ/ถูกเชิญออก ไม่งั้นล็อกอินใหม่จะเปิดกล้องไม่ได้ (กล้องถูกจับค้างไว้)"""
        self.is_running = False; self.detector.stop()
        if self._metrics is not None:
            try: self._metrics.close()
            except Exception: pass
            self._metrics = None
        self._cam_token = getattr(self, "_cam_token", 0) + 1; self._cam_opening = False
        if self.cap is not None:
            try: self.cap.release()
            except Exception: pass
            self.cap = None
        self.is_camera_ready = False

    def _detect_faces(self, rgb, w, h, now):
        """คืน (เหตุผลดิบของเฟรมนี้, จำนวนใบหน้า)"""
        reasons = set()
        fd = face_detection.process(rgb)
        dets = [d for d in (fd.detections or [])
                if d.location_data.relative_bounding_box.width * d.location_data.relative_bounding_box.height >= 0.004]
        fm = face_mesh.process(rgb)
        faces = fm.multi_face_landmarks or []
        n_faces = max(len(dets), len(faces))
        boxes = []
        for d in dets:
            bb = d.location_data.relative_bounding_box
            boxes.append((bb.xmin * w, bb.ymin * h, (bb.xmin + bb.width) * w, (bb.ymin + bb.height) * h))
        if not boxes:   # ไม่มีผลจาก FaceDetection ใช้กรอบจากจุด FaceMesh แทน
            for fl in faces:
                xs = [p.x * w for p in fl.landmark]; ys = [p.y * h for p in fl.landmark]
                boxes.append((min(xs), min(ys), max(xs), max(ys)))
        self._last_face_boxes = boxes
        if n_faces == 0:
            reasons.add("ไม่พบใบหน้า"); self.face_ai.mar_hist.clear()
            if self.exam_active and self.face_ai.base is not None and self.face_ai.head.face_lost(True, now):
                reasons.add("ก้มหลบกล้อง")   # ก้มลึกแล้วหน้าหายทันที
        elif n_faces > 1:
            reasons.add("พบหลายคน")
        if n_faces > 0: self.face_ai.head.face_lost(False, now)
        if faces:
            # คนสอบ = ใบหน้าที่ใหญ่ที่สุด (ใกล้กล้องที่สุด)
            def face_w(fl): xs = [p.x for p in fl.landmark]; return max(xs) - min(xs)
            main_face = max(faces, key=face_w)
            reasons |= self.face_ai.analyze(main_face.landmark, w, h, now, self.exam_active)
        return reasons, n_faces

    def update_cam(self):
        if not self.is_running: return
        if self.exam_active and self.kicked:
            self.kicked = False
            self.stop_cam(); messagebox.showerror("ถูกเชิญออก", "ถูกเชิญออก")
            self.force_stop_exam(); self.controller.logout(); return

        ret, frame = self.cap.read() if self.cap is not None else (False, None)
        if not ret:
            self.frame_count += 1
            if self.frame_count % 100 == 0:
                print(f"⚠️ cap.read() คืนค่า ret=False ({self.frame_count} ครั้งแล้ว) — กล้องเปิดได้แต่อ่านภาพไม่ได้")
        if ret:
            try:
                _frame_perf_start = time.perf_counter()
                now = time.time()
                dt = now - self.last_t; self.last_t = now
                if 0 < dt < 1: self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)

                frame = cv2.flip(frame, 1)
                if frame.shape[1] != 640 or frame.shape[0] != 480: frame = cv2.resize(frame, (640, 480))
                if is_black_frame(frame):
                    self.black_since = self.black_since or now
                    if now - self.black_since > 1.5:
                        self.is_camera_ready = False
                        self.status_lbl.configure(text="⚠️ กล้องส่งภาพดำ: เช็กฝาปิดกล้อง / โปรแกรมอื่นใช้กล้องอยู่", text_color=AMBER_WARN)
                        if not self.exam_active: self.btn.configure(state="disabled", fg_color=BG_PANEL_2, text_color=TEXT_MUTED)
                        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=self.cam_display_size)
                        self.cam_lbl.configure(image=ctk_img); self.cam_lbl.image = ctk_img
                        # หลุดนานเกิน 5 วินาที ลองเปิดกล้องใหม่เอง
                        if now - self.black_since > 5 and not self.exam_active:
                            self.stop_cam(); self.after(300, self.start_cam); return
                        self.after(30, self.update_cam); return
                else:
                    self.black_since = None
                ai_frame = enhance_low_light(frame)
                h, w = ai_frame.shape[:2]
                rgb = cv2.cvtColor(ai_frame, cv2.COLOR_BGR2RGB)
                cheating = False
                raw = set(); n_faces = 0

                # ---- Face AI (MediaPipe) ----
                if face_detection is not None and face_mesh is not None:
                    face_raw, n_faces = self._detect_faces(rgb, w, h, now)
                    raw |= face_raw
                    # ---- Occlusion AI (มือบังหน้า / บังกล้อง / บังคนข้างหลัง) ----
                    self.occlusion.set_hands(hand_model)
                    raw |= self.occlusion.analyze(ai_frame, rgb, self._last_face_boxes, n_faces, now)

                # ---- Object AI (YOLO, เธรดแยก) ----
                self.frame_count += 1
                self.detector.hand_boxes = list(self.occlusion.hand_boxes)
                self.detector.submit(ai_frame.copy())
                if self.detector.persons >= 2: raw.add("พบหลายคน")

                cheating_reasons = self.vfilter.update(raw, now)
                if self.detector.found_obj:
                    cheating_reasons.append(f"พบ{self.detector.obj_label or 'มือถือ/หนังสือ'}")
                if self.exam_active: self._track_violations(cheating_reasons, dt if 0 < dt < 1 else 0)

                # บันทึกค่าที่ AI วัดได้ (ตัวเลขล้วน) ไว้ตรวจย้อนหลัง/จูนเกณฑ์
                try:
                    if self.exam_active:
                        if self._metrics is None:
                            self._metrics = MetricsLogger(os.path.join(BASE_DIR, "evidence"), self.controller.username)
                        ly, lp, lg = self.face_ai.last
                        self._metrics.log(now, n_faces, self.detector.persons, ly, lp, lg, self.face_ai.head_label,
                                          len(self.occlusion.hand_boxes), self.detector.obj_label, cheating_reasons)
                    elif self._metrics is not None:
                        self._metrics.close(); self._metrics = None
                except Exception as ex:
                    print(f"⚠️ metrics log error: {ex}"); self._metrics = None

                if face_detection is None:
                    # AI ใบหน้าใช้ไม่ได้: แจ้งชัดๆ แทนที่จะให้กล้องค้างเงียบๆ
                    self.status_lbl.configure(text="⚠️ AI ตรวจใบหน้าไม่ทำงาน (ดูหน้าต่าง console)", text_color=AMBER_WARN)

                if cheating_reasons:
                    cheating = True; self.is_camera_ready = False
                    self.status_lbl.configure(text=f"⚠️ เตือน: {', '.join(cheating_reasons[:2])}", text_color=RED_ALERT)
                    if not self.exam_active: self.btn.configure(state="disabled", fg_color=BG_PANEL_2, text_color=TEXT_MUTED)
                elif face_detection is not None or self.exam_active:
                    self.is_camera_ready = True
                    if not self.exam_active:
                        self.status_lbl.configure(text="✅ พร้อมสอบ", text_color=GREEN_OK)
                        self.btn.configure(state="normal", text="พร้อมเริ่มการสอบ", fg_color=TEAL, text_color="#04121a")
                    else:
                        hl = self.face_ai.head_label
                        self.status_lbl.configure(text=f"🔵 กำลังสอบ... · {hl}" if hl else "🔵 กำลังสอบ...", text_color=CYAN)

                self.occlusion.draw(frame)
                if FaceAnalyzer.SHOW_ANGLES and self.face_ai.last != (0.0, 0.0, 0.0):
                    fa = self.face_ai; ly, lp, _ = fa.last
                    if fa.base is not None: txt = f"pitch {lp:.0f} (d {lp - fa.base[1]:.0f} / limit {fa.PITCH_DOWN_LIMIT:.0f})"
                    else: txt = f"pitch {lp:.0f} (limit {fa.PRE_PITCH_LIMIT:.0f})"
                    cv2.putText(frame, txt, (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
                # วาดกรอบวัตถุที่ตรวจพบลงภาพ (เห็นในจอและในคลิปหลักฐาน)
                for name, conf, (x1, y1, x2, y2) in list(self.detector.boxes):
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"{name} {conf:.2f}", (x1, max(15, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                # Record หน้าจอ
                if self.exam_active and self.out_screen and now >= self.next_screen_t:
                    self.next_screen_t += 1.0 / self.SCREEN_FPS
                    if now - self.next_screen_t > 2: self.next_screen_t = now   # ตามไม่ทัน ข้ามไป
                    try:
                        shot = cv2.cvtColor(np.array(self.sct.grab(self.sct.monitors[1])), cv2.COLOR_BGRA2BGR)
                        self.out_screen.write(cv2.resize(shot, self.screen_size, interpolation=cv2.INTER_AREA))
                    except Exception: pass

                if self.exam_active and cheating:
                    if not self.rec_cheat:
                        evidence_dir = os.path.join(BASE_DIR, "evidence"); os.makedirs(evidence_dir, exist_ok=True)
                        rec_fps = float(min(30, max(5, round(self.fps))))   # ใช้ fps จริง คลิปจะไม่เร็ว/ช้าผิดปกติ
                        self.out_webcam, self.cheat_file = open_video_writer(
                            os.path.join(evidence_dir, f"CHEAT_{self.controller.username}_{int(now)}"), rec_fps, (640, 480))
                        self.rec_cheat = True
                    self.stop_rec_time = now + 5
                    cv2.putText(frame, "VIOLATION!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                elif self.rec_cheat and now > self.stop_rec_time:
                    self._finish_cheat_clip()

                if self.rec_cheat and self.out_webcam: self.out_webcam.write(frame)

                # เวลาประมวลผลเฟรมจริงตั้งแต่เริ่ม update_cam จนพร้อมแสดงผล
                self._last_frame_latency_ms = (time.perf_counter() - _frame_perf_start) * 1000.0

                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=self.cam_display_size)
                self.cam_lbl.configure(image=ctk_img); self.cam_lbl.image = ctk_img
            except Exception as e:
                import traceback
                print("=" * 60)
                print(f"❌ update_cam ERROR: {e}")
                traceback.print_exc()
                print("=" * 60)
        self.after(10, self.update_cam)

    def _apply_sensitivity(self, level):
        prof = SENSITIVITY.get(level, SENSITIVITY["medium"])
        self.face_ai.set_sensitivity(prof["angle"])
        self.vfilter.set_sensitivity(prof["ratio"])
        self.detector.set_sensitivity(prof["conf"])
        print(f"⚙️ ความเข้ม AI: {prof['label']}")

    def _start_heartbeat(self):
        """ส่งสถานะสดให้อาจารย์ทุก 1 วินาทีในเธรดแยก (เน็ตช้าก็ไม่ทำให้กล้อง/หน้าจอค้าง)"""
        self.kicked = False; self.hb_token = getattr(self, "hb_token", 0) + 1; token = self.hb_token
        def loop():
            fails = 0
            while self.exam_active and token == self.hb_token:
                try:
                    d = api.call("/api/heartbeat", {"live_status": ", ".join(self.live_reasons),
                                                    "exam_id": self.current_exam_id}, timeout=5)
                    fails = 0
                    if d.get("kicked"): self.kicked = True; return
                except Exception as ex:
                    fails += 1
                    if fails in (3, 30): print(f"⚠️ heartbeat ส่งไม่ได้ {fails} ครั้งติด: {ex}")
                time.sleep(1.0)
        threading.Thread(target=loop, daemon=True).start()

    def _lockdown_alert(self, reason):
        """เหตุผลจากระบบล็อกเครื่อง (คีย์ลัด/จอแยก/รีโมท) — ให้ AI นับเป็นพฤติกรรมเสี่ยงและส่งอาจารย์"""
        now = time.time()
        self.lockdown_alerts.append((now, reason))
        if now - self.last_logged.get(reason, 0) < 5: return
        self.last_logged[reason] = now
        args = (self.current_exam_id, self.controller.user_id, self.controller.username, reason)
        threading.Thread(target=lambda a=args: self._safe_log(a), daemon=True).start()

    def _start_security_monitor(self):
        """ตรวจจอแยก/โปรแกรมรีโมทซ้ำทุก ~5 วินาทีระหว่างสอบ (บางอย่างเปิดกลางคันได้)"""
        self.sec_token = getattr(self, "sec_token", 0) + 1; token = self.sec_token
        def loop():
            while self.exam_active and token == self.sec_token:
                try:
                    if self.lockdown.monitor_count() > 1:
                        self._safe_after(lambda: self._lockdown_alert("เปิดจอภาพที่สอง"))
                    for app_name in self.lockdown.scan_remote_apps():
                        self._safe_after(lambda a=app_name: self._lockdown_alert(f"เปิดโปรแกรม {a}"))
                except Exception as ex:
                    print(f"⚠️ security monitor: {ex}")
                for _ in range(50):   # รวม ~5 วินาที แต่ออกไวถ้าเลิกสอบ
                    if not self.exam_active or token != self.sec_token: return
                    time.sleep(0.1)
        threading.Thread(target=loop, daemon=True).start()

    def _safe_after(self, fn):
        try: self.after(0, fn)
        except Exception: pass

    def _track_violations(self, reasons, dt):
        """สะสมเวลาที่ทำผิดกฎ (ใช้คิดเปอร์เซ็นต์ความเสี่ยง) และบันทึกเหตุการณ์ใหม่ให้อาจารย์เห็นทันที"""
        keys = {risk_key(r) for r in reasons}
        now0 = time.time()   # แสดงเหตุจากระบบล็อกเครื่อง (คีย์ลัด/จอ/รีโมท) ที่เพิ่งเกิดใน 3 วินาทีล่าสุดด้วย
        for t0, r in list(self.lockdown_alerts):
            if now0 - t0 <= 3: keys.add(risk_key(r))
        self.live_reasons = sorted(keys)
        if keys:
            self.flag_seconds += dt
            self.risk_seconds += dt * max(RISK_WEIGHTS.get(k, 0.5) for k in keys)   # ไม่นับซ้ำถ้าผิดหลายอย่างพร้อมกัน
            for k in keys: self.viol_seconds[k] += dt
        now = time.time()
        for k in keys - self.prev_reasons:
            if now - self.last_logged.get(k, 0) < 5: continue    # กันบันทึกถี่เกินไปเวลาสถานะกะพริบ
            self.last_logged[k] = now
            args = (self.current_exam_id, self.controller.user_id, self.controller.username, k)
            threading.Thread(target=lambda a=args: self._safe_log(a), daemon=True).start()
        self.prev_reasons = keys

    @staticmethod
    def _safe_log(args):
        exam_id, _uid, _name, reason = args
        try: api.call("/api/violation", {"exam_id": exam_id, "reason": reason}, timeout=5)
        except Exception as ex: print(f"⚠️ log_violation error: {ex}")

    @staticmethod
    def _quiet_call(path, payload=None):
        try: api.call(path, payload, timeout=5)
        except Exception: pass

    def _finish_cheat_clip(self):
        self.rec_cheat = False
        if self.out_webcam is not None:
            self.out_webcam.release(); self.out_webcam = None
            upload_to_teacher(self.cheat_file)

# ================= Teacher Dashboard =================
# ================= รายงาน / วิเคราะห์ข้อสอบ / Export =================
def _difficulty(p):
    if p >= 0.80: return "ง่ายมาก"
    if p >= 0.60: return "ง่าย"
    if p >= 0.40: return "ปานกลาง"
    if p >= 0.20: return "ยาก"
    return "ยากมาก"

def _disc_label(r):
    if r < 0: return "ติดลบ — ควรตรวจเฉลย"
    if r >= 0.40: return "ดีมาก"
    if r >= 0.20: return "ใช้ได้"
    return "ต่ำ — ควรปรับปรุง"

def _fmt_secs(sec):
    sec = int(round(sec or 0)); m, s = divmod(sec, 60)
    return f"{m} นาที {s} วินาที" if m else f"{s} วินาที"

# ================= หลักฐาน (คลิปที่นักเรียนส่งมา) =================
def list_evidence(username):
    """ไฟล์คลิปของนักเรียนคนนี้ใน Submitted_Exams (CHEAT ก่อน แล้ว FULL) ใหม่สุดอยู่บน"""
    safe = secure_filename(username or "")
    try:
        names = os.listdir(UPLOAD_FOLDER)
    except OSError:
        return []
    out = []
    for n in names:
        if not n.lower().endswith((".mp4", ".avi", ".mkv", ".webm")):
            continue
        if username and username not in n and (not safe or safe not in n):
            continue
        out.append(os.path.join(UPLOAD_FOLDER, n))
    out.sort(key=lambda p: (0 if os.path.basename(p).upper().startswith("CHEAT") else 1,
                            -os.path.getmtime(p) if os.path.exists(p) else 0))
    return out

def count_evidence(username):
    return len(list_evidence(username))

def open_file(path):
    """เปิดไฟล์ด้วยโปรแกรมเริ่มต้นของเครื่อง (วิดีโอเพลเยอร์)"""
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception as ex:
        try: messagebox.showerror("เปิดไฟล์ไม่ได้", f"{ex}\n\nไฟล์อยู่ที่:\n{path}")
        except Exception: print(f"open_file error: {ex}")

def open_evidence_folder():
    try:
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        open_file(UPLOAD_FOLDER)
    except Exception as ex:
        print(f"open folder error: {ex}")


def build_exam_report(exam_id):
    """รวมข้อมูลทั้งหมดของชุดข้อสอบหนึ่งชุด ใช้ทั้งหน้าจอรายงานและไฟล์ Export"""
    info = db.get_exam_info(exam_id) or {"name": f"exam {exam_id}", "code": ""}
    questions = [q for q in db.get_questions_by_exam(exam_id) if q['type'] == 'MCQ']
    results = db.get_exam_results(exam_id)
    answers = db.get_exam_answers(exam_id)
    violations = db.get_exam_violations(exam_id)

    # ---- รายบุคคล ----
    students = []
    for r in results:
        mx = r['max']
        score_txt = f"{r['score']:g}/{mx:g}" if mx else f"{r['score']:g} คะแนน"
        pct = (100.0 * r['score'] / mx) if mx else None
        detail_parts = [f"{k} {_fmt_secs(v)}" for k, v in sorted(r['detail'].items(), key=lambda kv: -kv[1])]
        detail_txt = ""
        if r['exam_seconds']:
            detail_txt = (f"เวลาสอบ {_fmt_secs(r['exam_seconds'])} · ถูกตรวจพบรวม {_fmt_secs(r['flagged_seconds'])} "
                          f"({100.0 * (r['flagged_seconds'] or 0) / r['exam_seconds']:.1f}% ของเวลา) · แจ้งเตือน {r['violations']} ครั้ง")
            if detail_parts: detail_txt += "\n" + " · ".join(detail_parts)
        students.append({**r, "score_txt": score_txt, "pct": pct, "detail_txt": detail_txt,
                         "risk_level": risk_level(r['risk'])[0]})

    # ---- Item analysis ----
    by_result = {}
    for a in answers: by_result.setdefault(a['result_id'], {})[a['question_id']] = a
    scored = [r for r in results if r['id'] in by_result]
    ranked = sorted(scored, key=lambda r: r['score'], reverse=True)
    k = max(1, round(len(ranked) * 0.27)) if len(ranked) >= 4 else 0
    upper = {r['id'] for r in ranked[:k]}; lower = {r['id'] for r in ranked[-k:]} if k else set()

    items = []
    for no, q in enumerate(questions, start=1):
        options = [o for o in (q['options'] or "").split('|') if o.strip()]
        it = {"no": no, "id": q['id'], "text": q['text'], "type": q['type'], "options": options,
              "correct_idx": normalize_mcq_answer(str(q['correct'] or ""), options) if q['type'] == 'MCQ' else None,
              "answered": 0, "blank": 0, "correct_n": 0, "p": None, "difficulty": "-", "r": None, "disc_label": "-",
              "dist": [0] * len(options)}
        rows = [by_result[r['id']].get(q['id']) for r in scored]
        rows_by_rid = {r['id']: by_result[r['id']].get(q['id']) for r in scored}
        for a in rows:
            if a is None: continue
            if not (a['answer'] or "").strip(): it['blank'] += 1; continue
            it['answered'] += 1
            if q['type'] == 'MCQ':
                try:
                    idx = int(a['answer'])
                    if 1 <= idx <= len(options): it['dist'][idx - 1] += 1
                except ValueError: pass
                if a['correct'] == 1: it['correct_n'] += 1
        n = it['answered'] + it['blank']   # ไม่ตอบ = ตอบผิด
        if q['type'] == 'MCQ' and n:
            it['p'] = it['correct_n'] / n; it['difficulty'] = _difficulty(it['p'])
            if k:
                pu = sum(1 for rid in upper if rows_by_rid.get(rid) and rows_by_rid[rid]['correct'] == 1) / k
                pl = sum(1 for rid in lower if rows_by_rid.get(rid) and rows_by_rid[rid]['correct'] == 1) / k
                it['r'] = pu - pl; it['disc_label'] = _disc_label(it['r'])
        items.append(it)

    scores = [r['score'] for r in results]
    summary = {"n": len(results), "n_with_answers": len(scored),
               "avg_txt": f"{sum(scores) / len(scores):.2f}" if scores else "-",
               "minmax_txt": f"{max(scores):g} / {min(scores):g}" if scores else "-",
               "high_risk": sum(1 for r in results if (r['risk'] or 0) >= 15)}
    return {"exam_name": info['name'], "exam_code": info['code'], "questions": questions, "students": students,
            "items": items, "violations": violations, "summary": summary, "by_result": by_result}

def _report_tables(rep):
    """ข้อมูลแต่ละแผ่นงานในรูป (ชื่อแผ่น, หัวตาราง, แถว)"""
    qs = rep['questions']
    t1 = ("ผลคะแนน", ["ชื่อนักเรียน", "เวลาส่ง", "คะแนน", "คะแนนเต็ม(ปรนัย)", "ร้อยละ", "ความเสี่ยงทุจริต(%)",
                      "ระดับความเสี่ยง", "เวลาสอบ(วินาที)", "เวลาที่ถูกตรวจพบ(วินาที)", "จำนวนแจ้งเตือน", "รายละเอียดพฤติกรรม"],
          [[s['name'], s['time'], s['score'], s['max'], round(s['pct'], 1) if s['pct'] is not None else None,
            s['risk'], s['risk_level'], s['exam_seconds'], s['flagged_seconds'], s['violations'],
            ", ".join(f"{k} {v:.0f}s" for k, v in s['detail'].items())] for s in rep['students']])
    head2 = ["ชื่อนักเรียน"] + [f"ข้อ {i}" for i in range(1, len(qs) + 1)] + ["คะแนน"]
    rows2 = []
    for s in rep['students']:
        ans = rep['by_result'].get(s['id'], {})
        row = [s['name']]
        for q in qs:
            a = ans.get(q['id'])
            if a is None: row.append("")
            else: row.append(f"{a['answer'] or '-'} {'✓' if a['correct'] == 1 else '✗'}")
        rows2.append(row + [s['score']])
    t2 = ("คำตอบรายข้อ", head2, rows2)
    t3 = ("วิเคราะห์ข้อสอบ", ["ข้อ", "คำถาม", "ประเภท", "เฉลย(ข้อ)", "ผู้ตอบ", "ไม่ตอบ", "ตอบถูก", "ร้อยละที่ตอบถูก",
                             "ระดับความยาก", "อำนาจจำแนก(r)", "การแปลผล r", "การกระจายคำตอบ"],
          [[it['no'], it['text'], it['type'], it['correct_idx'], it['answered'], it['blank'], it['correct_n'],
            round(it['p'] * 100, 1) if it['p'] is not None else None, it['difficulty'],
            round(it['r'], 2) if it['r'] is not None else None, it['disc_label'],
            " | ".join(f"{i}) {op}: {n}" for i, (op, n) in enumerate(zip(it['options'], it['dist']), start=1))]
           for it in rep['items']])
    t4 = ("ประวัติพฤติกรรม", ["เวลา", "ชื่อนักเรียน", "พฤติกรรม"],
          [[(v['time'] or "")[:19], v['name'], v['reason']] for v in rep['violations']])
    return [t1, t2, t3, t4]

def _write_xlsx(path, sheets, fills_by_sheet=None, widths_by_sheet=None):
    """
    เขียนไฟล์ .xlsx ด้วยไลบรารีมาตรฐานของ Python ล้วนๆ (zipfile) ไม่ต้องติดตั้ง openpyxl
    sheets = [(ชื่อแผ่น, หัวตาราง หรือ None, แถว)], fills_by_sheet = {ชื่อแผ่น: {(row, col): "RRGGBB"}} (เริ่มนับ 0)
    """
    import zipfile
    from xml.sax.saxutils import escape
    fills_by_sheet = fills_by_sheet or {}; widths_by_sheet = widths_by_sheet or {}

    colors = []   # สีพื้นหลังที่ใช้ทั้งหมด -> index ของ style
    for f in fills_by_sheet.values():
        for c in f.values():
            if c not in colors: colors.append(c)
    # style 0 = ปกติ, 1 = หัวตาราง, 2 = ตัวหนา, 3+ = พื้นสีตามลำดับ colors
    fills_xml = '<fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill>'
    fills_xml += '<fill><patternFill patternType="solid"><fgColor rgb="FF0E7490"/></patternFill></fill>'
    for c in colors: fills_xml += f'<fill><patternFill patternType="solid"><fgColor rgb="FF{c}"/></patternFill></fill>'
    xfs = ('<xf numFmtId="0" fontId="0" fillId="0" borderId="0"/>'
           '<xf numFmtId="0" fontId="1" fillId="2" borderId="0" applyFont="1" applyFill="1" applyAlignment="1">'
           '<alignment horizontal="center" vertical="center" wrapText="1"/></xf>'
           '<xf numFmtId="0" fontId="2" fillId="0" borderId="0" applyFont="1"/>')
    for i, _ in enumerate(colors): xfs += f'<xf numFmtId="0" fontId="0" fillId="{3 + i}" borderId="0" applyFill="1"/>'
    styles = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
              '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
              '<fonts count="3"><font><sz val="11"/><name val="Tahoma"/></font>'
              '<font><b/><sz val="11"/><color rgb="FFFFFFFF"/><name val="Tahoma"/></font>'
              '<font><b/><sz val="11"/><name val="Tahoma"/></font></fonts>'
              f'<fills count="{3 + len(colors)}">{fills_xml}</fills>'
              '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
              '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
              f'<cellXfs count="{3 + len(colors)}">{xfs}</cellXfs>'
              '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>')

    def col_name(i):
        s = ""; i += 1
        while i: i, r = divmod(i - 1, 26); s = chr(65 + r) + s
        return s

    def cell(r, c, v, style):
        ref = f"{col_name(c)}{r + 1}"; st = f' s="{style}"' if style else ""
        if v is None or v == "": return f'<c r="{ref}"{st}/>'
        if isinstance(v, bool): v = int(v)
        if isinstance(v, (int, float)): return f'<c r="{ref}"{st}><v>{v}</v></c>'
        txt = escape(str(v)).replace('"', "&quot;")
        return f'<c r="{ref}"{st} t="inlineStr"><is><t xml:space="preserve">{txt}</t></is></c>'

    sheet_xml = []
    for name, head, rows in sheets:
        fills = fills_by_sheet.get(name, {})
        all_rows = ([head] if head else []) + list(rows)
        ncol = max([len(r) for r in all_rows] + [1])
        widths = widths_by_sheet.get(name) or [
            min(60, max(10, max(len(str(r[i])) for r in all_rows if i < len(r) and r[i] is not None) + 2
                        if any(i < len(r) and r[i] is not None for r in all_rows) else 10)) for i in range(ncol)]
        cols = "".join(f'<col min="{i + 1}" max="{i + 1}" width="{w}" customWidth="1"/>' for i, w in enumerate(widths))
        body = []
        for ri, r in enumerate(all_rows):
            cells = []
            for ci, v in enumerate(r):
                if head and ri == 0: style = 1
                elif (ri, ci) in fills: style = 3 + colors.index(fills[(ri, ci)])
                elif not head and ci == 0: style = 2
                else: style = 0
                cells.append(cell(ri, ci, v, style))
            body.append(f'<row r="{ri + 1}">{"".join(cells)}</row>')
        pane = ('<sheetViews><sheetView workbookViewId="0"><pane xSplit="1" ySplit="1" topLeftCell="B2" '
                'activePane="bottomRight" state="frozen"/></sheetView></sheetViews>') if head else ""
        sheet_xml.append('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                         '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                         f'{pane}<cols>{cols}</cols><sheetData>{"".join(body)}</sheetData></worksheet>')

    n = len(sheets)
    def safe_name(s):
        for ch in '[]:*?/\\': s = s.replace(ch, " ")
        return escape(s[:31])
    ct = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
          '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
          '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
          '<Default Extension="xml" ContentType="application/xml"/>'
          '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
          '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
          + "".join(f'<Override PartName="/xl/worksheets/sheet{i + 1}.xml" '
                    'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' for i in range(n))
          + '</Types>')
    rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
            'Target="xl/workbook.xml"/></Relationships>')
    wb = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
          '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
          'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>'
          + "".join(f'<sheet name="{safe_name(s[0])}" sheetId="{i + 1}" r:id="rId{i + 1}"/>' for i, s in enumerate(sheets))
          + '</sheets></workbook>')
    wb_rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
               '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
               + "".join(f'<Relationship Id="rId{i + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
                         f'Target="worksheets/sheet{i + 1}.xml"/>' for i in range(n))
               + f'<Relationship Id="rId{n + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
                 'Target="styles.xml"/></Relationships>')
    tmp = path + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", ct); z.writestr("_rels/.rels", rels)
        z.writestr("xl/workbook.xml", wb); z.writestr("xl/_rels/workbook.xml.rels", wb_rels)
        z.writestr("xl/styles.xml", styles)
        for i, x in enumerate(sheet_xml): z.writestr(f"xl/worksheets/sheet{i + 1}.xml", x)
    os.replace(tmp, path)   # เขียนเสร็จค่อยแทนที่ ถ้าไฟล์เดิมเปิดใน Excel อยู่จะ error ชัดเจน

def export_report(rep, path):
    """บันทึกเป็น .xlsx (หลายแผ่นงาน ไม่ต้องติดตั้งอะไรเพิ่ม) หรือ .csv (แยก 4 ไฟล์) คืนรายชื่อไฟล์ที่สร้าง"""
    import csv
    tables = _report_tables(rep)
    if path.lower().endswith(".csv"):
        base = path[:-4]; files = []
        for i, (name, head, rows) in enumerate(tables):
            fp = path if i == 0 else f"{base}_{name}.csv"
            with open(fp, "w", newline="", encoding="utf-8-sig") as f:   # utf-8-sig ให้ Excel อ่านภาษาไทยถูก
                w = csv.writer(f); w.writerow(head); w.writerows(rows)
            files.append(fp)
        return files
    if not path.lower().endswith(".xlsx"): path += ".xlsx"
    risk_fill = {"ต่ำ": "DCFCE7", "ปานกลาง": "FEF3C7", "สูง": "FEE2E2", "สูงมาก": "FCA5A5"}
    diff_fill = {"ง่ายมาก": "CFFAFE", "ง่าย": "CCFBF1", "ปานกลาง": "DCFCE7", "ยาก": "FEF3C7", "ยากมาก": "FEE2E2"}
    fills = {}
    for name, head, rows in tables:
        col = "ระดับความเสี่ยง" if name == "ผลคะแนน" else ("ระดับความยาก" if name == "วิเคราะห์ข้อสอบ" else None)
        if not col: continue
        ci = head.index(col); table = risk_fill if name == "ผลคะแนน" else diff_fill
        fills[name] = {(ri + 1, ci): table[r[ci]] for ri, r in enumerate(rows) if r[ci] in table}
    notes = [["ชุดข้อสอบ", rep['exam_name']], ["รหัสวิชา", rep['exam_code']],
             ["ส่งออกเมื่อ", time.strftime("%Y-%m-%d %H:%M")], ["จำนวนผู้ส่ง", rep['summary']['n']],
             ["คะแนนเฉลี่ย", rep['summary']['avg_txt']], [],
             ["ความเสี่ยงทุจริต(%)", "เวลาที่ AI ตรวจพบพฤติกรรมผิดกฎ × น้ำหนัก ÷ เวลาสอบ (ต่ำ <5, ปานกลาง 5-15, สูง 15-30, สูงมาก ≥30)"],
             ["", "เป็นสัญญาณให้ครูตรวจคลิปหลักฐานในโฟลเดอร์ Submitted_Exams ไม่ใช่ข้อพิสูจน์ว่าทุจริต"],
             ["ร้อยละที่ตอบถูก (p)", "≥80 ง่ายมาก, 60-79 ง่าย, 40-59 ปานกลาง, 20-39 ยาก, <20 ยากมาก"],
             ["อำนาจจำแนก (r)", "กลุ่มคะแนนสูง 27% − กลุ่มคะแนนต่ำ 27% (≥0.40 ดีมาก, 0.20-0.39 ใช้ได้, <0.20 ควรปรับปรุง, ติดลบ ควรตรวจเฉลย)"]]
    sheets = [("หมายเหตุ", None, notes)] + tables
    _write_xlsx(path, sheets, fills, {"หมายเหตุ": [22, 100]})
    return [path]

# ================= นำเข้าข้อสอบจากไฟล์ (Word / PDF / รูปภาพ / ข้อความ) =================
# อ่านข้อความจากไฟล์ แล้วแยกเป็นข้อสอบตาม "รูปแบบ" (ไม่ใช้ AI ภายนอก ไม่เสียเงิน ใช้ได้แม้ไม่มีเน็ต)
#   1. คำถาม...                  (หรือ "ข้อ 1 คำถาม...", "1) คำถาม...")
#   ก. ช้อยส์   ข. ช้อยส์ ...      (หรือ a. b. c. d. / (1) (2) (3) (4) / อยู่บรรทัดเดียวกันก็ได้)
#   เฉลย: ข                      (หรือใส่ * หน้าช้อยส์ที่ถูก, หรือมีตารางเฉลยท้ายไฟล์ "1. ข  2. ค ...")
#   ข้อที่ไม่มีช้อยส์จะถูกข้าม (ระบบใช้เฉพาะข้อสอบช้อยส์)
import re as _re

IMPORT_EXTS = (".docx", ".pdf", ".txt", ".png", ".jpg", ".jpeg", ".bmp", ".webp")
THAI_CHOICE = "กขคงจฉ"
LATIN_CHOICE = "abcdef"

IMPORT_FORMAT_HELP = """รูปแบบที่ระบบอ่านได้ (พิมพ์ใน Word หรือไฟล์ข้อความ):

1. 2 + 2 เท่ากับเท่าไร
ก. 3
ข. 4
ค. 5
ง. 6
เฉลย: ข

2. เมืองหลวงของประเทศไทยคือ   ก. เชียงใหม่   ข. กรุงเทพฯ   ค. ภูเก็ต
เฉลย ข

• เลขข้อ: "1." "1)" หรือ "ข้อ 1"
• ช้อยส์: ก. ข. ค. ง.  /  a. b. c. d.  /  (1) (2) (3) (4)
• บอกเฉลยได้ 3 แบบ: บรรทัด "เฉลย: ข" / ใส่ * หน้าช้อยส์ที่ถูก เช่น *ข. 4 / ตารางเฉลยท้ายไฟล์
  เช่น บรรทัด "เฉลย" แล้วตามด้วย "1. ข  2. ข  3. ค"
• ข้อที่ไม่มีช้อยส์ (ช้อยส์น้อยกว่า 2 ตัว) จะถูกข้าม"""


# ---------- อ่านข้อความจากไฟล์ ----------
def _docx_text(path):
    """อ่าน .docx ด้วย zipfile (ไม่ต้องติดตั้งเพิ่ม) รวมข้อความในตารางด้วย"""
    import zipfile
    from xml.etree import ElementTree as ET
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read("word/document.xml"))
    lines = []
    for p in root.iter(ns + "p"):
        parts = []
        for node in p.iter():
            if node.tag == ns + "t" and node.text: parts.append(node.text)
            elif node.tag == ns + "tab": parts.append("   ")
            elif node.tag in (ns + "br", ns + "cr"): parts.append("\n")
        # เลขข้ออัตโนมัติของ Word (numbering) ไม่ได้อยู่ในข้อความ — ใส่ตัวบอกไว้ให้ parser เดาเลขข้อ
        if p.find(f"{ns}pPr/{ns}numPr") is not None and parts:
            parts.insert(0, "⁣")
        lines.append("".join(parts))
    return "\n".join(lines)


_OCR_READER = None

def _ocr_reader():
    global _OCR_READER
    if _OCR_READER is None:
        try:
            import easyocr
        except ImportError:
            raise RuntimeError("อ่านรูป/PDF สแกนต้องติดตั้ง EasyOCR ก่อน:\n.venv\\Scripts\\python.exe -m pip install easyocr")
        # ครั้งแรกจะดาวน์โหลดโมเดลภาษาไทย+อังกฤษ (~100 MB) ต้องต่อเน็ต
        _OCR_READER = easyocr.Reader(["th", "en"], gpu=False, verbose=False)
    return _OCR_READER


def _ocr_image(img_bgr):
    """OCR รูปภาพ แล้วเรียงผลเป็นบรรทัด (บน→ล่าง ซ้าย→ขวา)"""
    h, w = img_bgr.shape[:2]
    if max(h, w) < 1400:   # ภาพเล็ก ขยายก่อน อ่านภาษาไทยแม่นขึ้น
        s = 1400.0 / max(h, w); img_bgr = cv2.resize(img_bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC)
    res = _ocr_reader().readtext(img_bgr, detail=1, paragraph=False)
    items = []
    for box, text, conf in res:
        ys = [p[1] for p in box]; xs = [p[0] for p in box]
        items.append({"y": (min(ys) + max(ys)) / 2, "h": max(ys) - min(ys), "x": min(xs), "t": text})
    items.sort(key=lambda it: (it["y"], it["x"]))
    lines, cur = [], []
    for it in items:   # กล่องที่อยู่แนวเดียวกัน (ต่างกันไม่เกินครึ่งความสูง) = บรรทัดเดียวกัน
        if cur and abs(it["y"] - cur[-1]["y"]) > max(8, 0.5 * max(it["h"], cur[-1]["h"])):
            lines.append(cur); cur = []
        cur.append(it)
    if cur: lines.append(cur)
    return "\n".join("   ".join(i["t"] for i in sorted(l, key=lambda i: i["x"])) for l in lines)


def extract_exam_text(path, progress=None):
    """คืน (ข้อความ, หมายเหตุ) จากไฟล์ .docx / .pdf / .txt / รูปภาพ"""
    ext = os.path.splitext(path)[1].lower()
    say = progress or (lambda msg: None)
    if ext == ".txt":
        raw = open(path, "rb").read()
        for enc in ("utf-8-sig", "cp874", "tis-620"):
            try: return raw.decode(enc), ""
            except UnicodeDecodeError: pass
        return raw.decode("utf-8", "ignore"), ""
    if ext == ".docx":
        say("กำลังอ่านไฟล์ Word..."); return _docx_text(path), ""
    if ext == ".pdf":
        try:
            import pymupdf
        except ImportError:
            try: import fitz as pymupdf
            except ImportError:
                raise RuntimeError("อ่าน PDF ต้องติดตั้ง PyMuPDF ก่อน:\n.venv\\Scripts\\python.exe -m pip install pymupdf")
        doc = pymupdf.open(path); out = []; ocr_pages = 0
        for i, page in enumerate(doc):
            say(f"กำลังอ่าน PDF หน้า {i + 1}/{len(doc)}...")
            txt = page.get_text("text", sort=True)
            if len(txt.strip()) >= 20:
                out.append(txt)
            else:   # หน้านี้เป็นภาพสแกน → OCR
                say(f"หน้า {i + 1} เป็นภาพสแกน กำลังอ่านด้วย OCR (อาจใช้เวลาสักครู่)...")
                pix = page.get_pixmap(dpi=200)
                img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR if pix.n == 4 else cv2.COLOR_RGB2BGR)
                out.append(_ocr_image(img)); ocr_pages += 1
        note = f"อ่านด้วย OCR {ocr_pages} หน้า — ควรตรวจคำผิดก่อนบันทึก" if ocr_pages else ""
        return "\n".join(out), note
    if ext in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
        say("กำลังโหลด OCR (ครั้งแรกจะดาวน์โหลดโมเดลภาษาไทย อาจใช้เวลา 1-2 นาที)...")
        _ocr_reader()
        say("กำลังอ่านตัวอักษรจากรูป...")
        img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)   # รองรับชื่อไฟล์ภาษาไทย
        if img is None: raise RuntimeError("เปิดรูปไม่ได้")
        return _ocr_image(img), "อ่านด้วย OCR — ควรตรวจคำผิดก่อนบันทึก"
    raise RuntimeError("รองรับเฉพาะ .docx .pdf .txt และรูปภาพ (.png .jpg)")


# ---------- แยกข้อสอบจากข้อความ ----------
_Q_RE = _re.compile(r"^\s*(?:ข้อ(?:ที่)?\s*(\d{1,3})\s*[\.\):]?|(\d{1,3})\s*[\.\)]|⁣)\s*(.*)$")
_CH_TOKEN = (r"(?:(?<=\s)|^)(\*?)\s*(?:\(\s*([{th}{lat}{LAT}1-6])\s*\)|([{th}{lat}{LAT}])\s*[\.\)])\s*"
             .format(th=THAI_CHOICE, lat=LATIN_CHOICE, LAT=LATIN_CHOICE.upper()))
_CH_RE = _re.compile(_CH_TOKEN)
_ANS_RE = _re.compile(r"^\s*(?:เฉลย|คำตอบที่ถูก(?:ต้อง)?|คำตอบ|ตอบ|answer|ans|key)\s*(?:ข้อ)?\s*[:：\.\-=]?\s*(.*)$", _re.I)
_KEY_HEAD_RE = _re.compile(r"^\s*(?:เฉลย(?:ข้อสอบ)?|answer\s*key|answers?)\s*[:：]?\s*$", _re.I)
_KEY_PAIR_RE = _re.compile(r"(\d{{1,3}})\s*[\.\)\-:=]\s*\(?\s*([{th}{lat}{LAT}1-6])\b\)?"
                           .format(th=THAI_CHOICE, lat=LATIN_CHOICE, LAT=LATIN_CHOICE.upper()))


def _choice_index(mark):
    m = (mark or "").strip().strip("().").lower()
    if not m: return None
    if m in THAI_CHOICE: return THAI_CHOICE.index(m) + 1
    if m in LATIN_CHOICE: return LATIN_CHOICE.index(m) + 1
    if m.isdigit(): return int(m)
    return None


def _split_choices(line):
    """คืน [(ติดดาวไหม, เลขช้อยส์, ข้อความ)] ถ้าบรรทัดนี้เป็นช้อยส์ (รองรับหลายช้อยส์ในบรรทัดเดียว)"""
    ms = list(_CH_RE.finditer(line))
    if not ms or line[:ms[0].start()].strip(): return []
    out = []
    for i, m in enumerate(ms):
        end = ms[i + 1].start() if i + 1 < len(ms) else len(line)
        idx = _choice_index(m.group(2) or m.group(3))
        text = line[m.end():end].strip()
        star = bool(m.group(1)) or text.endswith(("✓", "✔", "(ถูก)"))
        text = text.rstrip("✓✔ ").replace("(ถูก)", "").strip()
        out.append((star, idx, text))
    # ต้องเรียงต่อกัน (ก ข ค / 1 2 3) ถึงจะนับว่าเป็นช้อยส์ กันคำอย่าง "ก. " ในประโยคปกติ
    idxs = [o[1] for o in out]
    if idxs != list(range(idxs[0], idxs[0] + len(idxs))): return []
    return out


def parse_exam_text(text):
    """คืน (รายการข้อสอบ, คำเตือน) ข้อสอบ = {no, text, type, options[], answer(เลขข้อ/ข้อความ), star_answer}"""
    lines = [l.replace(" ", " ").rstrip() for l in (text or "").splitlines()]
    questions, key, warnings = [], {}, []
    cur = None; in_key = False; auto_no = 0

    def finish():
        if cur is None: return
        cur["text"] = cur["text"].strip()
        if cur["text"] or cur["options"]: questions.append(cur)

    for raw in lines:
        line = raw.strip()
        if not line: continue
        if _KEY_HEAD_RE.match(line):   # หัวข้อ "เฉลย" ท้ายไฟล์ → โหมดอ่านตารางเฉลย
            in_key = True; continue
        if in_key:
            pairs = _KEY_PAIR_RE.findall(line)
            if pairs:
                for n, mk in pairs: key[int(n)] = _choice_index(mk)
                continue
            in_key = False
        m = _ANS_RE.match(line)
        if m and cur is not None:
            val = m.group(1).strip()
            pairs = _KEY_PAIR_RE.findall(val)
            if len(pairs) >= 2:   # "เฉลย 1.ก 2.ข ..." ในบรรทัดเดียว
                for n, mk in pairs: key[int(n)] = _choice_index(mk)
                continue
            cur["answer_raw"] = val; continue
        qm = _Q_RE.match(line)
        ch = _split_choices(line) if cur is not None else []
        # ช้อยส์ชุดใหม่ต้องเริ่มที่ ก./a./(1) และต่อเนื่องจากตัวก่อนหน้า
        if ch and ch[0][1] != len(cur["options"]) + 1: ch = []
        if not ch and cur is not None and cur["options"]:
            # OCR มักอ่านจุดหลังตัวอักษรหาย เช่น "ค   ภูเก็ต" → ยอมรับถ้าเป็นตัวถัดไปพอดี
            mm = _re.match(r"^(\*?)\s*([{th}])\s{{2,}}(\S.*)$".format(th=THAI_CHOICE), line) or \
                 _re.match(r"^(\*?)\s*([{th}])\s+(\S.*)$".format(th=THAI_CHOICE), line)
            if mm and _choice_index(mm.group(2)) == len(cur["options"]) + 1:
                ch = [(bool(mm.group(1)), _choice_index(mm.group(2)), mm.group(3).strip())]
        if ch:
            for star, idx, t in ch:
                cur["options"].append(t)
                if star: cur["star_answer"] = len(cur["options"])
            continue
        if qm:
            finish()
            no = int(qm.group(1) or qm.group(2)) if (qm.group(1) or qm.group(2)) else auto_no + 1
            auto_no = no
            rest = qm.group(3) or ""
            cur = {"no": no, "text": "", "options": [], "answer_raw": "", "star_answer": None}
            inline = _split_choices(rest)
            if not inline:   # ช้อยส์อาจต่อท้ายคำถามในบรรทัดเดียว: "คำถาม  ก. x  ข. y"
                mm = _re.search(r"\s(\*?\s*(?:\([{th}{lat}1-6]\)|[{th}{lat}]\s*[\.\)]))".format(th=THAI_CHOICE, lat=LATIN_CHOICE), rest)
                cand = _split_choices(rest[mm.start():].strip()) if mm else []
                if cand and cand[0][1] == 1 and len(cand) >= 2:
                    inline = cand; rest = rest[:mm.start()]
                else:
                    inline = []
            if inline and inline[0][1] != 1: inline = []
            cur["text"] = rest.strip() if inline or not _split_choices(rest) else rest.strip()
            for star, idx, t in inline:
                cur["options"].append(t)
                if star: cur["star_answer"] = len(cur["options"])
            continue
        if cur is not None:   # บรรทัดต่อของคำถามหรือช้อยส์ก่อนหน้า
            if cur["options"]: cur["options"][-1] = (cur["options"][-1] + " " + line).strip()
            else: cur["text"] = (cur["text"] + " " + line).strip()
    finish()

    out = []
    for q in questions:
        mcq = len(q["options"]) >= 2
        ans = None
        if mcq:
            if q["star_answer"]: ans = q["star_answer"]
            elif q["no"] in key and key[q["no"]]: ans = key[q["no"]]
            elif q["answer_raw"]:
                first = q["answer_raw"].split()[0] if q["answer_raw"].split() else ""
                ans = _choice_index(first) or normalize_mcq_answer(q["answer_raw"], q["options"])
            if ans is not None and not (1 <= ans <= len(q["options"])): ans = None
            if ans is None: warnings.append(f"ข้อ {q['no']}: ไม่พบเฉลย")
        else:
            warnings.append(f"ข้อ {q['no']}: ไม่มีช้อยส์ — ข้าม"); continue
        clean = lambda x: " ".join((x or "").split()).replace("|", "/")
        out.append({"no": q["no"], "text": clean(q["text"]), "type": "MCQ",
                    "options": [clean(o) for o in q["options"]], "answer": ans})
    nos = [q["no"] for q in out]
    if nos and len(set(nos)) != len(nos): warnings.append("มีเลขข้อซ้ำกัน — ตรวจดูว่าแยกข้อถูกไหม")
    return out, warnings


class ImportExamWindow(ctk.CTkToplevel):
    """นำเข้าข้อสอบจากไฟล์: อ่านไฟล์ → แยกข้อ → ให้ครูตรวจ/แก้ → บันทึกเข้าชุดข้อสอบ"""
    def __init__(self, master, exam_id, exam_name, on_saved=None):
        super().__init__(master)
        self.exam_id = exam_id; self.on_saved = on_saved; self.cards = []; self.alive = True
        self.title("นำเข้าข้อสอบจากไฟล์"); self.geometry("1180x760"); self.configure(fg_color=BG_APP)
        self.protocol("WM_DELETE_WINDOW", self.close); self.after(200, lambda: self.alive and self.lift())

        head = ctk.CTkFrame(self, fg_color="transparent"); head.pack(fill="x", padx=16, pady=(14, 4))
        ctk.CTkLabel(head, text=f"📥 นำเข้าข้อสอบ → ชุด \"{exam_name}\"", font=FONT_H2, text_color=WHITE_TXT).pack(side="left")
        ctk.CTkButton(head, text="❔ รูปแบบที่รองรับ", width=140, fg_color=BG_PANEL_2, border_width=1, border_color=BORDER_SOFT,
                      text_color=TEXT_MAIN, command=lambda: messagebox.showinfo("รูปแบบข้อสอบที่อ่านได้", IMPORT_FORMAT_HELP, parent=self)
                      ).pack(side="right")
        ctk.CTkButton(head, text="📂 เลือกไฟล์ (Word / PDF / รูป)", width=220, fg_color=CYAN_DIM, hover_color=CYAN,
                      text_color=WHITE_TXT, font=FONT_SMALL_B, command=self.pick).pack(side="right", padx=8)

        self.status = ctk.CTkLabel(self, text="เลือกไฟล์ข้อสอบ (.docx .pdf .txt .png .jpg) หรือวางข้อความลงช่องซ้ายแล้วกด \"แยกข้อ\"",
                                   font=FONT_SMALL, text_color=TEXT_MUTED, anchor="w")
        self.status.pack(fill="x", padx=18)
        self.bar = ctk.CTkProgressBar(self, mode="indeterminate", height=6, progress_color=CYAN, fg_color=BG_PANEL)

        body = ctk.CTkFrame(self, fg_color="transparent"); body.pack(fill="both", expand=True, padx=14, pady=8)
        body.grid_columnconfigure(0, weight=2); body.grid_columnconfigure(1, weight=3); body.grid_rowconfigure(0, weight=1)
        left = ctk.CTkFrame(body, fg_color=BG_PANEL, corner_radius=12); left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        ctk.CTkLabel(left, text="① ข้อความที่อ่านได้ (แก้ไขได้)", font=FONT_SMALL_B, text_color=TEXT_MAIN).pack(anchor="w", padx=12, pady=(10, 4))
        self.raw = ctk.CTkTextbox(left, fg_color=BG_INPUT, text_color=TEXT_MAIN, font=("Tahoma", 13), wrap="word")
        self.raw.pack(fill="both", expand=True, padx=10)
        ctk.CTkButton(left, text="🔄 แยกข้อจากข้อความนี้", height=36, fg_color=CYAN_DIM, hover_color=CYAN, text_color=WHITE_TXT,
                      font=FONT_SMALL_B, command=self.reparse).pack(fill="x", padx=10, pady=10)

        right = ctk.CTkFrame(body, fg_color=BG_PANEL, corner_radius=12); right.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        rh = ctk.CTkFrame(right, fg_color="transparent"); rh.pack(fill="x", padx=12, pady=(10, 4))
        ctk.CTkLabel(rh, text="② ตรวจ/แก้ข้อสอบที่แยกได้", font=FONT_SMALL_B, text_color=TEXT_MAIN).pack(side="left")
        self.count_lbl = ctk.CTkLabel(rh, text="", font=FONT_SMALL_B, text_color=TEAL); self.count_lbl.pack(side="right")
        self.warn_lbl = ctk.CTkLabel(right, text="", font=FONT_SMALL, text_color=AMBER_WARN, wraplength=640, justify="left", anchor="w")
        self.warn_lbl.pack(fill="x", padx=12)
        self.lst = ctk.CTkScrollableFrame(right, fg_color=BG_PANEL_2, corner_radius=10); self.lst.pack(fill="both", expand=True, padx=10, pady=(4, 10))

        foot = ctk.CTkFrame(self, fg_color="transparent"); foot.pack(fill="x", padx=16, pady=(0, 14))
        self.save_btn = ctk.CTkButton(foot, text="💾 บันทึกเข้าชุดข้อสอบ", height=42, fg_color=TEAL, hover_color=TEAL_DIM,
                                      text_color="#04121a", font=FONT_BOLD, state="disabled", command=self.save)
        self.save_btn.pack(side="right")
        ctk.CTkButton(foot, text="ยกเลิก", height=42, width=100, fg_color="transparent", border_width=1, border_color=BORDER_SOFT,
                      text_color=TEXT_MUTED, command=self.close).pack(side="right", padx=8)

    # ---------- อ่านไฟล์ ----------
    def pick(self):
        path = filedialog.askopenfilename(parent=self, title="เลือกไฟล์ข้อสอบ", filetypes=[
            ("ไฟล์ข้อสอบ", "*.docx *.pdf *.txt *.png *.jpg *.jpeg *.bmp *.webp"), ("Word", "*.docx"), ("PDF", "*.pdf"),
            ("รูปภาพ", "*.png *.jpg *.jpeg *.bmp *.webp"), ("ข้อความ", "*.txt")])
        if not path: return
        if path.lower().endswith(".doc"):
            messagebox.showwarning("ไฟล์ .doc", "กรุณาเปิดใน Word แล้ว Save As เป็น .docx ก่อน", parent=self); return
        self._busy(True, f"กำลังอ่าน {os.path.basename(path)} ...")
        def work():
            try:
                txt, note = extract_exam_text(path, progress=lambda m: self.after(0, lambda: self.alive and self.status.configure(text=m)))
                self.after(0, lambda: self._loaded(txt, note, os.path.basename(path)))
            except Exception as ex:
                import traceback; traceback.print_exc()
                err = str(ex)   # เก็บไว้ก่อน: Python ลบตัวแปร ex ทิ้งเมื่อออกจาก except แต่ lambda รันทีหลัง
                self.after(0, lambda: self._failed(err))
        threading.Thread(target=work, daemon=True).start()

    def _busy(self, on, msg=""):
        if on:
            self.bar.pack(fill="x", padx=18, pady=(2, 0), after=self.status); self.bar.start()
            self.status.configure(text=msg, text_color=TEXT_MAIN); self.save_btn.configure(state="disabled")
        else:
            self.bar.stop(); self.bar.pack_forget()

    def _failed(self, msg):
        if not self.alive: return
        self._busy(False); self.status.configure(text=f"❌ อ่านไฟล์ไม่สำเร็จ: {msg.splitlines()[0]}", text_color=RED_ALERT)
        messagebox.showerror("อ่านไฟล์ไม่สำเร็จ", msg, parent=self)

    def _loaded(self, txt, note, name):
        if not self.alive: return
        self._busy(False)
        self.raw.delete("1.0", "end"); self.raw.insert("1.0", txt)
        self.status.configure(text=f"✅ อ่าน {name} แล้ว" + (f" · ⚠️ {note}" if note else ""),
                              text_color=AMBER_WARN if note else GREEN_OK)
        self.reparse()

    # ---------- แยกข้อ + แสดงให้ตรวจ ----------
    def reparse(self):
        qs, warns = parse_exam_text(self.raw.get("1.0", "end"))
        for w in self.lst.winfo_children(): w.destroy()
        self.cards = []
        for q in qs: self._card(q)
        self.count_lbl.configure(text=f"พบ {len(qs)} ข้อ")
        self.warn_lbl.configure(text=("⚠️ " + " · ".join(warns[:8]) + (" ..." if len(warns) > 8 else "")) if warns else "")
        if not qs:
            ctk.CTkLabel(self.lst, text="ยังแยกข้อสอบไม่ได้\nตรวจว่าขึ้นต้นข้อด้วย \"1.\" และช้อยส์ด้วย \"ก.\" (กด ❔ รูปแบบที่รองรับ)",
                         font=FONT_SUB, text_color=TEXT_MUTED).pack(pady=40)
        self.save_btn.configure(state="normal" if qs else "disabled")

    def _card(self, q):
        f = ctk.CTkFrame(self.lst, fg_color=BG_PANEL, corner_radius=8, border_width=1,
                         border_color=AMBER_WARN if (q["type"] == "MCQ" and not q["answer"]) else BORDER_SOFT)
        f.pack(fill="x", pady=4, padx=2)
        top = ctk.CTkFrame(f, fg_color="transparent"); top.pack(fill="x", padx=10, pady=(8, 2))
        inc = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(top, text=f"ข้อ {q['no']}", variable=inc, font=FONT_SMALL_B, text_color=TEXT_MAIN,
                        fg_color=CYAN, width=70).pack(side="left")
        qe = ctk.CTkEntry(f, height=32, fg_color=BG_INPUT, border_color=BORDER_SOFT, text_color=TEXT_MAIN)
        qe.insert(0, q["text"]); qe.pack(fill="x", padx=10, pady=4)
        card = {"inc": inc, "type": q["type"], "q": qe, "opts": [], "frame": f}
        for i, op in enumerate(q["options"], start=1):
            row = ctk.CTkFrame(f, fg_color="transparent"); row.pack(fill="x", padx=10, pady=1)
            ctk.CTkLabel(row, text=f"{THAI_CHOICE[i - 1] if i <= len(THAI_CHOICE) else i}.", width=24,
                         text_color=TEXT_MUTED).pack(side="left")
            e = ctk.CTkEntry(row, height=28, fg_color=BG_INPUT, border_color=BORDER_SOFT, text_color=TEXT_MAIN)
            e.insert(0, op); e.pack(side="left", fill="x", expand=True); card["opts"].append(e)
        arow = ctk.CTkFrame(f, fg_color="transparent"); arow.pack(fill="x", padx=10, pady=(4, 8))
        ctk.CTkLabel(arow, text="เฉลย:", font=FONT_SMALL_B, text_color=TEXT_MAIN).pack(side="left")
        values = ["— ยังไม่เลือก —"] + [f"{THAI_CHOICE[i - 1] if i <= len(THAI_CHOICE) else i}. {_short(op, 30)}"
                                       for i, op in enumerate(q["options"], start=1)]
        ans = ctk.CTkOptionMenu(arow, values=values, width=260, fg_color=BG_INPUT, button_color=CYAN_DIM, text_color=TEXT_MAIN,
                                command=lambda v, fr=f: fr.configure(border_color=AMBER_WARN if v.startswith("—") else BORDER_SOFT))
        ans.set(values[q["answer"]] if q["answer"] else values[0]); ans.pack(side="left", padx=6)
        card["ans"] = ans; card["values"] = values
        self.cards.append(card)

    # ---------- บันทึก ----------
    def save(self):
        rows, missing = [], []
        for i, c in enumerate(self.cards, start=1):
            if not c["inc"].get(): continue
            text = c["q"].get().strip()
            opts = [e.get().strip().replace("|", "/") for e in c["opts"]]
            opts = [o for o in opts if o]
            v = c["ans"].get()
            ans = c["values"].index(v) if v in c["values"] and not v.startswith("—") else None
            if ans is None or len(opts) < 2: missing.append(i); continue
            rows.append(("MCQ", text, "|".join(opts), str(ans)))
        if missing:
            messagebox.showwarning("ยังไม่ครบ", f"ข้อที่ {', '.join(map(str, missing[:15]))} ยังไม่ได้เลือกเฉลย (หรือมีช้อยส์ไม่ถึง 2 ตัว)\n"
                                   "เลือกเฉลยให้ครบ หรือเอาเครื่องหมายถูกหน้าข้อนั้นออกเพื่อข้าม", parent=self); return
        if not rows: messagebox.showwarning("ไม่มีข้อสอบ", "ไม่ได้เลือกข้อใดเลย", parent=self); return
        if not messagebox.askyesno("ยืนยัน", f"เพิ่ม {len(rows)} ข้อเข้าชุดข้อสอบนี้?", parent=self): return
        for t, text, opts, ans in rows: db.add_question(self.exam_id, t, text, opts, ans)
        messagebox.showinfo("สำเร็จ", f"เพิ่มข้อสอบ {len(rows)} ข้อเรียบร้อย", parent=self)
        if self.on_saved: self.on_saved()
        self.close()

    def close(self):
        self.alive = False; self.withdraw(); self.after(150, self.destroy)



class TeacherDashboard(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=BG_APP); self.controller = controller
        set_page_background(self)
        self.active_exam_id = None; self.edit_q_id = None; self.choice_widgets = []

        self.side = ctk.CTkScrollableFrame(self, width=280, fg_color=BG_PANEL, corner_radius=0,
                                            border_width=0, scrollbar_button_color=BG_PANEL_2)
        self.side.pack(side="left", fill="y")
        ctk.CTkLabel(self.side, text="🧭 แผงควบคุมอาจารย์", font=FONT_H2, text_color=WHITE_TXT).pack(pady=(24, 18))

        def section(title):
            ctk.CTkLabel(self.side, text=title, font=FONT_SMALL_B, text_color=TEXT_MUTED,
                         anchor="w").pack(fill="x", padx=16, pady=(14, 4))

        # --- Server status card ---
        section("การเชื่อมต่อ")
        ip_f = ctk.CTkFrame(self.side, fg_color=BG_PANEL_2, corner_radius=12, border_width=1, border_color=BORDER_SOFT)
        ip_f.pack(fill="x", padx=16, pady=2)
        ctk.CTkLabel(ip_f, text="📡 IP สำหรับนักเรียน", font=FONT_SMALL_B, text_color=TEXT_MUTED).pack(pady=(10, 0))
        self.ip_lbl = ctk.CTkLabel(ip_f, text="Loading...", font=FONT_SMALL_B, text_color=TEAL, justify="center"); self.ip_lbl.pack(pady=(2, 10))

        # --- Exam set management card ---
        section("ชุดข้อสอบ")
        set_card = ctk.CTkFrame(self.side, fg_color=BG_PANEL_2, corner_radius=12, border_width=1, border_color=BORDER_SOFT)
        set_card.pack(fill="x", padx=16, pady=2)
        self.exam_option = ctk.CTkOptionMenu(set_card, command=self.change_exam_set, width=216, corner_radius=8,
                                              fg_color=BG_INPUT, button_color=CYAN_DIM, button_hover_color=CYAN,
                                              text_color=TEXT_MAIN, dropdown_fg_color=BG_PANEL_2)
        self.exam_option.pack(padx=12, pady=(12, 8))
        ctk.CTkButton(set_card, text="＋ สร้างชุดใหม่", command=self.create_set, corner_radius=8, height=32,
                      fg_color=TEAL, hover_color=TEAL_DIM, text_color="#04121a", width=216).pack(padx=12, pady=(0, 6))
        self.lbl_code = ctk.CTkLabel(set_card, text="รหัสวิชา: -", font=FONT_MONO, text_color=AMBER_WARN)
        self.lbl_code.pack(pady=(4, 0))
        self.lbl_duration = ctk.CTkLabel(set_card, text="⏱ เวลาสอบ: -", font=FONT_MONO, text_color=TEAL)
        self.lbl_duration.pack(pady=(0, 8))
        ctk.CTkButton(set_card, text="⏱  ตั้งเวลาสอบ", command=self.set_duration, corner_radius=8, height=32,
                      fg_color=BG_INPUT, hover_color=CYAN_DIM, border_width=1, border_color=BORDER_SOFT,
                      text_color=TEXT_MAIN, width=216).pack(padx=12, pady=(0, 6))
        ctk.CTkLabel(set_card, text="🎚 ความเข้ม AI คุมสอบ", font=FONT_SMALL_B, text_color=TEXT_MUTED).pack(pady=(2, 0))
        self.sens_labels = {"ผ่อนปรน": "low", "ปกติ": "medium", "เข้มงวด": "high"}
        self.sens_option = ctk.CTkOptionMenu(set_card, values=list(self.sens_labels.keys()), width=216, corner_radius=8,
                                             fg_color=BG_INPUT, button_color=CYAN_DIM, button_hover_color=CYAN,
                                             text_color=TEXT_MAIN, dropdown_fg_color=BG_PANEL_2, command=self.set_sensitivity)
        self.sens_option.set("ปกติ"); self.sens_option.pack(padx=12, pady=(0, 8))
        self.btn_status = ctk.CTkButton(set_card, text="สถานะ: -", width=216, corner_radius=8, height=32,
                                         command=self.toggle_status, fg_color=BG_INPUT, text_color=TEXT_MAIN)
        self.btn_status.pack(padx=12, pady=(0, 12))

        # --- Actions ---
        section("การจัดการ")
        ctk.CTkButton(self.side, text="📊  รายงานผล & วิเคราะห์ข้อสอบ", command=self.show_sc, corner_radius=10, height=36,
                      fg_color=CYAN_DIM, hover_color=CYAN, text_color=WHITE_TXT, font=FONT_SMALL_B,
                      width=216).pack(fill="x", padx=16, pady=3)
        ctk.CTkButton(self.side, text="📤  Export ผลคะแนน (Excel/CSV)", command=self.export_results, corner_radius=10,
                      height=36, fg_color=TEAL, hover_color=TEAL_DIM, text_color="#04121a", font=FONT_SMALL_B,
                      width=216).pack(fill="x", padx=16, pady=3)
        ctk.CTkButton(self.side, text="🗑  ลบชุดข้อสอบ", command=self.del_set, corner_radius=10, height=36,
                      fg_color="transparent", hover_color=RED_DIM, border_width=1, border_color=RED_ALERT,
                      text_color=RED_ALERT, font=FONT_SMALL_B, width=216).pack(fill="x", padx=16, pady=3)

        ctk.CTkButton(self.side, text="⚙  โปรไฟล์ & ตั้งค่า", command=lambda: SettingsWindow(self, controller),
                      corner_radius=10, fg_color=BG_PANEL_2, hover_color=CYAN_DIM, border_width=1, border_color=CYAN_DIM,
                      text_color=TEXT_MAIN, height=36).pack(fill="x", padx=16, pady=(24, 6))
        ctk.CTkButton(self.side, text="🚪  Logout", command=controller.logout, corner_radius=10,
                      fg_color="transparent", hover_color=BG_PANEL_2, border_width=1, border_color=BORDER_SOFT,
                      text_color=TEXT_MUTED, height=36).pack(fill="x", padx=16, pady=(0, 20))

        self.tab = ctk.CTkTabview(self, fg_color=BG_PANEL, corner_radius=14, segmented_button_fg_color=BG_PANEL_2,
                                   segmented_button_selected_color=CYAN, segmented_button_selected_hover_color=CYAN_DIM,
                                   segmented_button_unselected_color=BG_PANEL_2, text_color=TEXT_MAIN,
                                   border_width=1, border_color=BORDER_SOFT)
        self.tab.pack(side="right", fill="both", expand=True, padx=(10, 20), pady=16)
        self.t_mon = self.tab.add("📡 Monitor"); self.t_q = self.tab.add("📝 โจทย์ข้อสอบ"); self.t_s = self.tab.add("👥 รายชื่อนักเรียน")
        self.setup_mon(); self.setup_q(); self.setup_s(); self.mon_active = False

    def on_show(self):
        start_teacher_server()
        ts = get_tailscale_ip()
        self.ip_lbl.configure(text=f"ในวง: {get_local_ip()}" + (f"\nนอกวง (Tailscale): {ts}" if ts else ""))
        self.refresh_exam_list(); self.load_s(); self.clear_form()
        if not self.mon_active:
            self.mon_active = True; self.update_monitor_loop()

    def refresh_exam_list(self):
        exams = db.get_teacher_exams(self.controller.teacher_id)
        if not exams:
            self.exam_option.configure(values=["ไม่มีชุดข้อสอบ"]); self.exam_option.set("ไม่มีชุดข้อสอบ")
            self.active_exam_id = None; self.btn_status.configure(state="disabled")
            self.lbl_duration.configure(text="⏱ เวลาสอบ: -")
        else:
            names = [f"{e['name']} ({e['code']})" for e in exams]; self.exam_ids = {n: e for n, e in zip(names, exams)}
            self.exam_option.configure(values=names); self.exam_option.set(names[0]); self.change_exam_set(names[0])

    def change_exam_set(self, choice):
        if choice == "ไม่มีชุดข้อสอบ": return
        exam = self.exam_ids[choice]; self.active_exam_id = exam['id']; self.lbl_code.configure(text=f"รหัสวิชา: {exam['code']}")
        self.lbl_duration.configure(text=self._fmt_duration(exam.get('duration', 0)))
        lvl = exam.get('sensitivity', 'medium')
        self.sens_option.set({"low": "ผ่อนปรน", "medium": "ปกติ", "high": "เข้มงวด"}.get(lvl, "ปกติ"))
        self.update_status_btn(exam['active']); self.load_q(); self.btn_status.configure(state="normal")

    def _fmt_duration(self, minutes):
        return f"⏱ เวลาสอบ: {minutes} นาที" if minutes and minutes > 0 else "⏱ เวลาสอบ: ไม่จำกัด"

    def set_duration(self):
        if not self.active_exam_id: messagebox.showwarning("เตือน", "เลือกชุดข้อสอบก่อน"); return
        val = ctk.CTkInputDialog(text="ตั้งเวลาสอบ (นาที) — ใส่ 0 หากไม่จำกัดเวลา:", title="ตั้งเวลาสอบ").get_input()
        if val is None: return
        try:
            minutes = int(val.strip())
            if minutes < 0: raise ValueError
        except ValueError:
            messagebox.showerror("ผิดพลาด", "กรุณาใส่ตัวเลขจำนวนเต็มไม่ติดลบ"); return
        db.update_exam_duration(self.active_exam_id, minutes)
        self.lbl_duration.configure(text=self._fmt_duration(minutes))
        messagebox.showinfo("บันทึก", f"ตั้งเวลาสอบเป็น {minutes} นาที" if minutes > 0 else "ตั้งเป็นไม่จำกัดเวลาแล้ว")

    def set_sensitivity(self, label):
        if not self.active_exam_id:
            return
        level = self.sens_labels.get(label, "medium")
        db.update_exam_sensitivity(self.active_exam_id, level)
        if self.active_exam_id in [e['id'] for e in [self.exam_ids[k] for k in self.exam_ids]]:
            for k in self.exam_ids:
                if self.exam_ids[k]['id'] == self.active_exam_id: self.exam_ids[k]['sensitivity'] = level
        desc = {"low": "ผ่อนปรน — เตือนเฉพาะที่ชัดเจน เหมาะห้องแสงน้อย/กล้องไม่ดี",
                "medium": "ปกติ — สมดุลระหว่างจับได้กับเตือนผิด",
                "high": "เข้มงวด — ไวต่อการขยับ เหมาะสอบสำคัญ อาจเตือนบ่อยขึ้น"}[level]
        messagebox.showinfo("ตั้งความเข้ม AI", f"ตั้งเป็น \"{label}\" แล้ว\n\n{desc}")

    def create_set(self):
        n = ctk.CTkInputDialog(text="ชื่อชุดข้อสอบ:", title="New").get_input()
        if not n: return
        c = ctk.CTkInputDialog(text="รหัสเข้าสอบ:", title="Code").get_input()
        if not c: return
        d = ctk.CTkInputDialog(text="เวลาสอบ (นาที) — ใส่ 0 หากไม่จำกัดเวลา:", title="เวลาสอบ").get_input()
        try:
            duration = int(d.strip()) if d else 0
            if duration < 0: duration = 0
        except ValueError:
            duration = 0
        if db.create_exam_set(self.controller.teacher_id, n, c, duration): self.refresh_exam_list()
        else: messagebox.showerror("Error", "รหัสซ้ำ")

    def del_set(self):
        if self.active_exam_id and messagebox.askyesno("ยืนยัน", "ลบชุดข้อสอบนี้?"):
            for q in db.get_questions_by_exam(self.active_exam_id): media.delete_question_media(q['id'])
            db.delete_exam_set(self.active_exam_id); self.refresh_exam_list()

    def update_status_btn(self, status):
        col = GREEN_OK if status == 1 else RED_ALERT
        self.btn_status.configure(text="🟢 เปิดสอบ" if status == 1 else "🔴 ปิดสอบ", border_color=col, border_width=2,
                                  fg_color=BG_PANEL_2)

    def toggle_status(self):
        if not self.active_exam_id: return
        exams = db.get_teacher_exams(self.controller.teacher_id)
        curr = next(e['active'] for e in exams if e['id'] == self.active_exam_id)
        self.update_status_btn(db.toggle_exam_status(self.active_exam_id, curr))

    # ================= รายงานผลสอบ: คะแนน / ความเสี่ยง / วิเคราะห์ข้อสอบ / พฤติกรรม =================
    def show_evidence(self, username):
        """หน้าต่างรวมคลิปหลักฐานของนักเรียนคนนี้ — เปิดดูได้จากในโปรแกรมเลย"""
        files = list_evidence(username)
        win = ctk.CTkToplevel(self); win.title(f"หลักฐาน — {username}"); win.geometry("640x520")
        win.configure(fg_color=BG_APP); win.after(200, win.lift)
        ctk.CTkLabel(win, text=f"📹 คลิปหลักฐานของ {username}", font=FONT_H3, text_color=WHITE_TXT).pack(pady=(14, 2))
        ctk.CTkLabel(win, text="CHEAT = คลิปตอนตรวจพบพฤติกรรมเสี่ยง · FULL = อัดหน้าจอทั้งการสอบ",
                     font=FONT_SMALL, text_color=TEXT_MUTED).pack(pady=(0, 6))
        bar = ctk.CTkFrame(win, fg_color="transparent"); bar.pack(fill="x", padx=14)
        ctk.CTkButton(bar, text="📂 เปิดโฟลเดอร์หลักฐานทั้งหมด", height=32, fg_color=AMBER_WARN, hover_color="#b45309",
                      text_color="#1a1200", font=FONT_SMALL_B, command=open_evidence_folder).pack(side="left")
        lst = ctk.CTkScrollableFrame(win, fg_color=BG_PANEL, corner_radius=10); lst.pack(fill="both", expand=True, padx=14, pady=10)
        if not files:
            ctk.CTkLabel(lst, text="ยังไม่มีคลิปหลักฐานของนักเรียนคนนี้\n(คลิปจะถูกส่งมาระหว่าง/หลังสอบ ถ้าเครื่องนักเรียนเชื่อมต่อได้)",
                         font=FONT_SUB, text_color=TEXT_MUTED, justify="center").pack(pady=30); return
        for fp in files:
            name = os.path.basename(fp)
            is_cheat = name.upper().startswith("CHEAT")
            try: size_mb = os.path.getsize(fp) / 1e6
            except OSError: size_mb = 0
            try: when = time.strftime("%d/%m %H:%M", time.localtime(os.path.getmtime(fp)))
            except OSError: when = ""
            row = ctk.CTkFrame(lst, fg_color=BG_PANEL_2, corner_radius=8, border_width=1,
                               border_color=RED_ALERT if is_cheat else BORDER_SOFT)
            row.pack(fill="x", pady=3, padx=2)
            ctk.CTkLabel(row, text="⚠️ พบพฤติกรรมเสี่ยง" if is_cheat else "🖥 อัดหน้าจอ", font=FONT_SMALL_B,
                         text_color=RED_ALERT if is_cheat else TEAL, width=140, anchor="w").pack(side="left", padx=(10, 4), pady=8)
            ctk.CTkLabel(row, text=f"{when} · {size_mb:.1f} MB", font=FONT_SMALL, text_color=TEXT_MUTED).pack(side="left")
            ctk.CTkButton(row, text="▶ เปิดดู", width=90, height=28, corner_radius=6, font=FONT_SMALL_B,
                          fg_color=CYAN_DIM, hover_color=CYAN, text_color=WHITE_TXT,
                          command=lambda p=fp: open_file(p)).pack(side="right", padx=10, pady=6)

    def show_sc(self):
        if not self.active_exam_id: messagebox.showwarning("เตือน", "เลือกชุดข้อสอบก่อน"); return
        exam_id = self.active_exam_id
        rep = build_exam_report(exam_id)
        win = ctk.CTkToplevel(self); win.title("รายงานผลสอบ"); win.geometry("980x680"); win.configure(fg_color=BG_APP)
        win.after(200, win.lift)
        head = ctk.CTkFrame(win, fg_color="transparent"); head.pack(fill="x", padx=16, pady=(14, 4))
        ctk.CTkLabel(head, text=f"📊 รายงานผลสอบ — {rep['exam_name']}", font=FONT_H2, text_color=WHITE_TXT).pack(side="left")
        ctk.CTkButton(head, text="📤 Export Excel / CSV", command=lambda: self.export_results(exam_id), height=34,
                      corner_radius=8, fg_color=TEAL, hover_color=TEAL_DIM, text_color="#04121a",
                      font=FONT_SMALL_B).pack(side="right")

        s = rep['summary']
        stat = ctk.CTkFrame(win, fg_color="transparent"); stat.pack(fill="x", padx=16, pady=(4, 6))
        for label, value, col in [("ส่งแล้ว", f"{s['n']} คน", CYAN), ("คะแนนเฉลี่ย", s['avg_txt'], TEAL),
                                  ("สูงสุด / ต่ำสุด", s['minmax_txt'], TEXT_MAIN),
                                  ("ความเสี่ยงสูงขึ้นไป", f"{s['high_risk']} คน", RED_ALERT if s['high_risk'] else GREEN_OK)]:
            c = ctk.CTkFrame(stat, fg_color=BG_PANEL, corner_radius=10, border_width=1, border_color=BORDER_SOFT)
            c.pack(side="left", fill="x", expand=True, padx=4)
            ctk.CTkLabel(c, text=label, font=FONT_SMALL, text_color=TEXT_MUTED).pack(pady=(8, 0))
            ctk.CTkLabel(c, text=value, font=FONT_MONO, text_color=col).pack(pady=(0, 8))

        tabs = ctk.CTkTabview(win, fg_color=BG_PANEL, segmented_button_selected_color=CYAN,
                              segmented_button_fg_color=BG_PANEL_2, segmented_button_unselected_color=BG_PANEL_2,
                              text_color=TEXT_MAIN)
        tabs.pack(fill="both", expand=True, padx=14, pady=(0, 14))
        t1 = tabs.add("👤 คะแนน & ความเสี่ยงทุจริต"); t2 = tabs.add("🧪 วิเคราะห์ข้อสอบ"); t3 = tabs.add("🚨 ประวัติพฤติกรรม")
        self._report_students(t1, rep); self._report_items(t2, rep); self._report_violations(t3, rep)

    def _report_students(self, parent, rep):
        sc = ctk.CTkScrollableFrame(parent, fg_color=BG_PANEL_2, corner_radius=10); sc.pack(fill="both", expand=True)
        ctk.CTkLabel(sc, text="ความเสี่ยงทุจริต (%) = เวลาที่ AI ตรวจพบพฤติกรรมผิดกฎ × น้ำหนักความร้ายแรง ÷ เวลาสอบทั้งหมด "
                              "— ใช้เป็นสัญญาณให้ครูตรวจสอบคลิปหลักฐาน ไม่ใช่ข้อพิสูจน์ว่าโกง",
                     font=FONT_SMALL, text_color=TEXT_MUTED, wraplength=860, justify="left").pack(anchor="w", padx=8, pady=(6, 8))
        if not rep['students']:
            ctk.CTkLabel(sc, text="ยังไม่มีคนส่ง", font=FONT_SUB, text_color=TEXT_MUTED).pack(pady=30); return
        for st in rep['students']:
            f = ctk.CTkFrame(sc, fg_color=BG_PANEL, corner_radius=8, border_width=1, border_color=BORDER_SOFT)
            f.pack(fill="x", pady=3, padx=2)
            top = ctk.CTkFrame(f, fg_color="transparent"); top.pack(fill="x", padx=12, pady=(8, 2))
            ctk.CTkLabel(top, text=st['name'], font=FONT_BOLD, text_color=TEXT_MAIN).pack(side="left")
            ctk.CTkLabel(top, text=f"  {st['time'] or ''}", font=FONT_SMALL, text_color=TEXT_MUTED).pack(side="left")
            lvl, col = risk_level(st['risk'])
            risk_txt = "ไม่มีข้อมูล (ส่งก่อนอัปเดตระบบ)" if st['risk'] is None else f"ความเสี่ยง {st['risk']:.1f}% · {lvl}"
            n_ev = count_evidence(st['name'])
            ev_txt = f"📹 หลักฐาน ({n_ev})" if n_ev else "📹 หลักฐาน"
            ctk.CTkButton(top, text=ev_txt, width=120, height=28, corner_radius=6, font=FONT_SMALL_B,
                          fg_color=BG_INPUT, hover_color=CYAN_DIM, border_width=1,
                          border_color=CYAN_DIM if n_ev else BORDER_SOFT, text_color=CYAN if n_ev else TEXT_MUTED,
                          command=lambda nm=st['name']: self.show_evidence(nm)).pack(side="right", padx=(8, 0))
            ctk.CTkLabel(top, text=risk_txt, font=FONT_SMALL_B, text_color=col).pack(side="right", padx=(0, 8))
            ctk.CTkLabel(top, text=f"{st['score_txt']}   ", font=FONT_MONO, text_color=TEAL).pack(side="right")
            if st['risk'] is not None:
                bar = ctk.CTkProgressBar(f, height=8, progress_color=col, fg_color=BG_INPUT)
                bar.set(min(1.0, st['risk'] / 50.0)); bar.pack(fill="x", padx=12, pady=(2, 2))
            if st['detail_txt']:
                ctk.CTkLabel(f, text=st['detail_txt'], font=FONT_SMALL, text_color=TEXT_MUTED, wraplength=860,
                             justify="left").pack(anchor="w", padx=12, pady=(0, 8))
            else:
                ctk.CTkFrame(f, height=6, fg_color="transparent").pack()

    def _report_items(self, parent, rep):
        sc = ctk.CTkScrollableFrame(parent, fg_color=BG_PANEL_2, corner_radius=10); sc.pack(fill="both", expand=True)
        ctk.CTkLabel(sc, text=f"คิดจากผู้สอบที่มีคำตอบรายข้อ {rep['summary']['n_with_answers']} คน · "
                              "ค่า p = สัดส่วนที่ตอบถูก (ความยากง่าย) · ค่า r = อำนาจจำแนก (กลุ่มเก่ง 27% − กลุ่มอ่อน 27%)",
                     font=FONT_SMALL, text_color=TEXT_MUTED, wraplength=860, justify="left").pack(anchor="w", padx=8, pady=(6, 8))
        if not rep['items']:
            ctk.CTkLabel(sc, text="ยังไม่มีโจทย์", font=FONT_SUB, text_color=TEXT_MUTED).pack(pady=30); return
        for it in rep['items']:
            f = ctk.CTkFrame(sc, fg_color=BG_PANEL, corner_radius=8, border_width=1, border_color=BORDER_SOFT)
            f.pack(fill="x", pady=3, padx=2)
            top = ctk.CTkFrame(f, fg_color="transparent"); top.pack(fill="x", padx=12, pady=(8, 2))
            ctk.CTkLabel(top, text=f"ข้อ {it['no']}", font=FONT_SMALL_B, text_color=WHITE_TXT, fg_color=CYAN_DIM,
                         corner_radius=6, width=48).pack(side="left", padx=(0, 8))
            ctk.CTkLabel(top, text=_short(it['text'] or "(ไม่มีข้อความ)", 60), font=FONT_BOLD, text_color=TEXT_MAIN,
                         anchor="w").pack(side="left", fill="x", expand=True)
            if it['p'] is None:
                ctk.CTkLabel(f, text="ยังไม่มีข้อมูลคำตอบ", font=FONT_SMALL, text_color=TEXT_MUTED).pack(anchor="w", padx=12, pady=(0, 8))
                continue
            col = {"ง่ายมาก": CYAN, "ง่าย": TEAL, "ปานกลาง": GREEN_OK, "ยาก": AMBER_WARN, "ยากมาก": RED_ALERT}[it['difficulty']]
            ctk.CTkLabel(top, text=f"ตอบถูก {it['p']*100:.0f}% · {it['difficulty']}", font=FONT_SMALL_B,
                         text_color=col).pack(side="right")
            bar = ctk.CTkProgressBar(f, height=8, progress_color=col, fg_color=BG_INPUT); bar.set(it['p'])
            bar.pack(fill="x", padx=12, pady=(2, 4))
            dist = "   ".join(f"{'✅' if i == it['correct_idx'] else '▫️'} {_short(op, 18)}: {n}"
                               for i, (op, n) in enumerate(zip(it['options'], it['dist']), start=1))
            disc = "-" if it['r'] is None else f"{it['r']:+.2f} ({it['disc_label']})"
            ctk.CTkLabel(f, text=f"ผู้ตอบ {it['answered']} · ไม่ตอบ {it['blank']} · อำนาจจำแนก r = {disc}\n{dist}",
                         font=FONT_SMALL, text_color=TEXT_MUTED, wraplength=860, justify="left").pack(anchor="w", padx=12, pady=(0, 8))

    def _report_violations(self, parent, rep):
        sc = ctk.CTkScrollableFrame(parent, fg_color=BG_PANEL_2, corner_radius=10); sc.pack(fill="both", expand=True)
        if not rep['violations']:
            ctk.CTkLabel(sc, text="ไม่มีพฤติกรรมเสี่ยงที่ถูกบันทึก", font=FONT_SUB, text_color=TEXT_MUTED).pack(pady=30); return
        for v in reversed(rep['violations'][-500:]):
            f = ctk.CTkFrame(sc, fg_color=BG_PANEL, corner_radius=6); f.pack(fill="x", pady=2, padx=2)
            ctk.CTkLabel(f, text=(v['time'] or "")[:19], font=FONT_SMALL, text_color=TEXT_MUTED, width=150,
                         anchor="w").pack(side="left", padx=(10, 4), pady=4)
            ctk.CTkLabel(f, text=v['name'], font=FONT_SMALL_B, text_color=TEXT_MAIN, width=160, anchor="w").pack(side="left")
            ctk.CTkLabel(f, text=f"⚠️ {v['reason']}", font=FONT_SMALL_B, text_color=RED_ALERT, anchor="w").pack(side="left", padx=6)

    def export_results(self, exam_id=None):
        exam_id = exam_id or self.active_exam_id
        if not exam_id: messagebox.showwarning("เตือน", "เลือกชุดข้อสอบก่อน"); return
        rep = build_exam_report(exam_id)
        safe = "".join(ch for ch in rep['exam_name'] if ch.isalnum() or ch in " _-").strip() or "exam"
        default = f"ผลสอบ_{safe}_{time.strftime('%Y%m%d_%H%M')}"
        types = [("Excel", "*.xlsx"), ("CSV (เปิดใน Excel ได้)", "*.csv")]
        path = filedialog.asksaveasfilename(title="บันทึกผลคะแนน", initialfile=default, defaultextension=".xlsx",
                                            filetypes=types)
        if not path: return
        try:
            files = export_report(rep, path)
        except Exception as ex:
            import traceback; traceback.print_exc()
            messagebox.showerror("Export ไม่สำเร็จ", str(ex)); return
        messagebox.showinfo("Export สำเร็จ", "บันทึกแล้ว:\n" + "\n".join(files))

    # ================= Monitor แบบเรียลไทม์ =================
    def setup_mon(self):
        self.mon_rows = {}; self.mon_last_vid = None; self.mon_sig = None
        top = ctk.CTkFrame(self.t_mon, fg_color="transparent"); top.pack(fill="x", pady=(0, 6))
        self.mon_stats = {}
        for key, label, col in [("online", "ออนไลน์", GREEN_OK), ("exam", "กำลังสอบ", CYAN), ("alert", "ทำผิดกฎอยู่ตอนนี้", RED_ALERT)]:
            c = ctk.CTkFrame(top, fg_color=BG_PANEL_2, corner_radius=10); c.pack(side="left", fill="x", expand=True, padx=4)
            ctk.CTkLabel(c, text=label, font=FONT_SMALL, text_color=TEXT_MUTED).pack(pady=(6, 0))
            v = ctk.CTkLabel(c, text="0", font=FONT_MONO, text_color=col); v.pack(pady=(0, 6)); self.mon_stats[key] = v
        body = ctk.CTkFrame(self.t_mon, fg_color="transparent"); body.pack(fill="both", expand=True)
        body.grid_columnconfigure(0, weight=3); body.grid_columnconfigure(1, weight=2, minsize=230); body.grid_rowconfigure(0, weight=1)
        self.mon_lst = ctk.CTkScrollableFrame(body, fg_color=BG_PANEL_2); self.mon_lst.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        feed = ctk.CTkFrame(body, fg_color=BG_PANEL_2, corner_radius=10); feed.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        ctk.CTkLabel(feed, text="🔔 แจ้งเตือนล่าสุด", font=FONT_SMALL_B, text_color=TEXT_MAIN).pack(anchor="w", padx=10, pady=(8, 4))
        self.feed_lst = ctk.CTkScrollableFrame(feed, fg_color="transparent"); self.feed_lst.pack(fill="both", expand=True, padx=4, pady=(0, 6))
        self.mon_empty = ctk.CTkLabel(self.mon_lst, text="ยังไม่มีนักเรียน", font=FONT_SUB, text_color=TEXT_MUTED)

    def _mon_row(self, s):
        r = ctk.CTkFrame(self.mon_lst, corner_radius=10, border_width=1)
        dot = ctk.CTkLabel(r, text="●", font=FONT_BODY); dot.pack(side="left", padx=(14, 6), pady=10)
        name = ctk.CTkLabel(r, text=s['username'], font=FONT_BOLD, text_color=TEXT_MAIN); name.pack(side="left", pady=10)
        status = ctk.CTkLabel(r, text="", font=FONT_SMALL_B, anchor="w"); status.pack(side="left", padx=14, fill="x", expand=True)
        kick = ctk.CTkButton(r, text="KICK", corner_radius=8, fg_color=RED_ALERT, hover_color=RED_DIM, width=64, height=28,
                             font=FONT_SMALL_B, command=lambda x=s['id']: db.kick_user(x))
        count = ctk.CTkLabel(r, text="", font=FONT_SMALL_B, text_color=AMBER_WARN)
        return {"frame": r, "dot": dot, "status": status, "kick": kick, "count": count}

    def update_monitor_loop(self):
        if not self.mon_active: return
        if not self.winfo_ismapped() and self.controller.role != 'teacher':
            self.mon_active = False; return
        try:
            act = db.get_all_students_status(self.controller.teacher_id)
            self._render_monitor(act); self._render_feed()
        except Exception as ex:
            print(f"⚠️ monitor error: {ex}")
        self.after(1500, self.update_monitor_loop)

    def _render_monitor(self, act):
        # เรียง: กำลังทำผิด → กำลังสอบ → ออนไลน์ → ออฟไลน์
        def rank(s): return (0 if s['in_exam'] and s['live_status'] else 1 if s['in_exam'] else 2 if s['online'] else 3, s['username'])
        act = sorted(act, key=rank)
        sig = [s['id'] for s in act]
        if sig != self.mon_sig:   # สร้างแถวใหม่เฉพาะตอนรายชื่อ/ลำดับเปลี่ยน ไม่งั้นอัปเดตในที่เดิม (ไม่กะพริบ)
            for w in self.mon_rows.values(): w['frame'].destroy()
            self.mon_rows = {}; self.mon_sig = sig
            self.mon_empty.pack_forget()
            if not act: self.mon_empty.pack(pady=30)
            for s in act:
                row = self._mon_row(s); row['frame'].pack(fill="x", pady=4, padx=2); self.mon_rows[s['id']] = row
        n_on = n_exam = n_alert = 0
        for s in act:
            row = self.mon_rows[s['id']]
            n_on += s['online']; n_exam += s['in_exam']
            if s['in_exam'] and s['live_status']:
                n_alert += 1
                col, bg, txt = RED_ALERT, "#3a0f16", f"⚠️ {s['live_status']}"
            elif s['in_exam']:
                col, bg, txt = CYAN, BG_PANEL, "🔵 กำลังสอบ — ปกติ"
            elif s['online']:
                col, bg, txt = GREEN_OK, BG_PANEL, "ONLINE"
            else:
                col, bg, txt = RED_ALERT, "#2a0f14", "OFFLINE"
            row['frame'].configure(fg_color=bg, border_color=col); row['dot'].configure(text_color=col)
            row['status'].configure(text=txt, text_color=col)
            if s['online']: row['kick'].pack(side="right", padx=12, pady=8)
            else: row['kick'].pack_forget()
            if s['in_exam'] and s['violation_count']:
                row['count'].configure(text=f"เตือน {s['violation_count']} ครั้ง"); row['count'].pack(side="right", padx=6)
            else: row['count'].pack_forget()
        self.mon_stats['online'].configure(text=str(n_on)); self.mon_stats['exam'].configure(text=str(n_exam))
        self.mon_stats['alert'].configure(text=str(n_alert))

    def _render_feed(self):
        items = db.get_recent_violations(self.controller.teacher_id, 0, 40)
        newest = items[0]['id'] if items else 0
        if newest == self.mon_last_vid: return
        is_new = self.mon_last_vid is not None and newest > self.mon_last_vid
        self.mon_last_vid = newest
        for w in self.feed_lst.winfo_children(): w.destroy()
        if not items:
            ctk.CTkLabel(self.feed_lst, text="ยังไม่มีการแจ้งเตือน", font=FONT_SMALL, text_color=TEXT_MUTED).pack(pady=20); return
        for i, v in enumerate(items):
            f = ctk.CTkFrame(self.feed_lst, fg_color="#3a0f16" if (is_new and i == 0) else BG_PANEL, corner_radius=6)
            f.pack(fill="x", pady=2)
            ctk.CTkLabel(f, text=f"{(v['time'] or '')[11:19]}  {v['name']}", font=FONT_SMALL_B, text_color=TEXT_MAIN,
                         anchor="w").pack(fill="x", padx=8, pady=(4, 0))
            ctk.CTkLabel(f, text=f"⚠️ {v['reason']}", font=FONT_SMALL, text_color=RED_ALERT, anchor="w").pack(fill="x", padx=8, pady=(0, 4))
        if is_new:
            try: self.bell()   # เสียงเตือนเมื่อมีเหตุการณ์ใหม่
            except Exception: pass

    def setup_q(self):
        self.choice_widgets = []; self.choice_pickers = []
        self.t_q.grid_columnconfigure(0, weight=3, minsize=360); self.t_q.grid_columnconfigure(1, weight=2, minsize=240)
        self.t_q.grid_rowconfigure(0, weight=1)
        f = ctk.CTkScrollableFrame(self.t_q, fg_color=BG_PANEL_2, corner_radius=12)
        f.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
        self.form_title = ctk.CTkLabel(f, text="📝 เพิ่ม / แก้ไขคำถาม", font=FONT_H3, text_color=WHITE_TXT)
        self.form_title.pack(anchor="w", pady=(4, 8))
        self.q_e = ctk.CTkEntry(f, placeholder_text="คำถาม...", height=38, corner_radius=8, fg_color=BG_INPUT,
                                border_color=BORDER_SOFT, text_color=TEXT_MAIN); self.q_e.pack(fill="x", pady=5)

        # --- สื่อประกอบโจทย์: รูปภาพ 1 ไฟล์ + เสียง 1 ไฟล์ (เช่น ข้อสอบ Listening) ---
        mbox = ctk.CTkFrame(f, fg_color=BG_PANEL, corner_radius=10, border_width=1, border_color=BORDER_SOFT)
        mbox.pack(fill="x", pady=5)
        ctk.CTkLabel(mbox, text="สื่อประกอบโจทย์ (ไม่บังคับ)", font=FONT_SMALL_B, text_color=TEXT_MUTED,
                     anchor="w").pack(fill="x", padx=12, pady=(8, 2))
        self.q_pickers = {
            "image": MediaPicker(mbox, kinds=("image",), label="🖼  แนบรูปภาพ"),
            "audio": MediaPicker(mbox, kinds=("audio",), label="🎧  แนบไฟล์เสียง"),
        }
        self.q_pickers["image"].pack(fill="x", padx=12, pady=3)
        self.q_pickers["audio"].pack(fill="x", padx=12, pady=(3, 10))

        self.ans_e = ctk.CTkEntry(f, placeholder_text="เฉลย (ใส่เลขข้อที่ถูก เช่น 1, 2, 3)",
                                  height=38, corner_radius=8, fg_color=BG_INPUT, border_color=BORDER_SOFT, text_color=TEXT_MAIN)
        self.ans_e.pack(fill="x", pady=5)
        ctk.CTkLabel(f, text="ช้อยส์", font=FONT_SMALL_B, text_color=TEXT_MUTED, anchor="w").pack(fill="x", pady=(6, 0))
        self.choice_f = ctk.CTkFrame(f, fg_color="transparent"); self.choice_f.pack(fill="x", pady=5)
        self.btn_add = ctk.CTkButton(f, text="＋ เพิ่มช้อยส์", width=110, height=30, corner_radius=8, command=self.add_choice,
                                     fg_color=CYAN_DIM, hover_color=CYAN, text_color=TEXT_MAIN, font=FONT_SMALL_B)
        btn_row = ctk.CTkFrame(f, fg_color="transparent"); btn_row.pack(fill="x", pady=(10, 4))
        ctk.CTkButton(btn_row, text="💾 บันทึก", command=self.save_q, corner_radius=8, height=36, font=FONT_BOLD,
                     fg_color=TEAL, hover_color=TEAL_DIM, text_color="#04121a").pack(side="left", expand=True, fill="x", padx=(0, 6))
        ctk.CTkButton(btn_row, text="ล้าง", command=self.clear_form, corner_radius=8, height=36, fg_color="transparent",
                     border_width=1, border_color=BORDER_SOFT, text_color=TEXT_MUTED).pack(side="left", expand=True, fill="x", padx=(6, 0))
        right = ctk.CTkFrame(self.t_q, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)
        rhead = ctk.CTkFrame(right, fg_color="transparent"); rhead.pack(fill="x", pady=(0, 4))
        ctk.CTkLabel(rhead, text="รายการคำถามในชุดนี้", font=FONT_SMALL_B, text_color=TEXT_MUTED,
                     anchor="w").pack(side="left", padx=2)
        ctk.CTkButton(rhead, text="📥 นำเข้าจากไฟล์", width=130, height=28, corner_radius=8, font=FONT_SMALL_B,
                      fg_color=TEAL, hover_color=TEAL_DIM, text_color="#04121a",
                      command=self.open_import).pack(side="right")
        self.q_lst = ctk.CTkScrollableFrame(right, fg_color=BG_PANEL_2, corner_radius=12)
        self.q_lst.pack(fill="both", expand=True)
        self.upd_form()

    def open_import(self):
        if not self.active_exam_id: messagebox.showwarning("เตือน", "เลือกหรือสร้างชุดข้อสอบก่อน"); return
        name = self.exam_option.get().rsplit(" (", 1)[0]
        ImportExamWindow(self, self.active_exam_id, name, on_saved=self.load_q)

    def _reset_choices(self):
        for w in self.choice_f.winfo_children(): w.destroy()
        self.choice_widgets = []; self.choice_pickers = []

    def upd_form(self):
        self._reset_choices()
        self.btn_add.pack(anchor="w", after=self.choice_f); self.add_choice(); self.add_choice()

    def add_choice(self, txt="", media_path=None):
        row = ctk.CTkFrame(self.choice_f, fg_color="transparent"); row.pack(fill="x", pady=3)
        e = ctk.CTkEntry(row, placeholder_text=f"ข้อ {len(self.choice_widgets)+1}", height=34,
                         corner_radius=8, fg_color=BG_INPUT, border_color=BORDER_SOFT, text_color=TEXT_MAIN)
        e.pack(side="left", fill="x", expand=True)
        p = MediaPicker(row, kinds=("image", "audio"), compact=True); p.pack(side="left", padx=(8, 0))
        if txt: e.insert(0, txt)
        if media_path: p.set(media_path)
        self.choice_widgets.append(e); self.choice_pickers.append(p)

    def save_q(self):
        if not self.active_exam_id: return
        qtype = "MCQ"; text = self.q_e.get().strip()
        choices = []   # [(ข้อความ, ไฟล์สื่อ)] ลำดับ = เลข slot ที่เก็บในฐานข้อมูล
        if qtype == "MCQ":
            for e, p in zip(self.choice_widgets, self.choice_pickers):
                t = e.get().strip(); fpath = p.get()
                if not t and fpath: t = f"ตัวเลือก {len(choices)+1}"   # ช้อยส์ที่เป็นรูป/เสียงล้วน
                if t: choices.append((t, fpath))
        desired = []
        for kind in ("image", "audio"):
            pth = self.q_pickers[kind].get()
            if pth: desired.append((0, kind, pth))
        for i, (_, fpath) in enumerate(choices, start=1):
            if fpath: desired.append((i, media.kind_of(fpath), fpath))
        if not text and not desired:
            messagebox.showwarning("เตือน", "ใส่คำถาม หรือแนบรูป/เสียงอย่างน้อยหนึ่งอย่าง"); return
        opts = "|".join(t for t, _ in choices)
        answer = self.ans_e.get().strip()
        if qtype == "MCQ":
            # เฉลยปรนัยต้องเป็น "เลขข้อ" ถ้าครูพิมพ์ข้อความของช้อยส์มา จะแปลงเป็นเลขข้อให้อัตโนมัติ
            if len(choices) < 2:
                messagebox.showwarning("เตือน", "ข้อสอบแบบช้อยส์ต้องมีตัวเลือกอย่างน้อย 2 ข้อ"); return
            idx = normalize_mcq_answer(answer, [t for t, _ in choices])
            if idx is None:
                messagebox.showwarning("เตือน", f"ใส่เฉลยเป็นเลขข้อ 1-{len(choices)} (ตอนนี้ใส่ว่า: '{answer or 'ว่าง'}')"); return
            answer = str(idx)
        if self.edit_q_id:
            qid = self.edit_q_id
            db.update_question(qid, qtype, text, opts, answer)
        else:
            qid = db.add_question(self.active_exam_id, qtype, text, opts, answer)
        if qid is not None:
            try: media.replace_all(qid, desired)
            except media.MediaError as ex:
                messagebox.showerror("แนบสื่อไม่สำเร็จ", f"บันทึกโจทย์แล้ว แต่แนบสื่อไม่ได้:\n{ex}")
        elif desired:
            messagebox.showwarning("เตือน", "บันทึกโจทย์แล้ว แต่ระบุรหัสข้อสอบใหม่ไม่ได้ จึงยังไม่ได้แนบสื่อ\nลองกด ✏️ แก้ไข แล้วแนบอีกครั้ง")
        self.load_q(); self.clear_form()

    def edit_q(self, q):
        self.clear_form(); self.edit_q_id = q['id']
        self.form_title.configure(text="✏️ แก้ไขคำถาม")
        self.q_e.insert(0, q['text'] or ""); self.ans_e.insert(0, str(q['correct'] or ""))
        self.upd_form()
        items = media.list_media(q['id'])
        for it in items:
            if it['slot'] == 0 and os.path.exists(it['path']): self.q_pickers[it['kind']].set(it['path'])
        if (q['options'] or "").strip():
            self._reset_choices()
            by_slot = {it['slot']: it['path'] for it in items if it['slot'] > 0 and os.path.exists(it['path'])}
            for i, op in enumerate([o for o in (q['options'] or "").split('|') if o.strip()], start=1):
                self.add_choice(op, by_slot.get(i))
            if not self.choice_widgets: self.add_choice(); self.add_choice()

    def load_q(self):
        for w in self.q_lst.winfo_children(): w.destroy()
        if not self.active_exam_id: return
        for q in db.get_questions_by_exam(self.active_exam_id):
            r = ctk.CTkFrame(self.q_lst, fg_color=BG_PANEL, corner_radius=8, border_width=1, border_color=BORDER_SOFT)
            r.pack(fill="x", pady=3, padx=2)
            legacy = q['type'] != 'MCQ'   # ข้อเขียนเก่าที่สร้างไว้ก่อนเลิกใช้ — นักเรียนจะไม่เห็นข้อนี้
            tag_col = RED_DIM if legacy else CYAN_DIM
            tag = ctk.CTkLabel(r, text="เลิกใช้" if legacy else "MCQ", font=FONT_SMALL_B, text_color=WHITE_TXT, fg_color=tag_col,
                                corner_radius=6, width=48, height=20)
            tag.pack(side="left", padx=(10, 8), pady=8)
            items = media.list_media(q['id'])
            n_img = sum(1 for it in items if it['kind'] == 'image'); n_aud = sum(1 for it in items if it['kind'] == 'audio')
            label = q['text'] or "(ไม่มีข้อความ)"
            ctk.CTkLabel(r, text=label, text_color=TEXT_MAIN, wraplength=260, justify="left").pack(
                side="left", padx=2, pady=8, fill="x", expand=True)
            if n_img: ctk.CTkLabel(r, text=f"🖼 {n_img}", font=FONT_SMALL_B, text_color=CYAN).pack(side="left", padx=4)
            if n_aud: ctk.CTkLabel(r, text=f"🎧 {n_aud}", font=FONT_SMALL_B, text_color=AMBER_WARN).pack(side="left", padx=4)
            ctk.CTkButton(r, text="ลบ", width=44, height=26, corner_radius=6, font=FONT_SMALL_B,
                         fg_color=RED_ALERT, hover_color=RED_DIM,
                         command=lambda x=q['id']: self.delete_q(x)).pack(side="right", padx=(4, 8), pady=8)
            ctk.CTkButton(r, text="✏️ แก้ไข", width=64, height=26, corner_radius=6, font=FONT_SMALL_B,
                         fg_color=BG_INPUT, hover_color=CYAN_DIM, border_width=1, border_color=BORDER_SOFT,
                         command=lambda qq=q: self.edit_q(qq)).pack(side="right", padx=4, pady=8)

    def delete_q(self, qid):
        media.delete_question_media(qid); db.delete_question(qid)
        if self.edit_q_id == qid: self.clear_form()
        self.load_q()

    def clear_form(self):
        self.edit_q_id = None; self.q_e.delete(0, 'end'); self.ans_e.delete(0, 'end')
        for p in self.q_pickers.values(): p.clear()
        self.form_title.configure(text="📝 เพิ่ม / แก้ไขคำถาม"); self.upd_form()

    def setup_s(self):
        self.s_lst = ctk.CTkScrollableFrame(self.t_s, fg_color=BG_PANEL_2, corner_radius=12)
        self.s_lst.pack(fill="both", expand=True, padx=4, pady=4)
    def load_s(self):
        for w in self.s_lst.winfo_children(): w.destroy()
        students = db.get_my_students(self.controller.teacher_id)
        if not students:
            ctk.CTkLabel(self.s_lst, text="ยังไม่มีนักเรียน", font=FONT_SUB, text_color=TEXT_MUTED).pack(pady=30)
        for s in students:
            r = ctk.CTkFrame(self.s_lst, fg_color=BG_PANEL, corner_radius=8, border_width=1, border_color=BORDER_SOFT)
            r.pack(fill="x", pady=3, padx=2)
            ctk.CTkLabel(r, text=f"👤 {s['username']}", font=FONT_BOLD, text_color=TEXT_MAIN).pack(side="left", padx=14, pady=8)
            if not s['approved']:
                ctk.CTkLabel(r, text="รออนุมัติ", font=FONT_SMALL_B, text_color=AMBER_WARN).pack(side="left")
                ctk.CTkButton(r, text="อนุมัติ", corner_radius=6, height=26, font=FONT_SMALL_B,
                             fg_color=GREEN_OK, hover_color="#15803d", width=64,
                             command=lambda x=s['id']: (db.approve_user(x), self.load_s())).pack(side="right", padx=8, pady=8)
            ctk.CTkButton(r, text="ลบ", corner_radius=6, height=26, font=FONT_SMALL_B, fg_color=RED_ALERT,
                         hover_color=RED_DIM, width=54,
                         command=lambda x=s['id']: (db.delete_user(x), self.load_s())).pack(side="right", padx=8, pady=8)

# ================= Main App =================
class ExamApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AI Proctored Exam System (Ultimate Edition)")
        self.geometry("1280x800")
        self.configure(fg_color=BG_APP)
        self.user_id = None; self.username = None; self.role = None; self.teacher_id = None
        self.container = ctk.CTkFrame(self, fg_color=BG_APP)
        self.container.pack(fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1); self.container.grid_columnconfigure(0, weight=1)
        self.frames = {}
        for F in (LoginPage, RegisterPage, StudentExamPage, TeacherDashboard):
            page_name = F.__name__
            frame = F(parent=self.container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        self.precheck_passed = False
        self.show_frame("LoginPage")
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(300, start_ai_loading)   # เริ่มโหลด AI เบื้องหลังทันทีหลังหน้าต่างขึ้น

    def on_close(self):
        """ปิดโปรแกรมให้เรียบร้อย: หยุดอัดวิดีโอ ปล่อยกล้อง และหยุดเธรด AI ก่อน (กันไฟล์วิดีโอเสีย/โปรแกรมค้างตอนปิด)"""
        page = self.frames.get("StudentExamPage")
        if page:
            for step in (page.force_stop_exam, page.stop_cam, lambda: page.detector.join(1.5)):
                try: step()
                except Exception:
                    import traceback; traceback.print_exc()
        self.destroy()
        # จบโปรเซสทันที กันไลบรารี AI (torch/mediapipe) ค้างหรือขึ้น "terminate called..." ตอนปิด
        os._exit(0)

    def show_frame(self, page_name):
        frame = self.frames[page_name]; frame.tkraise()
        if hasattr(frame, "on_show"): frame.on_show()
        if page_name != "StudentExamPage" and "StudentExamPage" in self.frames:
            self.frames["StudentExamPage"].force_stop_exam()
            self.frames["StudentExamPage"].stop_cam()

    def logout(self): 
        self.user_id = None; self.teacher_id = None; self.role = None; self.precheck_passed = False
        self.show_frame("LoginPage"); api.logout()

if __name__ == "__main__":
    ExamApp().mainloop()