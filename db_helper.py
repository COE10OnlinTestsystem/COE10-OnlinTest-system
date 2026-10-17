import sqlite3
import os
import sys
import datetime
import json
import hashlib
import hmac
import secrets

# --- ฟังก์ชันแก้ Path สำหรับไฟล์ .exe ---
def get_base_path():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

BASE_DIR = get_base_path()
DB_NAME = os.path.join(BASE_DIR, "exam_system.db")

def _connect():
    # timeout: อาจารย์กับนักเรียนหลายคนเขียนไฟล์เดียวกันพร้อมกัน (heartbeat) ให้รอแทนที่จะ error "database is locked"
    return sqlite3.connect(DB_NAME, timeout=10)

def init_db():
    conn = _connect()
    c = conn.cursor()
    
    # 1. ตารางผู้ใช้งาน
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE,
                    password TEXT,
                    role TEXT,
                    classroom_code TEXT UNIQUE, 
                    linked_teacher_id INTEGER,
                    is_approved INTEGER DEFAULT 0,
                    last_active DATETIME,
                    is_kicked INTEGER DEFAULT 0)''')

    # 2. ตารางชุดข้อสอบ
    c.execute('''CREATE TABLE IF NOT EXISTS exams (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    owner_id INTEGER,
                    name TEXT,
                    exam_code TEXT UNIQUE,
                    is_active INTEGER DEFAULT 0,
                    duration_minutes INTEGER DEFAULT 0)''')

    # Migration: เพิ่มคอลัมน์ duration_minutes ให้ฐานข้อมูลเก่าที่สร้างไว้ก่อนหน้านี้
    try:
        c.execute("ALTER TABLE exams ADD COLUMN duration_minutes INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass  # คอลัมน์มีอยู่แล้ว ไม่ต้องทำอะไร

    # 3. ตารางโจทย์
    c.execute('''CREATE TABLE IF NOT EXISTS questions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    exam_id INTEGER,
                    q_type TEXT, 
                    question_text TEXT,
                    options TEXT, 
                    correct_answer TEXT,
                    FOREIGN KEY(exam_id) REFERENCES exams(id) ON DELETE CASCADE)''') 

    # 4. ตารางผลคะแนน
    c.execute('''CREATE TABLE IF NOT EXISTS results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    exam_id INTEGER,
                    student_name TEXT,
                    score INTEGER,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

    # 5. คำตอบรายข้อของนักเรียนแต่ละคน (ใช้ทำ Item Analysis)
    c.execute('''CREATE TABLE IF NOT EXISTS answers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    result_id INTEGER,
                    exam_id INTEGER,
                    question_id INTEGER,
                    student_name TEXT,
                    answer TEXT,
                    is_correct INTEGER)''')

    # 6. ประวัติพฤติกรรมเสี่ยงระหว่างสอบ (ส่งมาแบบเรียลไทม์)
    c.execute('''CREATE TABLE IF NOT EXISTS violations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    exam_id INTEGER,
                    user_id INTEGER,
                    student_name TEXT,
                    reason TEXT,
                    created_at TEXT)''')

    # Migration: คอลัมน์ใหม่สำหรับฐานข้อมูลเก่า
    for table, col, typ in [("exams", "sensitivity", "TEXT"),
                            ("users", "api_token", "TEXT"), ("users", "live_status", "TEXT"), ("users", "live_exam_id", "INTEGER"),
                            ("results", "user_id", "INTEGER"), ("results", "max_score", "REAL"),
                            ("results", "exam_seconds", "REAL"), ("results", "flagged_seconds", "REAL"),
                            ("results", "risk_pct", "REAL"), ("results", "violation_detail", "TEXT")]:
        try:
            c.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")
        except sqlite3.OperationalError:
            pass

    conn.commit()
    conn.close()

def _now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

# --- รหัสผ่าน: เก็บแบบแฮช (PBKDF2) ไม่เก็บตัวจริง ---
_PBKDF2_ITER = 200_000

def hash_password(pwd):
    salt = secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac("sha256", pwd.encode("utf-8"), bytes.fromhex(salt), _PBKDF2_ITER).hex()
    return f"pbkdf2${_PBKDF2_ITER}${salt}${h}"

def verify_password(pwd, stored):
    if not stored: return False
    if stored.startswith("pbkdf2$"):
        try:
            _, it, salt, h = stored.split("$")
            calc = hashlib.pbkdf2_hmac("sha256", pwd.encode("utf-8"), bytes.fromhex(salt), int(it)).hex()
            return hmac.compare_digest(calc, h)
        except ValueError:
            return False
    return hmac.compare_digest(stored, pwd)   # บัญชีเก่าที่ยังเก็บรหัสตรงๆ

# --- ระบบล็อกอิน ---
def login_user(user, pwd):
    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT id, role, is_approved, linked_teacher_id, password FROM users WHERE username=?", (user,))
    row = c.fetchone()
    if not row or not verify_password(pwd, row[4]):
        conn.close(); return None
    if not row[4].startswith("pbkdf2$"):   # อัปเกรดบัญชีเก่าให้เป็นแบบแฮชอัตโนมัติ
        c.execute("UPDATE users SET password=? WHERE id=?", (hash_password(pwd), row[0])); conn.commit()
    conn.close()
    result = row[:4]
    if result[1] == 'student':
        reset_kick_status(result[0])
    return result

def change_password(user_id, old_pwd, new_pwd):
    """เปลี่ยนรหัสผ่าน ต้องใส่รหัสเดิมถูกก่อน คืน True/False"""
    conn = _connect(); c = conn.cursor()
    c.execute("SELECT password FROM users WHERE id=?", (user_id,))
    r = c.fetchone()
    if not r or not verify_password(old_pwd, r[0]):
        conn.close(); return False
    c.execute("UPDATE users SET password=? WHERE id=?", (hash_password(new_pwd), user_id))
    conn.commit(); conn.close()
    return True

def issue_token(user_id):
    """ออก token ให้เครื่องนักเรียนใช้เรียก API (แทนการส่งรหัสผ่านทุกครั้ง)"""
    tok = secrets.token_urlsafe(24)
    conn = _connect()
    conn.cursor().execute("UPDATE users SET api_token=? WHERE id=?", (tok, user_id))
    conn.commit(); conn.close()
    return tok

def user_by_token(token):
    if not token: return None
    conn = _connect(); c = conn.cursor()
    c.execute("SELECT id, username, role, linked_teacher_id, is_approved FROM users WHERE api_token=?", (token,))
    r = c.fetchone(); conn.close()
    if not r: return None
    return {"id": r[0], "username": r[1], "role": r[2], "teacher_id": r[3], "approved": r[4]}

def register_teacher(username, password, class_code):
    try:
        conn = _connect()
        conn.cursor().execute("INSERT INTO users (username, password, role, classroom_code, is_approved) VALUES (?, ?, 'teacher', ?, 1)", (username, hash_password(password), class_code))
        conn.commit()
        conn.close()
        return "success"
    except sqlite3.IntegrityError:
        return "duplicate"

def register_student(username, password, target_class_code):
    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE classroom_code=?", (target_class_code,))
    t = c.fetchone()
    if not t:
        conn.close()
        return "invalid_code"
    try:
        conn.cursor().execute("INSERT INTO users (username, password, role, linked_teacher_id, is_approved) VALUES (?, ?, 'student', ?, 0)", (username, hash_password(password), t[0]))
        conn.commit()
        conn.close()
        return "success"
    except sqlite3.IntegrityError:
        conn.close()
        return "duplicate_user"

# --- ระบบจัดการข้อสอบ ---
def create_exam_set(teacher_id, name, code, duration_minutes=0, sensitivity="medium"):
    try:
        conn = _connect()
        conn.cursor().execute("INSERT INTO exams (owner_id, name, exam_code, is_active, duration_minutes, sensitivity) VALUES (?, ?, ?, 0, ?, ?)", (teacher_id, name, code, duration_minutes, sensitivity))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False

def update_exam_duration(exam_id, duration_minutes):
    conn = _connect()
    conn.cursor().execute("UPDATE exams SET duration_minutes=? WHERE id=?", (duration_minutes, exam_id))
    conn.commit()
    conn.close()

def update_exam_sensitivity(exam_id, sensitivity):
    conn = _connect()
    conn.cursor().execute("UPDATE exams SET sensitivity=? WHERE id=?", (sensitivity, exam_id))
    conn.commit()
    conn.close()

def get_teacher_exams(teacher_id):
    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT id, name, exam_code, is_active, duration_minutes, sensitivity FROM exams WHERE owner_id=?", (teacher_id,))
    data = [{"id": r[0], "name": r[1], "code": r[2], "active": r[3], "duration": r[4] or 0, "sensitivity": r[5] or "medium"} for r in c.fetchall()]
    conn.close()
    return data

def toggle_exam_status(exam_id, current_status):
    new_status = 0 if current_status == 1 else 1
    conn = _connect()
    conn.cursor().execute("UPDATE exams SET is_active=? WHERE id=?", (new_status, exam_id))
    conn.commit()
    conn.close()
    return new_status

def delete_exam_set(exam_id):
    conn = _connect()
    conn.cursor().execute("DELETE FROM answers WHERE exam_id=?", (exam_id,))
    conn.cursor().execute("DELETE FROM violations WHERE exam_id=?", (exam_id,))
    conn.cursor().execute("DELETE FROM questions WHERE exam_id=?", (exam_id,))
    conn.cursor().execute("DELETE FROM exams WHERE id=?", (exam_id,))
    conn.commit()
    conn.close()

def get_exam_full(exam_id):
    conn = _connect(); c = conn.cursor()
    c.execute("SELECT id, name, is_active, owner_id, duration_minutes, sensitivity FROM exams WHERE id=?", (exam_id,))
    r = c.fetchone(); conn.close()
    return {"id": r[0], "name": r[1], "active": r[2], "owner_id": r[3], "duration": r[4] or 0, "sensitivity": r[5] or "medium"} if r else None

def get_exam_info(exam_id):
    conn = _connect(); c = conn.cursor()
    c.execute("SELECT name, exam_code FROM exams WHERE id=?", (exam_id,))
    r = c.fetchone(); conn.close()
    return {"name": r[0], "code": r[1]} if r else None

def get_exam_by_code(code):
    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT id, name, is_active, owner_id, duration_minutes, sensitivity FROM exams WHERE exam_code=?", (code,))
    res = c.fetchone()
    conn.close()
    if res: return {"id": res[0], "name": res[1], "active": res[2], "owner_id": res[3], "duration": res[4] or 0, "sensitivity": res[5] or "medium"}
    return None

# --- ระบบโจทย์ ---
def get_questions_by_exam(exam_id):
    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT * FROM questions WHERE exam_id=?", (exam_id,))
    data = [{"id": r[0], "type": r[2], "text": r[3], "options": r[4], "correct": r[5]} for r in c.fetchall()]
    conn.close()
    return data

def add_question(exam_id, q_type, text, options, correct):
    """เพิ่มโจทย์ และคืน id ของโจทย์ใหม่ (ใช้ผูกรูป/เสียงประกอบ)"""
    conn = _connect()
    cur = conn.cursor()
    cur.execute("INSERT INTO questions (exam_id, q_type, question_text, options, correct_answer) VALUES (?, ?, ?, ?, ?)", 
                (exam_id, q_type, text, options, correct))
    new_id = cur.lastrowid
    conn.commit(); conn.close()
    return new_id

def delete_question(qid):
    conn = _connect()
    conn.cursor().execute("DELETE FROM questions WHERE id=?", (qid,))
    conn.commit()
    conn.close()

def update_question(qid, q_type, text, options, correct):
    conn = _connect()
    conn.cursor().execute("UPDATE questions SET q_type=?, question_text=?, options=?, correct_answer=? WHERE id=?", 
                          (q_type, text, options, correct, qid))
    conn.commit(); conn.close()

# --- ระบบคะแนน ---
def save_score(exam_id, name, score, max_score=None, user_id=None, exam_seconds=None,
               flagged_seconds=None, risk_pct=None, violation_detail=None, answers=None):
    """
    บันทึกผลสอบ + คำตอบรายข้อ คืน result_id
    answers = [(question_id, answer_text, is_correct 1/0/None), ...]  (None = อัตนัย รอครูตรวจ)
    violation_detail = dict {เหตุผล: วินาที}
    """
    conn = _connect(); cur = conn.cursor()
    cur.execute("""INSERT INTO results (exam_id, student_name, score, timestamp, user_id, max_score, exam_seconds,
                   flagged_seconds, risk_pct, violation_detail) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (exam_id, name, score, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), user_id, max_score,
                 exam_seconds, flagged_seconds, risk_pct,
                 json.dumps(violation_detail, ensure_ascii=False) if violation_detail is not None else None))
    rid = cur.lastrowid
    if answers:
        cur.executemany("INSERT INTO answers (result_id, exam_id, question_id, student_name, answer, is_correct) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        [(rid, exam_id, qid, name, ans, ok) for qid, ans, ok in answers])
    conn.commit(); conn.close()
    return rid

def get_exam_results(exam_id):
    conn = _connect(); c = conn.cursor()
    c.execute("""SELECT id, student_name, score, timestamp, max_score, exam_seconds, flagged_seconds, risk_pct,
                 violation_detail, user_id FROM results WHERE exam_id=? ORDER BY timestamp DESC""", (exam_id,))
    rows = c.fetchall()
    c.execute("SELECT student_name, COUNT(*) FROM violations WHERE exam_id=? GROUP BY student_name", (exam_id,))
    vcount = dict(c.fetchall())
    conn.close()
    data = []
    for r in rows:
        try: detail = json.loads(r[8]) if r[8] else {}
        except ValueError: detail = {}
        data.append({"id": r[0], "name": r[1], "score": r[2], "time": r[3], "max": r[4],
                     "exam_seconds": r[5], "flagged_seconds": r[6], "risk": r[7], "detail": detail,
                     "user_id": r[9], "violations": vcount.get(r[1], 0)})
    return data

def get_exam_answers(exam_id):
    conn = _connect(); c = conn.cursor()
    c.execute("SELECT result_id, question_id, student_name, answer, is_correct FROM answers WHERE exam_id=?", (exam_id,))
    data = [{"result_id": r[0], "question_id": r[1], "name": r[2], "answer": r[3], "correct": r[4]} for r in c.fetchall()]
    conn.close()
    return data

# --- พฤติกรรมเสี่ยง (เรียลไทม์) ---
def log_violation(exam_id, user_id, name, reason):
    conn = _connect()
    conn.cursor().execute("INSERT INTO violations (exam_id, user_id, student_name, reason, created_at) VALUES (?, ?, ?, ?, ?)",
                          (exam_id, user_id, name, reason, _now_str()))
    conn.commit(); conn.close()

def get_recent_violations(teacher_id, after_id=0, limit=40):
    """แจ้งเตือนล่าสุดของนักเรียนในห้องของอาจารย์คนนี้ (ใหม่สุดก่อน)"""
    conn = _connect(); c = conn.cursor()
    c.execute("""SELECT v.id, v.exam_id, v.user_id, v.student_name, v.reason, v.created_at FROM violations v
                 JOIN exams e ON e.id = v.exam_id WHERE e.owner_id=? AND v.id>? ORDER BY v.id DESC LIMIT ?""",
              (teacher_id, after_id, limit))
    data = [{"id": r[0], "exam_id": r[1], "user_id": r[2], "name": r[3], "reason": r[4], "time": r[5]} for r in c.fetchall()]
    conn.close()
    return data

def get_exam_violations(exam_id):
    conn = _connect(); c = conn.cursor()
    c.execute("SELECT student_name, reason, created_at FROM violations WHERE exam_id=? ORDER BY id", (exam_id,))
    data = [{"name": r[0], "reason": r[1], "time": r[2]} for r in c.fetchall()]
    conn.close()
    return data

def clear_live_status(user_id):
    conn = _connect()
    conn.cursor().execute("UPDATE users SET live_status=NULL, live_exam_id=NULL WHERE id=?", (user_id,))
    conn.commit(); conn.close()

# --- ระบบ Monitor ---
def update_heartbeat(user_id, live_status=None, exam_id=None):
    """live_status: "" = สอบปกติ, "หันซ้าย, ..." = กำลังทำผิดกฎ, None = ไม่ได้อยู่ในการสอบ"""
    conn = _connect()
    conn.cursor().execute("UPDATE users SET live_status=?, live_exam_id=? WHERE id=?", (live_status, exam_id, user_id))
    # เก็บเป็นข้อความรูปแบบเดียวกันเสมอ (เดิมถ้าไมโครวินาทีเป็น 0 จะไม่มี .%f ทำให้อ่านเวลาไม่ได้ นักเรียนโชว์ออฟไลน์)
    conn.cursor().execute("UPDATE users SET last_active=? WHERE id=?",
                          (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"), user_id))
    conn.commit()
    conn.close()

def check_user_status(user_id):
    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT is_kicked FROM users WHERE id=?", (user_id,))
    res = c.fetchone()
    conn.close()
    return res[0] if res else 0

def kick_user(user_id):
    conn = _connect()
    conn.cursor().execute("UPDATE users SET is_kicked=1 WHERE id=?", (user_id,))
    conn.commit()
    conn.close()

def reset_kick_status(user_id):
    conn = _connect()
    conn.cursor().execute("UPDATE users SET is_kicked=0, last_active=NULL, live_status=NULL, live_exam_id=NULL WHERE id=?", (user_id,))
    conn.commit()
    conn.close()

def get_all_students_status(teacher_id):
    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT id, username, last_active, live_status, live_exam_id FROM users WHERE linked_teacher_id=? AND is_approved=1",
              (teacher_id,))
    rows = c.fetchall()
    c.execute("""SELECT user_id, exam_id, COUNT(*), MAX(created_at) FROM violations
                 WHERE user_id IN (SELECT id FROM users WHERE linked_teacher_id=?) GROUP BY user_id, exam_id""", (teacher_id,))
    vstat = {(r[0], r[1]): (r[2], r[3]) for r in c.fetchall()}
    conn.close()
    student_list = []
    now = datetime.datetime.now()
    for r in rows:
        uid, name, last_active, live_status, live_exam = r
        is_online = False
        if last_active:
            try:
                last_time = datetime.datetime.fromisoformat(str(last_active))
                if (now - last_time).total_seconds() < 15: is_online = True
            except ValueError: pass
        cnt, last_v = vstat.get((uid, live_exam), (0, None)) if live_exam else (0, None)
        student_list.append({"id": uid, "username": name, "online": is_online,
                             "in_exam": is_online and live_status is not None and live_exam is not None,
                             "live_status": live_status or "", "exam_id": live_exam,
                             "violation_count": cnt, "last_violation": last_v})
    return student_list

# --- ระบบนักเรียน ---
def get_my_students(teacher_id):
    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT id, username, is_approved FROM users WHERE linked_teacher_id=?", (teacher_id,))
    data = [{"id": r[0], "username": r[1], "approved": r[2]} for r in c.fetchall()]
    conn.close()
    return data

def approve_user(uid):
    conn = _connect()
    conn.cursor().execute("UPDATE users SET is_approved=1 WHERE id=?", (uid,))
    conn.commit()
    conn.close()

def delete_user(uid):
    conn = _connect()
    conn.cursor().execute("DELETE FROM users WHERE id=?", (uid,))
    conn.commit()
    conn.close()

init_db()