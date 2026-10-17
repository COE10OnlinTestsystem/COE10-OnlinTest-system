"""
สคริปต์ทดสอบกล้อง - รันแล้วดูว่ามี index ไหนเปิดกล้องได้บ้าง
วิธีรัน: .venv\\Scripts\\python.exe camera_test.py
"""
import cv2

print("กำลังสแกนหากล้องที่ใช้งานได้ (index 0-5)...\n")

found_any = False
for i in range(6):
    for backend_name, backend in [("DSHOW", cv2.CAP_DSHOW), ("DEFAULT", cv2.CAP_ANY)]:
        cap = cv2.VideoCapture(i, backend)
        opened = cap.isOpened()
        if opened:
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f"✅ index={i} backend={backend_name} -> เปิดได้ และอ่านภาพได้ ({w}x{h})")
                found_any = True
            else:
                print(f"⚠️  index={i} backend={backend_name} -> เปิดได้ แต่อ่านภาพไม่ได้ (ret={ret})")
        cap.release()

if not found_any:
    print("\n❌ ไม่พบกล้องที่ใช้งานได้เลยจาก index 0-5")
    print("แปลว่าเป็นปัญหาที่ระดับฮาร์ดแวร์/ไดรเวอร์/สิทธิ์การเข้าถึงของ Windows แน่นอน")
    print("ไม่เกี่ยวกับโค้ดของโปรแกรมสอบเลย")
else:
    print("\nพบกล้องที่ใช้งานได้ ถ้า index ไม่ใช่ 0 ให้แจ้งผมเพื่อแก้โค้ดให้ใช้ index ที่ถูกต้อง")

input("\nกด Enter เพื่อปิดหน้าต่างนี้...")
