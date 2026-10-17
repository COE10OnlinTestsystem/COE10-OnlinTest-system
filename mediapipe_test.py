"""
สคริปต์ทดสอบการโหลด MediaPipe แยกต่างหาก
วิธีรัน: python mediapipe_test.py
"""
import sys
print(f"Python version: {sys.version}\n")

try:
    import numpy
    print(f"✅ numpy version: {numpy.__version__}")
except Exception as e:
    print(f"❌ numpy import error: {e}")

try:
    import mediapipe as mp
    print(f"✅ mediapipe version: {mp.__version__}")
except Exception as e:
    print(f"❌ mediapipe import error: {e}")
    import traceback
    traceback.print_exc()
    input("\nกด Enter เพื่อปิด...")
    sys.exit(1)

if not hasattr(mp, "solutions"):
    print(f"\n❌ mediapipe {mp.__version__} ไม่มี mp.solutions แล้ว (ถูกตัดออกตั้งแต่ 0.10.30)")
    print("   แก้โดยรัน: .venv\\Scripts\\python.exe -m pip install mediapipe==0.10.21 \"numpy<2\" \"opencv-python<4.12\"")
    input("\nกด Enter เพื่อปิด...")
    sys.exit(1)

print("\nกำลังลองสร้าง FaceMesh และ FaceDetection...")
try:
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=2, refine_landmarks=True)
    print("✅ สร้าง FaceMesh สำเร็จ")

    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
    print("✅ สร้าง FaceDetection สำเร็จ")

    print("\n🎉 MediaPipe ทำงานได้ปกติทั้งหมด ปัญหาไม่ได้อยู่ที่ MediaPipe")
except Exception as e:
    print(f"\n❌ สร้าง FaceMesh/FaceDetection ไม่สำเร็จ: {e}")
    import traceback
    traceback.print_exc()

input("\nกด Enter เพื่อปิด...")