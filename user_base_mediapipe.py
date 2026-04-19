import cv2
from mediapipe.python.solutions import hands as mp_hand
from mediapipe.python.solutions import drawing_utils as mp_drawing

# ตั้งตัวแปรสองตัว อันแรกไว้สำหรับตรวจรับค่ามือ อีกอันไว้กำหนดเส้น
# We use explicit path to bypass potential 'AttributeError' in some Windows builds
hand = mp_hand.Hands(
    static_image_mode=False,         # ตั้งค่าเป็น False เพราะเราทำ Real-time (ถ้าเป็นภาพนิ่งให้ตั้งเป็น True)
    max_num_hands=2,                # ตรวจจับมือได้สูงสุด 2 มือในคราวเดียว
    min_detection_confidence=0.5,    # ค่าความมั่นใจขั้นต่ำในการตรวจพบมือครั้งแรก (50%)
    min_tracking_confidence=0.5     # ค่าความมั่นใจขั้นต่ำในการติดตามมือในเฟรมถัดไป (50%)
)

# เปิดใช้งานกล้อง
cap = cv2.VideoCapture(0)

# วนลูปเพื่ออ่านภาพจากกล้องทีละเฟรม
while cap.isOpened():
    # ตั้งตัวแปรที่สามารถรับค่ารูปแบบ Tuple จากกล้องได้
    success, image = cap.read()
    # ถ้าตรวจจับไม่พบจะขึ้นข้อความ nope และจะถูกสั่งยกเลิกโปรแกรมทันที
    if not success:
        print("nope")
        break

    # กลับด้านภาพให้เหมือนกระจก
    image = cv2.flip(image, 1)
    # เปลี่ยนสี
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # ส่งภาพเข้าไอเอไอ เพื่อรับค่าตำแหน่งของมือ
    res = hand.process(image_rgb)

    # ตรวจสอบว่าในภาพมีการตรวจพบมือหรือไม่
    if res.multi_hand_landmarks:
        # วนลูปวาดจุดและเส้นเชื่อมข้อต่อนิ้วลงบนภาพสำหรับทุกๆ มือที่เจอ
        for hand_landmarks in res.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hand.HAND_CONNECTIONS)

    # แสดงผลลัพธ์ภาพที่วาดเส้นเสร็จแล้วผ่านหน้าต่างโปรแกรม      
    cv2.imshow('Gundam is the best',image)

    # รอรับคำสั่งถ้ากด 'q' ให้จบโปรแกรม
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดกล้องและปิดหน้าต่างทั้งหมดเมื่อโปรแกรมจบ
cap.release()
cv2.destroyAllWindows()
