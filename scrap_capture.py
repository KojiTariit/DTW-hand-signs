import cv2
from mediapipe.python.solutions import holistic as mp_holistic
from mediapipe.python.solutions import drawing_utils as mp_drawing
import json
import time
import socket

def main():
    # Setup UDP Socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_address = ('127.0.0.1', 5005)

    # --- 1. SETUP MODEL ---
    # Upgraded to Complexity 1 for "Pure Gold" data
    holistic = mp_holistic.Holistic(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    # Attempting to boost FPS. 
    cap.set(cv2.CAP_PROP_FPS, 60)
    
    recording = False
    
    print("--- LIVE SKELETAL CAPTURE (Gen-2.5) ---")
    print("Capturing 10-point Spatial Star + Hands")
    print("Press 'R' to toggle signing.")
    print("Press 'Q' to quit.")

    while cap.isOpened():
        success, image = cap.read()
        if not success: break

        # --- 2. PERFORMANCE OPTIMIZATION ---
        # 320x240 is mandatory for Complexity 1 on laptops.
        image = cv2.resize(image, (320, 240))
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # RUN SKELETAL AI
        results = holistic.process(image_rgb)

        status_text = "Status: WAITING... (Press 'R')"
        if recording:
            status_text = "Status: SIGNING! (Press 'R' to stop)"

        # --- 3. DRAWING LOGIC ---
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Draw the 12 Spatial Anchors (Face 8 + Ears 2 + Shoulders 2)
        h, w, c = image.shape
        anchor_dots = []
        if results.face_landmarks:
            f_lms = results.face_landmarks.landmark
            for idx in [10, 152, 4, 234, 454, 13, 133, 362]:
                anchor_dots.append((int(f_lms[idx].x * w), int(f_lms[idx].y * h)))
        if results.pose_landmarks:
            p_lms = results.pose_landmarks.landmark
            for idx in [7, 8, 11, 12]:
                anchor_dots.append((int(p_lms[idx].x * w), int(p_lms[idx].y * h)))
        for dot in anchor_dots:
            cv2.circle(image, dot, 3, (0, 255, 255), -1)

        # --- 4. UDP PAYLOAD LOGIC ---
        if recording:
            payload = {
                "type": "FRAME",
                "hands": [],
                "face": None,
                "ears": None,
                "timestamp": time.time()
            }
            
            # A. Hands
            def pack_hand(lms):
                wrist = lms.landmark[0]
                frame_lms = []
                for lm in lms.landmark:
                    frame_lms.append({"x": lm.x - wrist.x, "y": lm.y - wrist.y, "z": lm.z - wrist.z})
                return {"landmarks": frame_lms, "wrist_pos": {"x": wrist.x, "y": wrist.y, "z": wrist.z}}

            if results.left_hand_landmarks:
                payload["hands"].append(pack_hand(results.left_hand_landmarks))
            if results.right_hand_landmarks:
                payload["hands"].append(pack_hand(results.right_hand_landmarks))

            # B. Face (8 Anchors)
            if results.face_landmarks:
                f = results.face_landmarks.landmark
                payload["face"] = {
                    "forehead": {"x": f[10].x, "y": f[10].y, "z": f[10].z},
                    "chin": {"x": f[152].x, "y": f[152].y, "z": f[152].z},
                    "nose": {"x": f[4].x, "y": f[4].y, "z": f[4].z},
                    "l_cheek": {"x": f[234].x, "y": f[234].y, "z": f[234].z},
                    "r_cheek": {"x": f[454].x, "y": f[454].y, "z": f[454].z},
                    "mouth": {"x": f[13].x, "y": f[13].y, "z": f[13].z},
                    "l_eye": {"x": f[133].x, "y": f[133].y, "z": f[133].z},
                    "r_eye": {"x": f[362].x, "y": f[362].y, "z": f[362].z}
                }

            # C. Ears & Shoulders (Pose Anchors)
            if results.pose_landmarks:
                p = results.pose_landmarks.landmark
                payload["pose_anchors"] = {
                    "l_ear": {"x": p[7].x, "y": p[7].y, "z": p[7].z},
                    "r_ear": {"x": p[8].x, "y": p[8].y, "z": p[8].z},
                    "l_shoulder": {"x": p[11].x, "y": p[11].y, "z": p[11].z},
                    "r_shoulder": {"x": p[12].x, "y": p[12].y, "z": p[12].z}
                }

            # Stream even if zero hands (to keep temporal alignment)
            sock.sendto(json.dumps(payload).encode('utf-8'), server_address)

        cv2.putText(image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if recording else (0, 255, 0), 2)
        cv2.imshow('Skeletal Capture Gen-2.5', image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            recording = not recording
            if not recording:
                print("Sign finished! Asking C++ to classify...")
                sock.sendto(json.dumps({"type": "END_OF_SIGN"}).encode('utf-8'), server_address)
            else:
                print("Started capturing sign...")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    holistic.close()

if __name__ == "__main__":
    main()
