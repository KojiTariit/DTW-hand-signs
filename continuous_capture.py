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
    # Upgraded to Complexity 2 to perfectly match the Golden Templates!
    holistic = mp_holistic.Holistic(
        model_complexity=1,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    # Attempting to boost FPS. 
    cap.set(cv2.CAP_PROP_FPS, 60)
    
    recording = False
    current_sign_buffer = []
    prev_right_wrist, prev_left_wrist = None, None
    silence_start_time = None
    last_sign_time = 0
    movement_threshold = 0.04 
    freeze_duration = 0.7 
    sign_break = 0.4
    
    print("--- LIVE SKELETAL CAPTURE (Gen-2.5) ---")
    print("Capturing 10-point Spatial Star + Hands")
    print("AUTO-SEGMENTATION ENABLED: Just sign, no need to press 'R'.")
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

        # --- AUTO-SEGMENTER LOGIC (Freeze/Velocity Based) ---
        is_moving = False
        hand_in_frame = False
        max_dist = 0.0
        
        # Track Right Hand
        if results.right_hand_landmarks:
            hand_in_frame = True
            wrist = results.right_hand_landmarks.landmark[0]
            if prev_right_wrist is not None:
                dx = wrist.x - prev_right_wrist[0]
                dy = wrist.y - prev_right_wrist[1]
                max_dist = max(max_dist, (dx*dx + dy*dy)**0.5)
            prev_right_wrist = (wrist.x, wrist.y)
        else:
            prev_right_wrist = None

        # Track Left Hand
        if results.left_hand_landmarks:
            hand_in_frame = True
            wrist = results.left_hand_landmarks.landmark[0]
            if prev_left_wrist is not None:
                dx = wrist.x - prev_left_wrist[0]
                dy = wrist.y - prev_left_wrist[1]
                max_dist = max(max_dist, (dx*dx + dy*dy)**0.5)
            prev_left_wrist = (wrist.x, wrist.y)
        else:
            prev_left_wrist = None

        if max_dist > movement_threshold:
            is_moving = True
            
        if is_moving:
            if not recording and (time.time() - last_sign_time > sign_break):
                recording = True
                print("Movement detected! Started capturing...")
            silence_start_time = None
            current_sign_buffer.append(1)
        elif hand_in_frame:
            # Hand is present but NOT moving significantly (The "Freeze")
            if recording:
                if silence_start_time is None:
                    silence_start_time = time.time()
                
                current_sign_buffer.append(1)
                
                elapsed_silence = time.time() - silence_start_time
                if elapsed_silence > freeze_duration:
                    recording = False
                    if len(current_sign_buffer) > 10:
                        print(f"Freeze detected! Sign length: {len(current_sign_buffer)} frames. Classifying...")
                        sock.sendto(json.dumps({"type": "END_OF_SIGN"}).encode('utf-8'), server_address)
                        last_sign_time = time.time()
                    else:
                        print(f"Ignored short movement ({len(current_sign_buffer)} frames).")
                    current_sign_buffer = []
                    silence_start_time = None
        else:
            # No hands in frame at all
            if recording:
                recording = False
                if len(current_sign_buffer) > 10:
                    print(f"Hand left scene! Sign length: {len(current_sign_buffer)} frames. Classifying...")
                    sock.sendto(json.dumps({"type": "END_OF_SIGN"}).encode('utf-8'), server_address)
                    last_sign_time = time.time()
                else:
                    print(f"Ignored short movement ({len(current_sign_buffer)} frames).")
                current_sign_buffer = []
            silence_start_time = None
            prev_right_wrist = None
            prev_left_wrist = None

        status_text = "Status: WAITING..."
        if recording:
            elapsed = time.time() - (silence_start_time if silence_start_time else time.time())
            status_text = f"Status: SIGNING! (Silence: {elapsed:.1f}s)"
        elif time.time() - last_sign_time < sign_break:
            status_text = "Status: BREAK..."

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
                "pose_anchors": None,
                "timestamp": time.time()
            }

            # 4. Body-Centric Origin (Nose Anchor)
            nose_lm = None
            if results.face_landmarks:
                nose_lm = results.face_landmarks.landmark[4]

            # A. Hands (Relative to Nose)
            def pack_hand(lms, nose, label):
                wrist = lms.landmark[0]
                frame_lms = []
                for lm in lms.landmark:
                    # Internal landmarks are relative to wrist
                    frame_lms.append({"x": lm.x - wrist.x, "y": lm.y - wrist.y, "z": lm.z - wrist.z})
                
                # Wrist position is relative to nose
                return {
                    "label": label,
                    "landmarks": frame_lms, 
                    "wrist_pos": {
                        "x": wrist.x - nose.x if nose else wrist.x, 
                        "y": wrist.y - nose.y if nose else wrist.y, 
                        "z": wrist.z - nose.z if nose else wrist.z
                    }
                }

            # A. Always send hands — use nose_lm as origin if available, else raw position
            if results.left_hand_landmarks:
                payload["hands"].append(pack_hand(results.left_hand_landmarks, nose_lm, "Left"))
            if results.right_hand_landmarks:
                payload["hands"].append(pack_hand(results.right_hand_landmarks, nose_lm, "Right"))

            # B. Face (8 Anchors - Relative to Nose) — only if face is detected
            if nose_lm and results.face_landmarks:
                f = results.face_landmarks.landmark
                n = nose_lm
                payload["face"] = {
                    "forehead": {"x": f[10].x - n.x, "y": f[10].y - n.y, "z": f[10].z - n.z},
                    "chin": {"x": f[152].x - n.x, "y": f[152].y - n.y, "z": f[152].z - n.z},
                    "nose": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "l_cheek": {"x": f[234].x - n.x, "y": f[234].y - n.y, "z": f[234].z - n.z},
                    "r_cheek": {"x": f[454].x - n.x, "y": f[454].y - n.y, "z": f[454].z - n.z},
                    "mouth": {"x": f[13].x - n.x, "y": f[13].y - n.y, "z": f[13].z - n.z},
                    "l_eye": {"x": f[133].x - n.x, "y": f[133].y - n.y, "z": f[133].z - n.z},
                    "r_eye": {"x": f[362].x - n.x, "y": f[362].y - n.y, "z": f[362].z - n.z}
                }

                # C. Ears & Shoulders (Pose Anchors - Relative to Nose)
                if results.pose_landmarks:
                    p = results.pose_landmarks.landmark
                    payload["pose_anchors"] = {
                        "l_ear": {"x": p[7].x - n.x, "y": p[7].y - n.y, "z": p[7].z - n.z},
                        "r_ear": {"x": p[8].x - n.x, "y": p[8].y - n.y, "z": p[8].z - n.z},
                        "l_shoulder": {"x": p[11].x - n.x, "y": p[11].y - n.y, "z": p[11].z - n.z},
                        "r_shoulder": {"x": p[12].x - n.x, "y": p[12].y - n.y, "z": p[12].z - n.z}
                    }

            # Stream even if zero hands (to keep temporal alignment)
            sock.sendto(json.dumps(payload).encode('utf-8'), server_address)

        cv2.putText(image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if recording else (0, 255, 0), 2)
        cv2.imshow('Skeletal Capture Gen-2.5', image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('f'):
            print("\nFINISH signal sent! Check C++ terminal for the Lattice.")
            sock.sendto(json.dumps({"type": "FINISH_BATCH"}).encode('utf-8'), server_address)

    cap.release()
    cv2.destroyAllWindows()
    holistic.close()

if __name__ == "__main__":
    main()
