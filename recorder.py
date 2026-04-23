"""
Signs Sense - Sign Template Recorder (Gen-2.5)
Press 'R' to toggle recording on and off.
Press 'S' to save the recording to a JSON file.
"""

import cv2
from mediapipe.python.solutions import holistic as mp_holistic
from mediapipe.python.solutions import drawing_utils as mp_drawing
import json
import time
import os

def main():
    # --- 1. SETUP MODEL ---
    # Upgraded to Complexity 1 for "Pure Gold" quality data.
    holistic = mp_holistic.Holistic(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 60)
    
    # --- 2. RECORDING STATE VARIABLES ---
    recording = False        
    waiting_for_hand = False
    arming_start_time = 0
    current_sequence = []    
    
    print("--- Gen-2.5 Template Recorder (10-Anchor Star) ---")
    print("Commands:")
    print("  'r' : Start/Stop recording")
    print("  's' : Save current recording")
    print("  'q' : Quit")

    while cap.isOpened():
        success, image = cap.read()
        if not success: break

        # --- 3. PERFORMANCE OPTIMIZATION ---
        # 320x240 is mandatory for Complexity 1 on laptops.
        image = cv2.resize(image, (320, 240))
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = holistic.process(image_rgb)

        # --- 3. ARMING TIMER LOGIC ---
        status_text = "Status: IDLE"
        if arming_start_time > 0:
            elapsed = time.time() - arming_start_time
            if elapsed < 1.0:
                status_text = f"Status: ARMING in {1.0 - elapsed:.1f}s..."
            else:
                arming_start_time = 0
                waiting_for_hand = True
                print("Armed! Waiting for hand to enter frame...")

        if waiting_for_hand:
            status_text = "Status: WAITING FOR HAND..."
        elif recording:
            status_text = "Status: RECORDING..."

        # --- 4. DRAW SKELETON & ANCHORS ---
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        
        # Draw the 12 Spatial Anchors (8 Face + 2 Ears + 2 Shoulders)
        h, w, c = image.shape
        # Face: indices 10, 152, 4, 234, 454, 13, 133, 362
        # Ears/Shoulders: pose indices 7, 8, 11, 12
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

        # --- 5. RECORDING LOGIC ---
        has_hand = results.left_hand_landmarks or results.right_hand_landmarks
        
        if waiting_for_hand and has_hand:
            waiting_for_hand = False
            recording = True
            print("Hand detected! Recording started...")

        if recording:
            frame_data = {
                "hands": [],
                "face": None,
                "ears": None,
                "timestamp": time.time()
            }
            
            # A. Extract Hands (Normalized to Wrist, but Wrist is relative to NOSE)
            def pack_hand(lms, origin_nose):
                wrist = lms.landmark[0]
                frame_lms = []
                for lm in lms.landmark:
                    frame_lms.append({"x": lm.x - wrist.x, "y": lm.y - wrist.y, "z": lm.z - wrist.z})
                
                # NOSE ANCHOR: Wrist position is now relative to Nose
                rel_wrist = {"x": 0, "y": 0, "z": 0}
                if origin_nose:
                    rel_wrist = {"x": wrist.x - origin_nose.x, "y": wrist.y - origin_nose.y, "z": wrist.z - origin_nose.z}
                
                return {"landmarks": frame_lms, "wrist_pos": rel_wrist}

            nose_lm = results.face_landmarks.landmark[4] if results.face_landmarks else None

            if results.left_hand_landmarks:
                frame_data["hands"].append(pack_hand(results.left_hand_landmarks, nose_lm))
            if results.right_hand_landmarks:
                frame_data["hands"].append(pack_hand(results.right_hand_landmarks, nose_lm))

            # B. Extract Face (8 Anchors - Relative to Nose)
            if results.face_landmarks:
                f = results.face_landmarks.landmark
                n = f[4] # Nose origin
                frame_data["face"] = {
                    "forehead": {"x": f[10].x - n.x, "y": f[10].y - n.y, "z": f[10].z - n.z},
                    "chin": {"x": f[152].x - n.x, "y": f[152].y - n.y, "z": f[152].z - n.z},
                    "nose": {"x": 0, "y": 0, "z": 0},
                    "l_cheek": {"x": f[234].x - n.x, "y": f[234].y - n.y, "z": f[234].z - n.z},
                    "r_cheek": {"x": f[454].x - n.x, "y": f[454].y - n.y, "z": f[454].z - n.z},
                    "mouth": {"x": f[13].x - n.x, "y": f[13].y - n.y, "z": f[13].z - n.z},
                    "l_eye": {"x": f[133].x - n.x, "y": f[133].y - n.y, "z": f[133].z - n.z},
                    "r_eye": {"x": f[362].x - n.x, "y": f[362].y - n.y, "z": f[362].z - n.z}
                }

            # C. Extract Ears & Shoulders (Pose Anchors - Relative to Nose)
            if results.pose_landmarks and nose_lm:
                p = results.pose_landmarks.landmark
                n = nose_lm
                frame_data["pose_anchors"] = {
                    "l_ear": {"x": p[7].x - n.x, "y": p[7].y - n.y, "z": p[7].z - n.z},
                    "r_ear": {"x": p[8].x - n.x, "y": p[8].y - n.y, "z": p[8].z - n.z},
                    "l_shoulder": {"x": p[11].x - n.x, "y": p[11].y - n.y, "z": p[11].z - n.z},
                    "r_shoulder": {"x": p[12].x - n.x, "y": p[12].y - n.y, "z": p[12].z - n.z}
                }
            else:
                frame_data["pose_anchors"] = None

            current_sequence.append(frame_data)

            # --- FREEZE-TO-STOP DETECTION (Auto-Stop Mapping) ---
            # This logic only runs while 'recording' is True (after hand was detected).
            # It calculates the average 'velocity' of your wrist relative to your nose.
            # If the movement stays below a threshold for 0.4 seconds, it stops for you.
            if len(current_sequence) > 12: # We look back at the last 12 frames (~0.4s)
                recent = current_sequence[-12:]
                total_move = 0
                for idx in range(1, len(recent)):
                    # Check if both frames have hand data to calculate distance
                    if recent[idx]["hands"] and recent[idx-1]["hands"]:
                        p1 = recent[idx]["hands"][0]["wrist_pos"]
                        p0 = recent[idx-1]["hands"][0]["wrist_pos"]
                        # Path distance between frames
                        total_move += ((p1['x'] - p0['x'])**2 + (p1['y'] - p0['y'])**2)**0.5
                
                # --- CALIBRATION SETTING ---
                # SETTING: 0.006 is the 'freeze' sensitivity. 
                # TO DISABLE AUTO-STOP: Comment out the next 5 lines if you prefer manual stops ('r').
                if total_move < 0.006: 
                    recording = False
                    current_sequence = current_sequence[:-12] # Trim off the frozen frames
                    print(f"Freeze detected! Auto-stopped. Final count: {len(current_sequence)}")

        # --- 6. DISPLAY UI ---
        color = (0, 255, 0) # Green for Idle
        if recording: color = (0, 0, 255) # Red for Recording
        if waiting_for_hand or arming_start_time > 0: color = (0, 255, 255) # Yellow
        
        cv2.putText(image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(image, f"Frames: {len(current_sequence)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow('Gen-2.5 Recorder', image)

        # --- 7. KEYBOARD CONTROLS ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            if not recording and not waiting_for_hand:
                arming_start_time = time.time()
                current_sequence = []
                print("Initializing Arming Timer...")
            else:
                recording = False
                waiting_for_hand = False
                arming_start_time = 0
                print(f"Stopped. Captured {len(current_sequence)} frames.")
        
        elif key == ord('s'):
            if not current_sequence:
                print("Nothing to save!")
                continue
            
            # --- AUTO-TRIM TRAILING EMPTY FRAMES ---
            last_hand_idx = -1
            for idx, frame in enumerate(current_sequence):
                if frame["hands"]:
                    last_hand_idx = idx
            
            if last_hand_idx == -1:
                print("No hands detected in recording. Discarding.")
                current_sequence = []
                continue
            
            trimmed_sequence = current_sequence[:last_hand_idx + 1]
            print(f"Auto-Trim: Removed {len(current_sequence) - len(trimmed_sequence)} trailing frames.")

            # --- Choice 2: Category ---
            print("\nSelect Category:")
            print("1. static (Alphabet)")
            print("2. movement (Words)")
            cat_choice = input("Enter 1 or 2: ").strip()
            
            category = "static" if cat_choice == "1" else "movement"
            
            print("\nSelect Number of Hands:")
            print("1. single_hand")
            print("2. 2_hands")
            choices = input("Enter 1 or 2: ").strip()
            
            num_hand = "single_hand" if choices == "1" else "2_hands"

            filename = input("Enter sign name: ").strip()
            
            # Final Path: templates/{purpose}/{category}/{name}.json
            dir_path = f"c:/Users/USER/Desktop/DTW/templates/{category}/{num_hand}"
            os.makedirs(dir_path, exist_ok=True)
            
            filepath = f"{dir_path}/{filename}.json"
            with open(filepath, 'w') as f:
                json.dump(trimmed_sequence, f)
            print(f"Saved to {filepath}")
            current_sequence = []

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    holistic.close()

if __name__ == "__main__":
    main()
