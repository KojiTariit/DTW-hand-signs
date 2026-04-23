import json
import numpy as np
import os
import glob

def extract_all_features(filepath):
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, 'r') as f:
        data = json.load(f)
        
    sequence_features = []
    for frame in data:
        # We simulate the 90-feature (single) or 180-feature (double) logic
        # For simplicity in this diagnostic, we'll look at Face Proximity + Trajectory
        f_vec = []
        
        hands = frame.get("hands", [])
        face = frame.get("face")
        pose = frame.get("pose_anchors")
        
        # We look at up to 2 hands
        for h_idx in range(2):
            if h_idx < len(hands) and face and pose:
                hand = hands[h_idx]
                fh = np.array([face['forehead']['x'], face['forehead']['y'], face['forehead']['z']])
                ch = np.array([face['chin']['x'], face['chin']['y'], face['chin']['z']])
                ns = np.array([face['nose']['x'], face['nose']['y'], face['nose']['z']])
                lc = np.array([face['l_cheek']['x'], face['l_cheek']['y'], face['l_cheek']['z']])
                rc = np.array([face['r_cheek']['x'], face['r_cheek']['y'], face['r_cheek']['z']])
                le = np.array([pose['l_ear']['x'], pose['l_ear']['y'], pose['l_ear']['z']])
                re = np.array([pose['r_ear']['x'], pose['r_ear']['y'], pose['r_ear']['z']])
                
                face_h = np.linalg.norm(fh - ch)
                if face_h < 0.01: face_h = 1.0
                
                wst = np.array([hand['wrist_pos']['x'], hand['wrist_pos']['y'], hand['wrist_pos']['z']])
                idx_tip = np.array([hand['landmarks'][8]['x'], hand['landmarks'][8]['y'], hand['landmarks'][8]['z']]) + wst
                
                # Check proximity to all 7 anchors
                anchors = [fh, ch, ns, lc, rc, le, re]
                for a in anchors:
                    f_vec.append(np.linalg.norm(idx_tip - a) / face_h)
                
                # Trajectory (relative to nose)
                f_vec.append(wst[0]) # X
                f_vec.append(wst[1]) # Y
            else:
                f_vec.extend([0.0] * 9) # Padding
        
        sequence_features.append(f_vec)
        
    return np.array(sequence_features)

def compute_dtw_dist(s1, s2):
    # Simple DTW implementation for diagnostic
    n, m = len(s1), len(s2)
    dtw_matrix = np.zeros((n+1, m+1))
    dtw_matrix[1:, 0] = np.inf
    dtw_matrix[0, 1:] = np.inf
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = np.linalg.norm(s1[i-1] - s2[j-1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])
            
    return dtw_matrix[n, m] / (n + m)

print("--- System Diagnostic: Multi-Sign Consistency Check ---")
template_files = glob.glob("templates/movement/**/*.json", recursive=True)

templates = {}
for f in template_files:
    name = os.path.basename(f)
    feat = extract_all_features(f)
    if feat is not None:
        templates[name] = feat

if not templates:
    print("No templates found!")
    exit()

print(f"Loaded {len(templates)} templates. Comparing distances...\n")

# Matrix: Row vs Column
names = list(templates.keys())
for i, name1 in enumerate(names):
    print(f"[{name1}] results:")
    for j, name2 in enumerate(names):
        if i == j: continue
        dist = compute_dtw_dist(templates[name1], templates[name2])
        status = "PASS" if dist > 0.15 else "WARN"
        print(f"  vs {name2:20} : Score {dist:.4f} {status}")
    print("-" * 40)
