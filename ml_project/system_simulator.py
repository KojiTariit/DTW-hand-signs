import json
import os
import glob
import numpy as np
import joblib
from collections import defaultdict

# ==========================================
# 1. FEATURE EXTRACTION (Matches C++ Logic)
# ==========================================

def extract_ml_features(hand_data):
    """HYBRID CORE (44 Features)"""
    lms = hand_data['landmarks']
    pts = np.array([[lm['x'], lm['y'], lm['z']] for lm in lms])
    
    p0 = pts[0]
    p9 = pts[9]
    hand_size = np.linalg.norm(p9 - p0)
    if hand_size < 1e-6: hand_size = 1.0

    features = []

    # 1. Wrist-Relative Distances (20)
    for i in range(1, 21):
        features.append(np.linalg.norm(pts[i] - p0) / hand_size)

    # 2. Finger Extension Ratios (5)
    tips = [4, 8, 12, 16, 20]
    mcps = [2, 5, 9, 13, 17]
    for t, m in zip(tips, mcps):
        features.append(np.linalg.norm(pts[t] - pts[m]) / (np.linalg.norm(pts[m] - p0) + 1e-6))

    # 3. Tip Spreads (4)
    for i in range(4):
        features.append(np.linalg.norm(pts[tips[i]] - pts[tips[i+1]]) / hand_size)

    # 4. Joint Angles (15)
    chains = [[0,1,2,3,4],[0,5,6,7,8],[0,9,10,11,12],[0,13,14,15,16],[0,17,18,19,20]]
    for chain in chains:
        for i in range(1, 4):
            ba = pts[chain[i-1]] - pts[chain[i]]
            bc = pts[chain[i+1]] - pts[chain[i]]
            cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
            features.append(float(cos))

    return features

def extract_dtw_features(sequence):
    """Weighted features for Dynamic signs (Matches DtwEngine.hpp)"""
    feat_seq = []
    if not sequence: return feat_seq
    
    start_frame = sequence[0]
    
    for i in range(1, len(sequence)):
        f = sequence[i]
        frame_feat = []
        for h_idx, hand in enumerate(f['hands']):
            lms = np.array([[lm['x'], lm['y'], lm['z']] for lm in hand['landmarks']])
            
            # 1. Handshape (Weighted 21 Landmarks)
            weights = np.ones(21)
            for tip in [4, 8, 12, 16, 20]: weights[tip] = 5.0
            sq_weights = np.sqrt(weights)
            for j in range(21):
                frame_feat.extend(lms[j] * sq_weights[j])
            
            # 2. Finger Extension
            tips = [4, 8, 12, 16, 20]
            for t in tips:
                frame_feat.append(np.linalg.norm(lms[t]) * 6.0)
                
            # 3. Finger Spread
            for j in range(4):
                frame_feat.append(np.linalg.norm(lms[tips[j]] - lms[tips[j+1]]) * 6.0)
                
            # 4. Trajectory (Relative to start)
            if h_idx < len(start_frame['hands']):
                sw = start_frame['hands'][h_idx]['wrist_pos']
                dx = hand['wrist_pos']['x'] - sw['x']
                dy = hand['wrist_pos']['y'] - sw['y']
                frame_feat.append(dx * 7.0)
                frame_feat.append(dy * 7.0)
            else:
                frame_feat.extend([0.0, 0.0])
        
        if frame_feat: feat_seq.append(frame_feat)
    return feat_seq

# ==========================================
# 2. MATCHING ENGINES (Matches C++ Logic)
# ==========================================

def compute_dtw(seq1, seq2, window_percent=0.15):
    n, m = len(seq1), len(seq2)
    window = max(int(max(n, m) * window_percent), abs(n - m))
    
    dtw_matrix = np.full((n + 1, m + 1), 999999.0)
    dtw_matrix[0, 0] = 0.0
    
    for i in range(1, n + 1):
        for j in range(max(1, i - window), min(m, i + window) + 1):
            f1, f2 = np.array(seq1[i-1]), np.array(seq2[j-1])
            if f1.shape != f2.shape:
                cost = 999.0 # High penalty for hand count mismatch
            else:
                cost = np.linalg.norm(f1 - f2)
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])
            
    return dtw_matrix[n, m] / (n + m)

def compute_dual_score(seq1, seq2, alpha=0.4):
    shape = compute_dtw(seq1, seq2)
    
    # Rhythm (Derivative) logic
    def get_derivs(seq):
        res = []
        for i in range(1, len(seq)):
            f1, f2 = np.array(seq[i-1]), np.array(seq[i])
            if f1.shape == f2.shape:
                res.append(f2 - f1)
        return res

    d1, d2 = get_derivs(seq1), get_derivs(seq2)
    rhythm = compute_dtw(d1, d2) if d1 and d2 else 9999.0
    
    return (alpha * shape) + ((1.0 - alpha) * rhythm)

# ==========================================
# 3. SIMULATION RUNNER
# ==========================================

def run_simulation():
    print("\n" + "="*50)
    print(" SYSTEM ACCURACY SIMULATOR (Gen-2.5) ")
    print("="*50)

    # A. Initializing
    try:
        clf = joblib.load("static_ml_model.pkl")
        classes = joblib.load("static_ml_classes.pkl")
    except:
        print("!! Error: Model files not found. Run train_ml.py first.")
        return

    # B. Load Dictionary (Train_case)
    print("\n[1] Loading Dictionary from Train_case...")
    dictionary = {"movement": {}}
    train_dir = r"c:/Users/USER/Desktop/DTW/templates/Train_case/movement"
    for f in glob.glob(os.path.join(train_dir, "*.json")):
        name = os.path.basename(f).split('.')[0].split('_')[0]
        with open(f, 'r') as jf:
            seq = json.load(jf)
            dictionary["movement"][name] = extract_dtw_features(seq)
    print(f"    Loaded {len(dictionary['movement'])} movement templates.")

    # C. Process Test Cases
    test_root = r"c:/Users/USER/Desktop/DTW/templates/Test_case"
    stats = {
        "total": 0, "correct": 0,
        "router_fail": 0, "engine_fail": 0,
        "categories": { "static": {"total": 0, "ok": 0}, "movement": {"total": 0, "ok": 0} }
    }

    print("\n[2] Running Simulation...")
    print(f"{'Actual':<12} | {'Router':<10} | {'Prediction':<15} | {'Status'}")
    print("-" * 60)

    for cat in ["static", "movement"]:
        files = glob.glob(os.path.join(test_root, cat, "*.json"))
        for f in sorted(files):
            actual_name = os.path.basename(f).split('.')[0].split('_')[0]
            with open(f, 'r') as jf:
                sequence = json.load(jf)
            
            if not sequence: continue
            stats["total"] += 1
            stats["categories"][cat]["total"] += 1

            # --- ROUTING LOGIC (from scrap_receiver.cpp) ---
            max_wrist_dist = 0.0
            max_shape_variance = 0.0
            start_f = sequence[0]
            for frame in sequence:
                if frame['hands'] and start_f['hands']:
                    h = frame['hands'][0]
                    sh = start_f['hands'][0]
                    d = np.linalg.norm(np.array([h['wrist_pos']['x'], h['wrist_pos']['y'], h['wrist_pos']['z']]) - 
                                       np.array([sh['wrist_pos']['x'], sh['wrist_pos']['y'], sh['wrist_pos']['z']]))
                    max_wrist_dist = max(max_wrist_dist, d)
            
            is_dynamic = (max_wrist_dist > 0.15 or max_shape_variance > 0.06)
            router_pred = "movement" if is_dynamic else "static"
            
            # --- CLASSIFICATION ---
            prediction = "None"
            if router_pred == "static":
                mid_frame = sequence[len(sequence)//2]
                if mid_frame['hands']:
                    feat = extract_ml_features(mid_frame['hands'][0])
                    prediction = clf.predict([feat])[0]
            else:
                live_feat = extract_dtw_features(sequence)
                best_score = 9999.0
                for name, temp_feat in dictionary["movement"].items():
                    score = compute_dual_score(live_feat, temp_feat)
                    if score < best_score:
                        best_score, prediction = score, name

            # --- LOGGING ---
            is_ok = (prediction == actual_name)
            router_ok = (router_pred == cat)
            
            if is_ok: stats["correct"] += 1; stats["categories"][cat]["ok"] += 1
            if not router_ok: stats["router_fail"] += 1
            elif not is_ok: stats["engine_fail"] += 1

            status = "PASS" if is_ok else "FAIL"
            print(f"{actual_name:<12} | {router_pred:<10} | {prediction:<15} | {status}")

    # D. Final Report
    print("\n" + "="*50)
    print(" FINAL REPORT CARD ")
    print("="*50)
    print(f"  SYSTEM ACCURACY : {stats['correct']/stats['total']*100:.2f}% ({stats['correct']}/{stats['total']})")
    print(f"  ROUTER SUCCESS  : {(stats['total']-stats['router_fail'])/stats['total']*100:.2f}%")
    print(f"  ML SUCCESS      : {stats['categories']['static']['ok']/stats['categories']['static']['total']*100:.2f}%")
    if stats['categories']['movement']['total'] > 0:
        print(f"  DTW SUCCESS     : {stats['categories']['movement']['ok']/stats['categories']['movement']['total']*100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    run_simulation()
