import json
import os
import glob
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def extract_features(frame):
    """
    HYBRID CORE (58 Features - Sniper Optimized):
    1. Wrist-Relative Distances (20)
    2. Finger Extension Ratios (5)
    3. Joint Angles (15)
    4. Face Context (4)
    5. Full Tip Matrix (10)       -> Solves U vs V spacing.
    6. Thumb-Cross Matrix (4)     -> Solves A vs S wrapping.
    """
    hand_data = frame["hands"][0]
    lms = hand_data['landmarks']
    pts = np.array([[lm['x'], lm['y'], lm['z']] for lm in lms])
    
    p0 = pts[0]
    p9 = pts[9]
    hand_size = np.linalg.norm(p9 - p0)
    if hand_size < 1e-6: hand_size = 1.0

    features = []

    # 1. Wrist-Relative XYZ Coordinates (60 features) - Solves Orientation (H vs R)
    for i in range(1, 21):
        diff = pts[i] - p0
        features.extend((diff / hand_size).tolist())

    # 2. Finger Extension Ratios (5) - Curl Detection
    tips = [4, 8, 12, 16, 20]
    mcps = [2, 5, 9, 13, 17]
    for t, m in zip(tips, mcps):
        features.append(float(np.linalg.norm(pts[t] - pts[m]) / (np.linalg.norm(pts[m] - p0) + 1e-6)))

    # 3. Joint Angles (15 features)
    chains = [[0,1,2,3,4],[0,5,6,7,8],[0,9,10,11,12],[0,13,14,15,16],[0,17,18,19,20]]
    for chain in chains:
        for i in range(1, 4):
            ba = pts[chain[i-1]] - pts[chain[i]]
            bc = pts[chain[i+1]] - pts[chain[i]]
            cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
            features.append(float(cos))

    # 4. Face Context (16 Semantic Probes) - Normalized by Face Height for Zoom-Independence
    if frame.get("face") and frame.get("pose_anchors"):
        face = frame["face"]
        pose = frame["pose_anchors"]
        fh = np.array([face['forehead']['x'], face['forehead']['y'], face['forehead']['z']])
        ch = np.array([face['chin']['x'], face['chin']['y'], face['chin']['z']])
        ns = np.array([face['nose']['x'], face['nose']['y'], face['nose']['z']])
        lc = np.array([face['l_cheek']['x'], face['l_cheek']['y'], face['l_cheek']['z']])
        rc = np.array([face['r_cheek']['x'], face['r_cheek']['y'], face['r_cheek']['z']])
        le = np.array([pose['l_ear']['x'], pose['l_ear']['y'], pose['l_ear']['z']])
        re = np.array([pose['r_ear']['x'], pose['r_ear']['y'], pose['r_ear']['z']])
        
        face_h = np.linalg.norm(fh - ch)
        if face_h < 0.01: face_h = 1.0
        
        w_abs = np.array([hand_data['wrist_pos']['x'], hand_data['wrist_pos']['y'], hand_data['wrist_pos']['z']])
        
        # Landmarks are relative to wrist, wrist is relative to Nose
        idx_rel_nose = pts[8] + w_abs
        mid_rel_nose = pts[12] + w_abs
        tmb_rel_nose = pts[4] + w_abs
        wst_rel_nose = w_abs
        
        anchors = [fh, ch, ns, lc, rc, le, re]
        # 1. Index Probes (7)
        for a in anchors:
            features.append(float(np.linalg.norm(idx_rel_nose - a) / face_h))
        # 2. Middle Probes (7)
        for a in anchors:
            features.append(float(np.linalg.norm(mid_rel_nose - a) / face_h))
        # 3. Thumb Probes (7)
        for a in anchors:
            features.append(float(np.linalg.norm(tmb_rel_nose - a) / face_h))
        # 4. Wrist Probes (2)
        features.append(float(np.linalg.norm(wst_rel_nose - fh) / face_h))
        features.append(float(np.linalg.norm(wst_rel_nose - ch) / face_h))
    else:
        # Fill 23 zeros
        features.extend([0.0] * 23)

    # 5. Full Tip Matrix (10 features) - Solves U vs V natively
    for i in range(len(tips)):
        for j in range(i+1, len(tips)):
            features.append(float(np.linalg.norm(pts[tips[i]] - pts[tips[j]]) / hand_size))

    # 6. Thumb-Cross Matrix (4 features) - Solves A vs S (PIPs)
    cross_pips = [6, 10, 14, 18]
    for m in cross_pips:
        features.append(float(np.linalg.norm(pts[4] - pts[m]) / hand_size))
    # 7. Palm Orientation (3 features) - Solves Palm Up/Down
    v1 = pts[5] - pts[0]
    v2 = pts[17] - pts[0]
    normal = np.cross(v1, v2)
    norm_mag = np.linalg.norm(normal)
    if norm_mag > 1e-6:
        normal /= norm_mag
        features.extend(normal.tolist())
    else:
        features.extend([0.0, 0.0, 0.0])

    # 8. Orientation Highlighters (2 features: Index and Middle X/Y Ratios)
    idx_ratio, mid_ratio = 0.0, 0.0
    for i, (tip_idx, mcp_idx) in enumerate([(8, 5), (12, 9)]):
        dx = abs(pts[tip_idx][0] - pts[mcp_idx][0])
        dy = abs(pts[tip_idx][1] - pts[mcp_idx][1])
        ratio = float(dx / (dx + dy + 1e-6))
        features.append(ratio)
        if i == 0: idx_ratio = ratio
        else: mid_ratio = ratio

    # 9. FEATURE BOOSTING: Duplicate highlighters 10x to force AI focus
    for _ in range(10):
        features.append(idx_ratio)
        features.append(mid_ratio)

    return features

def main():
    print("=== Signs Sense: HYBRID TRAINER (77 Features 'Sniper Core+') ===")
    # --- Search both the main templates and the backup folders ---
    static_dirs = [
        r"c:/Users/USER/Desktop/DTW/templates/static/",
        r"c:/Users/USER/Desktop/DTW/templates_backup/static/"
    ]
    
    files = []
    for s_dir in static_dirs:
        if os.path.exists(s_dir):
            files.extend(glob.glob(os.path.join(s_dir, "**/*.json"), recursive=True))
    files = sorted(files)
    
    X, y = [], []
    for f in files:
        sign = os.path.basename(f).split('.')[0].split('_')[0]
        with open(f, 'r') as jf:
            try:
                frames = json.load(jf)
                if len(frames) < 5: continue
                
                # Use only the middle 60% of frames (Auto-Trim)
                start_idx = int(len(frames) * 0.2)
                end_idx = int(len(frames) * 0.8)
                
                for frame in frames[start_idx:end_idx]:
                    if frame.get("hands"):
                        feat = extract_features(frame)
                        X.append(feat)
                        y.append(sign)
                        
                        # Data Augmentation (Jittering)
                        jittered_feat = [min(max(val + np.random.normal(0, 0.002), -1.0), 1.0) for val in feat]
                        X.append(jittered_feat)
                        y.append(sign)
                        
            except Exception as e:
                print(f"Skipping {f}: {e}")

    print(f"Training on {len(X)} augmented samples...")
    X, y = np.array(X), np.array(y)
    clf = RandomForestClassifier(n_estimators=300, max_depth=None, class_weight='balanced', random_state=42)
    clf.fit(X, y)
    
    acc = accuracy_score(y, clf.predict(X)) * 100
    print(f"Training complete. Features: {len(X[0])}. Self-Accuracy: {acc:.2f}%")

    joblib.dump(clf, "static_ml_model.pkl")
    joblib.dump(clf.classes_, "static_ml_classes.pkl")
    print("Model Exported -> 'static_ml_model.pkl'")

if __name__ == "__main__":
    main()
