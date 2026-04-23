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

    # 1. Wrist-Relative Distances (20 features)
    for i in range(1, 21):
        features.append(float(np.linalg.norm(pts[i] - p0) / hand_size))

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

    # 6. Thumb-Cross Matrix (8 features) - Solves A vs S (PIPs) and M/N/T (MCPs)
    cross_pips = [6, 10, 14, 18]
    cross_mcps = [5, 9, 13, 17]
    for m in cross_pips:
        features.append(float(np.linalg.norm(pts[4] - pts[m]) / hand_size))
    for m in cross_mcps:
        features.append(float(np.linalg.norm(pts[4] - pts[m]) / hand_size))

    return features

def main():
    print("=== Signs Sense: HYBRID TRAINER (77 Features 'Sniper Core+') ===")
    static_dir = r"c:/Users/USER/Desktop/DTW/templates_backup/static/"
    files = sorted(glob.glob(os.path.join(static_dir, "**/*.json"), recursive=True))
    
    X, y = [], []
    for f in files:
        sign = os.path.basename(f).split('.')[0].split('_')[0]
        with open(f, 'r') as jf:
            try:
                for frame in json.load(jf):
                    if frame.get("hands"):
                        X.append(extract_features(frame))
                        y.append(sign)
            except Exception as e:
                print(f"Skipping corrupt or empty file: {f}")

    X, y = np.array(X), np.array(y)
    clf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
    clf.fit(X, y)
    
    print(f"Training complete. Features: {len(X[0])}. Self-Accuracy: {accuracy_score(y, clf.predict(X)) * 100:.2f}%")

    joblib.dump(clf, "static_ml_model.pkl")
    joblib.dump(clf.classes_, "static_ml_classes.pkl")
    print("Model Exported -> 'static_ml_model.pkl'")

if __name__ == "__main__":
    main()
