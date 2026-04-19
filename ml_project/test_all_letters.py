import json
import os
import glob
import numpy as np
import joblib

def extract_features(frame):
    """
    HYBRID CORE (58 Features - Sniper Optimized)
    Matches the logic in train_ml.py exactly.
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

    # 4. Face Context (4 features)
    if frame.get("face"):
        face = frame["face"]
        fh = np.array([face['forehead']['x'], face['forehead']['y'], face['forehead']['z']])
        ch = np.array([face['chin']['x'], face['chin']['y'], face['chin']['z']])
        d1 = float(np.linalg.norm(pts[4] - fh))
        d2 = float(np.linalg.norm(pts[4] - ch))
        d3 = float(np.linalg.norm(p0 - fh))
        d4 = float(np.linalg.norm(p0 - ch))
        features.extend([d1, d2, d3, d4])
    else:
        features.extend([0.0, 0.0, 0.0, 0.0])

    # 5. Full Tip Matrix (10 features) - Solves U vs V natively
    for i in range(len(tips)):
        for j in range(i+1, len(tips)):
            features.append(float(np.linalg.norm(pts[tips[i]] - pts[tips[j]]) / hand_size))

    # 6. Thumb-Cross Matrix (4 features) - Solves A vs S natively
    cross_pips = [6, 10, 14, 18]
    for m in cross_pips:
        features.append(float(np.linalg.norm(pts[4] - pts[m]) / hand_size))

    return features

def main():
    try:
        clf = joblib.load("static_ml_model.pkl")
    except:
        print("!! Error: static_ml_model.pkl not found.")
        return

    test_dir = r"c:/Users/USER/Desktop/DTW/templates/Train_case/static/"
    files = sorted(glob.glob(os.path.join(test_dir, "*.json")))

    print(f"\n{'Letter':<8} | {'Pred (Mid)':<15} | {'Status'}")
    print("-" * 55)

    total_correct = 0
    for f in files:
        actual = os.path.basename(f).split('.')[0].split('_')[0]
        with open(f, 'r') as jf:
            data = json.load(jf)
        
        preds = []
        for frame in data:
            if "hands" in frame and len(frame["hands"]) > 0:
                feat = extract_features(frame)
                preds.append(clf.predict([feat])[0])
        
        if not preds: continue
        
        mid_pred = preds[len(preds)//2]
        all_same = all(p == actual for p in preds)
        
        if all_same:
            total_correct += 1
            status = "100% OK"
        else:
            status = f"SHAKY ({set(preds)})"
        
        print(f"{actual:<8} | {mid_pred:<15} | {status}")

    print("-" * 55)
    print(f"REPORT CARD: {total_correct}/{len(files)} Correct ({(total_correct/len(files))*100:.2f}%)")

if __name__ == "__main__":
    main()
