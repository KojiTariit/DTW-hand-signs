import json
import os
import glob
import numpy as np
import joblib

def extract_features(hand_data):
    """
    UPGRADED: Pairwise Distance Matrix + Joint Angles.
    Matches the logic in train_ml.py exactly.
    """
    lms = hand_data['landmarks']
    points = np.array([[lm['x'], lm['y'], lm['z']] for lm in lms])
    
    hand_size = np.linalg.norm(points[9] - points[0])
    if hand_size < 1e-6: hand_size = 1.0

    features = []
    # 1. Full Pairwise Distances (210 pairs)
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            features.append(float(np.linalg.norm(points[i] - points[j]) / hand_size))

    # 2. Joint Angles
    finger_chains = [
        [0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12],
        [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]
    ]
    for chain in finger_chains:
        for i in range(1, len(chain) - 1):
            a, b, c = points[chain[i-1]], points[chain[i]], points[chain[i+1]]
            ba, bc = a - b, c - b
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
            features.append(float(cosine_angle))

    return features

def main():
    try:
        clf = joblib.load("static_ml_model.pkl")
    except:
        print("!! Error: static_ml_model.pkl not found.")
        return

    test_dir = r"c:/Users/USER/Desktop/DTW/templates/Test_case/static/"
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
                feat = extract_features(frame["hands"][0])
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
