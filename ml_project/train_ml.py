import json
import os
import glob
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def extract_features(hand_data):
    """
    HYBRID CORE (44 Features):
    1. Wrist-Relative Distances (20) - The "Big Picture"
    2. Finger Extension Ratios (5) - Curl Detection
    3. Fingertip Spreads (4) - U vs V Detection
    4. Joint Angles (15) - Fine Pose
    """
    lms = hand_data['landmarks']
    pts = np.array([[lm['x'], lm['y'], lm['z']] for lm in lms])
    
    p0 = pts[0]
    p9 = pts[9]
    hand_size = np.linalg.norm(p9 - p0)
    if hand_size < 1e-6: hand_size = 1.0

    features = []

    # 1. Wrist-Relative Distances (20 features) - BACK IN!
    for i in range(1, 21):
        features.append(np.linalg.norm(pts[i] - p0) / hand_size)

    # 2. Finger Extension Ratios (5 features)
    tips = [4, 8, 12, 16, 20]
    mcps = [2, 5, 9, 13, 17]
    for t, m in zip(tips, mcps):
        features.append(np.linalg.norm(pts[t] - pts[m]) / (np.linalg.norm(pts[m] - p0) + 1e-6))

    # 3. Tip Spreads (4 features)
    for i in range(4):
        features.append(np.linalg.norm(pts[tips[i]] - pts[tips[i+1]]) / hand_size)

    # 4. Joint Angles (15 features)
    chains = [[0,1,2,3,4],[0,5,6,7,8],[0,9,10,11,12],[0,13,14,15,16],[0,17,18,19,20]]
    for chain in chains:
        for i in range(1, 4):
            ba = pts[chain[i-1]] - pts[chain[i]]
            bc = pts[chain[i+1]] - pts[chain[i]]
            cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
            features.append(float(cos))

    return features

def main():
    print("=== Signs Sense: HYBRID TRAINER (44 Features) ===")
    static_dir = r"c:/Users/USER/Desktop/DTW/templates/Train_case/static/"
    files = sorted(glob.glob(os.path.join(static_dir, "*.json")))
    
    X, y = [], []
    for f in files:
        sign = os.path.basename(f).split('.')[0].split('_')[0]
        with open(f, 'r') as jf:
            for frame in json.load(jf):
                if frame.get("hands"):
                    X.append(extract_features(frame["hands"][0]))
                    y.append(sign)

    X, y = np.array(X), np.array(y)
    clf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
    clf.fit(X, y)
    
    print(f"Training complete. Features: {len(X[0])}. Self-Accuracy: {accuracy_score(y, clf.predict(X)) * 100:.2f}%")

    joblib.dump(clf, "static_ml_model.pkl")
    joblib.dump(clf.classes_, "static_ml_classes.pkl")
    print("Model Exported -> 'static_ml_model.pkl'")

if __name__ == "__main__":
    main()
