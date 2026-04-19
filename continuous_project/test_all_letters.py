import json
import os
import glob
import numpy as np
import joblib

def extract_features(frame_data):
    lms = frame_data['landmarks']
    p0 = np.array([lms[0]['x'], lms[0]['y'], lms[0]['z']])
    p9 = np.array([lms[9]['x'], lms[9]['y'], lms[9]['z']])
    hand_size = np.linalg.norm(p9 - p0)
    if hand_size < 1e-6: hand_size = 1.0

    features = []
    for i in range(1, 21):
        pi = np.array([lms[i]['x'], lms[i]['y'], lms[i]['z']])
        features.append(np.linalg.norm(pi - p0) / hand_size)

    finger_chains = [
        [0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12],
        [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]
    ]
    for chain in finger_chains:
        for i in range(1, len(chain) - 1):
            a = np.array([lms[chain[i-1]]['x'], lms[chain[i-1]]['y'], lms[chain[i-1]]['z']])
            b = np.array([lms[chain[i]]['x'], lms[chain[i]]['y'], lms[chain[i]]['z']])
            c = np.array([lms[chain[i+1]]['x'], lms[chain[i+1]]['y'], lms[chain[i+1]]['z']])
            ba = a - b; bc = c - b
            features.append(float(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)))

    mcp = [1, 5, 9, 13, 17]
    for i in range(4):
        v1 = np.array([lms[mcp[i]]['x'], lms[mcp[i]]['y'], lms[mcp[i]]['z']]) - p0
        v2 = np.array([lms[mcp[i+1]]['x'], lms[mcp[i+1]]['y'], lms[mcp[i+1]]['z']]) - p0
        features.append(float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)))

    return features

clf = joblib.load("static_ml_model.pkl")
static_dir = r"c:/Users/USER/Desktop/DTW/templates/static/"
files = sorted(glob.glob(os.path.join(static_dir, "*.json")))

print(f"{'Letter':<8} | {'Pred (Mid Frame)':<18} | {'Frame Consistency'}")
print("-" * 55)

for f in files:
    sign = os.path.basename(f).replace(".json", "")
    with open(f, 'r') as jf:
        data = json.load(jf)
    
    # Predict on every frame, then show the tally
    preds = []
    for frame in data:
        feat = extract_features(frame)
        pred = clf.predict([feat])[0]
        preds.append(pred)
    
    mid_pred = preds[len(preds)//2]
    all_same = all(p == sign for p in preds)
    wrong_preds = [p for p in preds if p != sign]
    
    if all_same:
        status = "OK  ALL CORRECT"
    else:
        status = f"ERR Wrong: {set(wrong_preds)}"
    
    print(f"{sign:<8} | {mid_pred:<18} | {status}")
