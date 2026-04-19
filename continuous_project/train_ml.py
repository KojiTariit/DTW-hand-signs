import json
import os
import glob
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

def extract_features(frame_data):
    lms = frame_data['landmarks']
    # 0 = Wrist, 9 = Middle MCP (good anchor for hand size)
    p0 = np.array([lms[0]['x'], lms[0]['y'], lms[0]['z']])
    p9 = np.array([lms[9]['x'], lms[9]['y'], lms[9]['z']])
    hand_size = np.linalg.norm(p9 - p0)
    if hand_size < 1e-6: hand_size = 1.0

    features = []

    # 1. Normalized Distances from Wrist
    for i in range(1, 21):
        pi = np.array([lms[i]['x'], lms[i]['y'], lms[i]['z']])
        dist = np.linalg.norm(pi - p0) / hand_size
        features.append(dist)

    # 2. Joint Angles (Bending)
    # Fingers: [1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16], [17,18,19,20]
    finger_chains = [
        [0, 1, 2, 3, 4],    # Thumb
        [0, 5, 6, 7, 8],    # Index
        [0, 9, 10, 11, 12], # Middle
        [0, 13, 14, 15, 16],# Ring
        [0, 17, 18, 19, 20] # Pinky
    ]

    for chain in finger_chains:
        for i in range(1, len(chain) - 1):
            a = np.array([lms[chain[i-1]]['x'], lms[chain[i-1]]['y'], lms[chain[i-1]]['z']])
            b = np.array([lms[chain[i]]['x'], lms[chain[i]]['y'], lms[chain[i]]['z']])
            c = np.array([lms[chain[i+1]]['x'], lms[chain[i+1]]['y'], lms[chain[i+1]]['z']])
            
            ba = a - b
            bc = c - b
            
            # Cosine of the angle
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
            features.append(float(cosine_angle))

    # 3. Inter-finger spread angles (Angle between MCP bases)
    mcp_indices = [1, 5, 9, 13, 17]
    for i in range(len(mcp_indices) - 1):
        # Vector from wrist to MCP i and i+1
        v1 = np.array([lms[mcp_indices[i]]['x'], lms[mcp_indices[i]]['y'], lms[mcp_indices[i]]['z']]) - p0
        v2 = np.array([lms[mcp_indices[i+1]]['x'], lms[mcp_indices[i+1]]['y'], lms[mcp_indices[i+1]]['z']]) - p0
        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        features.append(float(cosine_angle))

    # 4. Fingertip Spreads (Normalized distances between adjacent fingertips)
    # This is critical for distinguishing H (fingers together) vs G (middle curled away)
    tips = [4, 8, 12, 16, 20]
    for i in range(len(tips) - 1):
        t1 = np.array([lms[tips[i]]['x'], lms[tips[i]]['y'], lms[tips[i]]['z']])
        t2 = np.array([lms[tips[i+1]]['x'], lms[tips[i+1]]['y'], lms[tips[i+1]]['z']])
        spread = np.linalg.norm(t1 - t2) / hand_size
        features.append(float(spread))

    return features

# 1. Load Data
static_dir = r"c:/Users/USER/Desktop/DTW/templates/static/"
files = sorted(glob.glob(os.path.join(static_dir, "*.json")))

X = [] # Features
y = [] # Labels

print("Loading dataset from JSON templates...")
for f in files:
    # sign_name logic: "G_1.json" -> "G", "Apple_2.json" -> "Apple"
    base = os.path.basename(f).split('.')[0]
    sign_name = base.split('_')[0]
    
    with open(f, 'r') as jf:
        data = json.load(jf)
        
        # Treat every single frame in the template as a training example!
        for frame in data:
            features = extract_features(frame)
            X.append(features)
            y.append(sign_name)

X = np.array(X)
y = np.array(y)

print(f"Total training examples: {len(X)}")
print(f"Features per example: {len(X[0])}")

# 2. Split into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train the Model (Random Forest is lightweight and fantastic for this)
print("\nTraining Random Forest model...")
clf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
clf.fit(X_train, y_train)

# 4. Evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy on unseen static frames: {accuracy * 100:.2f}%")

print("\nDetailed Report:")
print(classification_report(y_test, y_pred))

# 5. Save the model
joblib.dump(clf, "static_ml_model.pkl")
joblib.dump(clf.classes_, "static_ml_classes.pkl")
print("\nModel saved locally as 'static_ml_model.pkl'")
