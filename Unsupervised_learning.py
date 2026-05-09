import json
import glob
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

import math

def magnitude(v):
    return math.sqrt(v['x']**2 + v['y']**2 + v['z']**2)

def sub_points(a, b):
    return {'x': a['x'] - b['x'], 'y': a['y'] - b['y'], 'z': a['z'] - b['z']}

def dot_product(v1, v2):
    return v1['x']*v2['x'] + v1['y']*v2['y'] + v1['z']*v2['z']

def extract_features(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    frame_features = []
    for frame in data:
        # Check if hand exists
        if not frame.get('hands') or len(frame['hands']) == 0:
            continue
            
        hand = None
        for h in frame['hands']:
            if h.get('landmarks') and len(h['landmarks']) >= 21:
                hand = h
                break
                
        if not hand:
            continue
            
        lms = hand['landmarks']
        p0 = lms[0]
        p9 = lms[9]
        
        hand_size = magnitude(sub_points(p9, p0))
        if hand_size < 1e-6:
            hand_size = 1.0
            
        features = []
        
        # 1. Wrist-Relative Distances (20 Features)
        for i in range(1, 21):
            features.append(magnitude(sub_points(lms[i], p0)) / hand_size)
            
        # 2. Finger Extension Ratios (5 Features)
        curltips = [4, 8, 12, 16, 20]
        curlmcps = [2, 5, 9, 13, 17]
        for i in range(5):
            dist_tip_mcp = magnitude(sub_points(lms[curltips[i]], lms[curlmcps[i]]))
            dist_mcp_wrist = magnitude(sub_points(lms[curlmcps[i]], p0))
            features.append(dist_tip_mcp / (dist_mcp_wrist + 1e-6))
            
        # 3. Joint Angles (15 Features)
        chains = [
            [0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], 
            [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]
        ]
        for c in chains:
            for j in range(1, 4):
                a = lms[c[j-1]]
                b = lms[c[j]]
                c_pt = lms[c[j+1]]
                ba = sub_points(a, b)
                bc = sub_points(c_pt, b)
                m_ba = magnitude(ba)
                m_bc = magnitude(bc)
                features.append(dot_product(ba, bc) / (m_ba * m_bc + 1e-6))
                
        # 4. Face Context (23 Semantic Probes)
        face = frame.get('face')
        pose = frame.get('pose_anchors')
        
        if face and pose:
            face_h = magnitude(sub_points(face['forehead'], face['chin']))
            if face_h < 0.01:
                face_h = 1.0
                
            wrist = hand.get('wrist_pos', {'x':0, 'y':0, 'z':0})
            
            idx_rel_nose = {'x': wrist['x'] + lms[8]['x'], 'y': wrist['y'] + lms[8]['y'], 'z': wrist['z'] + lms[8]['z']}
            mid_rel_nose = {'x': wrist['x'] + lms[12]['x'], 'y': wrist['y'] + lms[12]['y'], 'z': wrist['z'] + lms[12]['z']}
            tmb_rel_nose = {'x': wrist['x'] + lms[4]['x'], 'y': wrist['y'] + lms[4]['y'], 'z': wrist['z'] + lms[4]['z']}
            
            anchors = [face['forehead'], face['chin'], face['nose'], face['l_cheek'], face['r_cheek'], pose['l_ear'], pose['r_ear']]
            
            for a in anchors: features.append(magnitude(sub_points(idx_rel_nose, a)) / face_h)
            for a in anchors: features.append(magnitude(sub_points(mid_rel_nose, a)) / face_h)
            for a in anchors: features.append(magnitude(sub_points(tmb_rel_nose, a)) / face_h)
            features.append(magnitude(sub_points(wrist, face['forehead'])) / face_h)
            features.append(magnitude(sub_points(wrist, face['chin'])) / face_h)
        else:
            features.extend([0.0] * 23)
            
        # 5. Full Tip Matrix (10 Features)
        tips = [4, 8, 12, 16, 20]
        for i in range(5):
            for j in range(i + 1, 5):
                features.append(magnitude(sub_points(lms[tips[i]], lms[tips[j]])) / hand_size)
                
        # 6. Thumb-Cross Matrix (4 Features)
        cross_pips = [6, 10, 14, 18]
        for i in range(4):
            features.append(magnitude(sub_points(lms[4], lms[cross_pips[i]])) / hand_size)
            
        # 7. Palm Orientation (3 Features: Palm Normal X, Y, Z)
        v1 = sub_points(lms[5], p0)
        v2 = sub_points(lms[17], p0)
        nx = v1['y']*v2['z'] - v1['z']*v2['y']
        ny = v1['z']*v2['x'] - v1['x']*v2['z']
        nz = v1['x']*v2['y'] - v1['y']*v2['x']
        norm_mag = math.sqrt(nx**2 + ny**2 + nz**2) + 1e-6
        features.extend([nx/norm_mag, ny/norm_mag, nz/norm_mag])

        frame_features.append(features)
    
    if not frame_features:
        return None
    
    # Use the mean of all frames to get a single 77-dimensional vector representing the "Shape" of the sign
    return np.mean(frame_features, axis=0)

# Use recursive=True to find all .json files in your templates folder
json_files = glob.glob("templates/**/*.json", recursive=True) 

all_vectors = []
valid_files = []

for f in json_files:
    feature = extract_features(f)
    if feature is not None:
        all_vectors.append(feature)
        valid_files.append(f)

if len(all_vectors) == 0:
    print("Error")
else:
    X = np.array(all_vectors)

    print(f"running {len(valid_files)}")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_groups = 10
    kmeans = KMeans(n_clusters=n_groups, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    if not os.path.exists('model_output'): os.makedirs('model_output')
    joblib.dump(kmeans, 'model_output/sign_kmeans.pkl')
    joblib.dump(scaler, 'model_output/scaler.pkl')

    # --- NEW: EXPORT TO JSON FOR C++ COMPATIBILITY ---
    # C++ cannot read .pkl files easily, so we export the "Math" to JSON.
    model_data = {
        "n_clusters": int(n_groups),
        "centroids": kmeans.cluster_centers_.tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist()
    }
    
    with open('model_output/cluster_model.json', 'w') as f_json:
        json.dump(model_data, f_json, indent=4)
    
    print("\n--- REVOLUTION READY ---")
    print(f"Exported {n_groups} clusters to 'model_output/cluster_model.json'")
    print("Each group now has a 'Mathematical Center' that C++ can use to prune the search!")

    # Print summary of which files are in which group
    groups = {}
    for f, label in zip(valid_files, labels):
        if label not in groups: groups[label] = []
        groups[label].append(os.path.basename(f))
        
    for label in sorted(groups.keys()):
        print(f"Group {label} ({len(groups[label])} files): {', '.join(groups[label][:3])}...")

    # SAVE MAPPING FOR C++ (Which file belongs to which cluster)
    mapping = {os.path.basename(f): int(label) for f, label in zip(valid_files, labels)}
    with open('model_output/file_cluster_mapping.json', 'w') as f_map:
        json.dump(mapping, f_map, indent=4)

    # --- NEW: EXPORT HUMAN READABLE CLUSTER MAP ---
    with open('cluster_map.txt', 'w') as f_map_txt:
        f_map_txt.write("=== SIGNS SENSE: CLUSTER MAP ===\n")
        f_map_txt.write(f"Total Signs: {len(valid_files)} | Clusters: {n_groups}\n\n")
        for label in sorted(groups.keys()):
            f_map_txt.write(f"CLUSTER {label} ({len(groups[label])} members):\n")
            members = sorted(groups[label])
            for m in members:
                f_map_txt.write(f"  - {m}\n")
            f_map_txt.write("\n")
    
    print(f"Human-readable map saved to 'cluster_map.txt'")