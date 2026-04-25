import json
import math
import os

def get_dist(p1, p2):
    return math.sqrt(sum((p1[k]-p2[k])**2 for k in ['x', 'y', 'z']))

def analyze_inter_hand(path):
    if not os.path.exists(path):
        return
    print(f"\n--- Analyzing Inter-Hand Distance: {os.path.basename(path)} ---")
    with open(path, 'r') as f:
        data = json.load(f)
    
    wrist_dists = []
    tip_dists = [] # Index tip (8) of hand 0 to Index tip (8) of hand 1
    
    for frame in data:
        if len(frame['hands']) >= 2:
            h0 = frame['hands'][0]
            h1 = frame['hands'][1]
            
            # Wrist distance
            w0 = h0['wrist_pos']
            w1 = h1['wrist_pos']
            wd = get_dist(w0, w1)
            wrist_dists.append(wd)
            
            # Tip distance
            # Absolute position = wrist_pos + relative_landmark
            t0 = {'x': w0['x'] + h0['landmarks'][8]['x'], 'y': w0['y'] + h0['landmarks'][8]['y'], 'z': w0['z'] + h0['landmarks'][8]['z']}
            t1 = {'x': w1['x'] + h1['landmarks'][8]['x'], 'y': w1['y'] + h1['landmarks'][8]['y'], 'z': w1['z'] + h1['landmarks'][8]['z']}
            td = get_dist(t0, t1)
            tip_dists.append(td)
            
    if wrist_dists:
        print(f"Wrist Dist - Min: {min(wrist_dists):.4f}, Max: {max(wrist_dists):.4f}, Delta: {max(wrist_dists) - min(wrist_dists):.4f}")
        print(f"Tip Dist   - Min: {min(tip_dists):.4f}, Max: {max(tip_dists):.4f}, Delta: {max(tip_dists) - min(tip_dists):.4f}")

analyze_inter_hand('c:/Users/USER/Desktop/DTW/templates/movement/2_hands/name.json')
analyze_inter_hand('c:/Users/USER/Desktop/DTW/templates/movement/2_hands/weight.json')
