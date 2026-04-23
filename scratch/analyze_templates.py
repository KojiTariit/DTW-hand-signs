import json
import math
import os

def get_dist(p1, p2):
    return math.sqrt(sum((p1[k]-p2[k])**2 for k in ['x', 'y', 'z']))

def analyze(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
    print(f"\n--- Analyzing {os.path.basename(path)} ---")
    with open(path, 'r') as f:
        data = json.load(f)
    print(f"Frames: {len(data)}")
    
    velocities = []
    hand_variances = []
    
    for i in range(1, len(data)):
        if data[i]['hands'] and data[i-1]['hands']:
            # Wrist movement
            v = get_dist(data[i]['hands'][0]['wrist_pos'], data[i-1]['hands'][0]['wrist_pos'])
            velocities.append(v)
            
            # Finger movement (Internal variance)
            lms1 = data[i]['hands'][0]['landmarks']
            lms0 = data[i-1]['hands'][0]['landmarks']
            var = sum(get_dist(lms1[j], lms0[j]) for j in range(len(lms1))) / len(lms1)
            hand_variances.append(var)
    
    if velocities:
        print(f"Wrist Max Velocity: {max(velocities):.6f}")
        print(f"Wrist Avg Velocity: {sum(velocities)/len(velocities):.6f}")
        print(f"Hand Max Variance: {max(hand_variances):.6f}")
        print(f"Hand Avg Variance: {sum(hand_variances)/len(hand_variances):.6f}")

analyze('c:/Users/USER/Desktop/DTW/templates/movement/2_hands/name.json')
analyze('c:/Users/USER/Desktop/DTW/templates/movement/2_hands/weight.json')
