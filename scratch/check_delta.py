import json
import math

def wrist_delta(path):
    data = json.load(open(path))
    dists = []
    for f in data:
        if 'hands' in f and len(f['hands']) >= 2:
            try:
                x1 = f['hands'][0]['wrist_pos']['x']
                y1 = f['hands'][0]['wrist_pos']['y']
                z1 = f['hands'][0]['wrist_pos']['z']
                x2 = f['hands'][1]['wrist_pos']['x']
                y2 = f['hands'][1]['wrist_pos']['y']
                z2 = f['hands'][1]['wrist_pos']['z']
                dists.append(math.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2))
            except:
                pass
    return max(dists) - min(dists) if dists else 0

print('Name Wrist Delta:', wrist_delta('templates/movement/2_hands/name.json'))
print('Weight Wrist Delta:', wrist_delta('templates/movement/2_hands/weight.json'))
