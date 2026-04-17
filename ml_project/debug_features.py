import json
import numpy as np
import glob

def extract(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    if not data or not data[0].get('hands'): return None
    
    lms = data[0]['hands'][0]['landmarks']
    pts = np.array([[lm['x'], lm['y'], lm['z']] for lm in lms])
    
    p0 = pts[0]
    hand_size = np.linalg.norm(pts[9] - p0)
    if hand_size < 1e-6: hand_size = 1.0

    ext = []
    # Extension: Tip to MCP normalized by MCP to Wrist
    tips = [4, 8, 12, 16, 20]
    mcps = [2, 5, 9, 13, 17]
    for t, m in zip(tips, mcps):
        tip_to_mcp = np.linalg.norm(pts[t] - pts[m])
        mcp_to_wrist = np.linalg.norm(pts[m] - p0)
        ext.append(tip_to_mcp / (mcp_to_wrist + 1e-6))
        
    # Spread: Adjacent Tips
    spr = []
    for i in range(4):
        spr.append(np.linalg.norm(pts[tips[i]] - pts[tips[i+1]]) / hand_size)
    
    return ext, spr

for letter in ['A', 'B', 'E', 'U', 'V', 'Y']:
    f = f"c:/Users/USER/Desktop/DTW/templates/Test_case/static/{letter}.json"
    res = extract(f)
    if res:
        ext, spr = res
        print(f"[{letter}]")
        print(f"  Extension (T, I, M, R, P): {[f'{x:.2f}' for x in ext]}")
        print(f"  Spread (T-I, I-M, M-R, R-P): {[f'{x:.2f}' for x in spr]}\n")
