import json
import socket
import time
import os

def simulate_sign(json_path):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_address = ('127.0.0.1', 5005)

    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return

    with open(json_path, 'r') as f:
        frames = json.load(f)

    print(f"Simulating {len(frames)} frames from {json_path}...")
    
    for frame in frames:
        # Wrap old format into new skeletal format if necessary
        payload = {
            "type": "FRAME",
            "timestamp": frame.get("timestamp", time.time()),
            "hands": []
        }
        
        if "hands" in frame:
            payload["hands"] = frame["hands"]
        elif "landmarks" in frame:
            payload["hands"].append({
                "landmarks": frame["landmarks"],
                "wrist_pos": frame["wrist_pos"]
            })

        sock.sendto(json.dumps(payload).encode('utf-8'), server_address)
        time.sleep(0.02) # Simulate 50 FPS

    # Send END_OF_SIGN
    sock.sendto(json.dumps({"type": "END_OF_SIGN"}).encode('utf-8'), server_address)
    print("Sign complete. Check the C++ terminal!")

if __name__ == "__main__":
    # Test with 'Name'
    simulate_sign(r"c:/Users/USER/Desktop/DTW/templates/movement/name.json")
