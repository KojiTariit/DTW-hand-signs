import socket
import json
import time

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    with open("templates_backup/static/single_hand/R_2.json", "r") as f:
        frames = json.load(f)
        
    for frame in frames:
        packet = json.dumps({
            "type": "FRAME",
            "hands": frame.get("hands", []),
            "face": frame.get("face"),
            "pose_anchors": frame.get("pose_anchors")
        }).encode('utf-8')
        sock.sendto(packet, ("127.0.0.1", 5005))
        time.sleep(0.01)
        
    sock.sendto(json.dumps({"type": "SIGN_COMPLETE"}).encode('utf-8'), ("127.0.0.1", 5005))
    print("Test packet sent.")

if __name__ == "__main__":
    main()
