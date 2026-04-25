import json
import math
import numpy as np

def load_template(path):
    with open(path, 'r') as f:
        return json.load(f)

def extract_features(sequence):
    features = []
    # Simplified extraction just to test face distances and trajectory
    # We will just print the max trajectory and face distances
    return sequence

print("Analysis ready. Please run C++ tester to get exact feature breakdown.")
