# Signs Sense: Project Core Manifesto & Architecture Report

## Purpose of This Project
The goal of this project is to build a high-performance Hybrid Sign Language Recognition System utilizing a C++ UDP engine and a Python (MediaPipe) frontend. It solves the performance and accuracy challenges of processing complex sign language by splitting the pipeline: Python handles the heavy vision/tracking logic in real-time, while an ultra-fast C++ backend handles the categorization and routing. 

The most critical feature is its ability to instantly route a sequence into either a Machine Learning Model (for static signs) or a Dynamic Time Warping algorithm (for movement signs) depending on the temporal profile of the captured data.

---

## Phase 1: Facial Anchors for Spatial Context
*   **The Problem:** Many signs share identical finger formations and only differ by where they are placed relative to the body. For instance, the signs for "Father" and "Grandpa" have very similar hand patterns but are anchored to the upper face (forehead), requiring facial awareness to classify accurately.
*   **Performance Impact:** Running the MediaPipe Holistic model to generate 478 face mesh points doesn't severely damage local Python FPS if complexity is set to `0`, but the real bottleneck is **network latency**. Serializing and transmitting an extra 478 3D coordinates (x, y, z) per frame over UDP to the C++ receiver would severely bloat the payload and cause massive lag, neutralizing the speed advantage of using C++ in the first place.
*   **The Solution (4-Point Anchor System):** Instead of dropping face detection or causing bandwidth lag, the C++ engine will only receive the data it explicitly needs to build spatial awareness. We will extract exactly **4 anchor points** from the facial landmark array:
    1.  **Forehead**
    2.  **Chin**
    3.  **Left Cheek**
    4.  **Right Cheek**
    
    This creates a lightweight "oval bounding box" that adds virtually zero UDP overhead, while giving the DTW/ML engines the exact relative context needed to tell if a hand is near the chin versus the forehead.

---

## Phase 2: The UDP Bridge Interface
*   **Current State:** The bridge acts as a highly optimized, uncoupled stream via `127.0.0.1:5005`.
*   **The Mechanism:** `scrap_capture.py` acts as the thin client, grabbing skeletal data at 60 FPS, turning it into lightweight JSON, and firing it.
*   In `ml_project/hybrid_receiver.cpp`, the listener handles two primary JSON signals:
    *   `"FRAME"`: Silently builds up a continuous sign array `current_sign_buffer`.
    *   `"END_OF_SIGN"`: Received when recording is toggled off. This triggers the decision tree instantly. By separating capture software from calculation software, the live video feed never freezes while the AI thinks.

---

## Phase 3: Hybrid Routing & Classification Engine (The Decision Tree)
The core of the system is the **Hierarchical Tree** logic that prunes possibilities to maintain speed. Currently, it leverages the topmost node of this tree: separating Static and Dynamic data.

### 1. The Static Path (Machine Learning)
*   **Trigger Condition:** If the hand is physically stationary (`max_wrist_dist` and `max_shape_variance` are below thresholds).
*   **Engine Used:** Random Forest Machine Learning (`StaticSignClassifier`).
*   **Validation:** Yes, the AI confirms this is the **ideal approach**. Doing DTW on static alphabet letters is messy because time doesn't matter for a static shape. Machine Learning excels at taking isolated snapshots of joint angles/distances and generalizing them to classes very fast.

### 2. The Dynamic Path (DTW + DDTW)
*   **Trigger Condition:** A dynamic sign is registered if there is major wrist movement, finger fluctuation (like folding fingers for 'J'), or if two hands are detected.
*   **Engine Used:** Custom C++ Dual-Score Dynamic Time Warping.
*   **Validation:** Using deep sequence learning (LSTMs or Transformers) requires massive datasets and extreme computing power. DTW is fantastic here because it is a "One-Shot" / template-based approach. If you have 1 good template, it works.
*   **Dual-Score:** 
    *   **Exact DTW:** Maps the geometric shape / path of the movement.
    *   **Derivative DTW (DDTW):** Analyzes the velocity and derivatives. This effectively maps the *rhythm* to differentiate if a sign is fast, slow, steady, or shaking.

### 3. The "Scale-Invariant" Feature Space (The Secret Sauce)
The biggest learning from the hybrid receiver is the transition away from raw 3D coordinates. To feed the Machine Learning model accurately, a 44-feature array is extracted from a single frame. This feature space is highly engineered to be "scale-invariant":
1.  **Wrist-Relative Distances (20 features):** Distances of all joints from the wrist, *normalized by the hand's overall size*.
2.  **Finger Extension Ratios (5 features):** Detects curling by comparing tip-to-knuckle vs knuckle-to-wrist distances.
3.  **Fingertip Spreads (4 features):** The distance between adjacent fingertips (differentiates flat shapes like 'U' vs 'V').
4.  **Joint Angles (15 features):** Dot products between finger segments.

Because these features use ratios and angles instead of absolute X, Y, Z pixel values, a child's hand 5 feet away from the camera looks mathematically identical to an adult's hand 1 foot away. This is why the Static Path is so robust.

### Future Addition: Deep Hierarchical Pruning
As the database scales for hundreds of signs, we will expand this tree. Before executing the expensive DTW matching on 100 templates, the C++ engine will execute cheap heuristic checks:
1.  **Hand Count Check:** (Skip 2-hand signs if only 1 hand is present).
2.  **Base State Check:** (Is the pinky up? Are they fists?).
With this tree mapped out, the heavy DTW engine will only ever have to compare the live feed against 3 to 5 candidate templates dynamically selected from their specific folders, preserving lightning-fast speeds.
