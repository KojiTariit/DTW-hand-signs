# Hybrid Sign Recognition System - Project State & Architecture

This document serves as a comprehensive handover for any future developer or AI agent working on this codebase. It details the architecture, feature pipelines, and critical gotchas that keep the system stable.

## 1. System Architecture
The system is a **Hybrid Inference Engine** split between Python (Capture/Training) and C++ (Real-Time Inference).
*   **Dynamic Signs (Movement):** Handled by a C++ Dynamic Time Warping (DTW) processor.
*   **Static Signs (Alphabet):** Handled by a 300-tree Random Forest Machine Learning model converted directly into C++ header code.

### Core Files
*   `scrap_capture.py`: Uses Mediapipe to extract 3D landmarks (Hands, Face, Pose). Transmits data via UDP (Port 5005) to the C++ receiver.
*   `scrap_receiver.cpp`: The core C++ engine. Listens for UDP packets, groups frames, and routes to DTW or ML based on hand movement profiles.
*   `ml_project/train_ml.py`: Python script that reads `.json` templates, extracts a 142-length feature vector, augments the data, and trains the Random Forest model.
*   `m2cgen_export.py`: Converts the `static_ml_model.pkl` into a massive `StaticSignClassifier.hpp` file for the C++ engine to compile.

---

## 2. The 142-Feature Extraction Pipeline
Both Python (`train_ml.py`) and C++ (`scrap_receiver.cpp`) must PERFECTLY mirror this feature extraction logic. If they mismatch, the system will crash or classify garbage.

| Feature Segment | Count | Purpose |
| :--- | :---: | :--- |
| **Wrist-Relative XYZ** | 60 | Baseline geometric shape (Indices 0-59). |
| **Finger Extension (Curl)** | 5 | Detects bent vs straight fingers (Indices 60-64). |
| **Joint Angles** | 15 | Captures finger spread (Indices 65-79). |
| **Face Context Probes** | 23 | Distance from hand to face anchors, normalized by Face Height (Indices 80-102). |
| **Full Tip Matrix** | 10 | Distance between all fingertips. Solves **U vs V** spacing (Indices 103-112). |
| **Thumb-Cross Matrix** | 4 | Thumb distance to PIP joints. Solves **A vs S** wrapping (Indices 113-116). |
| **Palm Orientation** | 3 | Normal vector of the palm. Solves Palm Up/Down (Indices 117-119). |
| **Orientation Highlighters** | 2 | `dx / (dx + dy)` for Index and Middle. Explicitly defines Vertical vs Horizontal (Indices 120-121). |
| **Feature Boosting** | 20 | Duplicates the Orientation Highlighters 10x to force the Random Forest to prioritize them (Indices 122-141). |

---

## 3. Critical Fixes & Stability Gotchas (DO NOT REVERT)

### A. The "H" vs "R" Crosstalk (Spatial Guard)
The ML model sometimes struggles to distinguish between "H" (horizontal) and "R" (vertical) purely from geometric features. 
**Solution:** A "Spatial Guard" is implemented in `scrap_receiver.cpp`. Before accepting a prediction, it reads the Orientation Highlighters at `ml_features[120]` and `ml_features[121]`. 
*   If Angle > 0.65 (Sideways) and the model predicted R/U/V/I -> **Force H**.
*   If Angle < 0.35 (Vertical) and the model predicted H -> **Force U**.

### B. The Stack Overflow Crash (16MB Flag)
The "Max Power" Random Forest model utilizes 300 estimators and `max_depth=None`. When converted to C++, this generates a 41,000+ line `score()` function with thousands of local array variables.
**The Problem:** Running this immediately exhausts the default 1MB Windows stack memory, resulting in a silent Crash (Exit Code 1).
**The Fix:** You **MUST** compile the C++ engine with the `-Wl,--stack,16777216` flag to provide 16MB of stack space.
*   *Compile Command:* `g++ scrap_receiver.cpp -o scrap_receiver.exe -lws2_32 -O3 -Wl,--stack,16777216`

### C. The "Phantom Hand" Crash (Uninitialized Memory)
If the C++ buffer grabs a frame where the left hand is missing, it must not feed an empty or partial vector to the ML model.
**The Fix:** `extract_ml_features` actively scans `f.hands` for the first valid hand (`h.is_present && !h.landmarks.empty()`). 
Furthermore, `scrap_receiver.cpp` strictly checks `if (ml_features.size() == 142)`. If it is not exactly 142, it aborts the prediction and returns `NONE`. Failure to do this will cause the AI to evaluate uninitialized memory (Signaling NaNs) and crash the system.

### D. Data Augmentation in Python
To prevent the AI from overfitting to perfect templates, `train_ml.py`:
1.  **Auto-Trims:** Uses only the middle 60% of frames (`start_idx:end_idx`) to ignore the chaotic start/end of hand movements.
2.  **Jittering:** Injects artificial noise (`np.random.normal(0, 0.002)`) to duplicate and distort the data, making the final C++ engine highly resilient to webcam shaking or minor tracking errors.

### E. The API Key / Model Quota Errors (Dynamic Model Verification)
API keys on the free tier or under specific restrictions often return HTTP 429 (Quota Exceeded) or HTTP 403/503 for certain models. Hardcoding models leads to total translation failure.
**Solution:** Active model verification and auto-discovery.
*   **How it works:** Both the python script ([ai_polisher.py](file:///c:/Users/USER/Desktop/DTW/ai_polisher.py)) and the frontend app ([camera.js](file:///c:/Users/USER/Desktop/DTW/frontend_app/src/camera.js)) query the models list, concurrently test all candidates with a test payload, rank the successful models (`gemini-3.5-flash` > `3.1` > `2.5` > `2.0` > `1.5` > other), and cache the selection.
*   **Stale Invalidation:** In [camera.js](file:///c:/Users/USER/Desktop/DTW/frontend_app/src/camera.js), the cached model is automatically deleted when the user changes their API key or if a translation call fails, forcing a re-check.

