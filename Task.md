# Task List - Signs Sense (DTW Sign Translator)

- [x] **Planning Phase**
    - [x] Initial Architecture Review
    - [x] Define IPC strategy (Python -> C++)

- [x] **Dataset Quality Audit**
    - [x] Verify structural integrity of all JSON templates
    - [x] Analyze landmark variance and jitter
    - [x] Compare inter-class similarity and identify confusion points
    - [x] Inspect movement trajectories for 'J' and 'Z'
    - [x] Validate normalization and centering consistency

- [x] **Pre-Step: MediaPipe & IPC**
    - [x] Setup MediaPipe Python streaming
- [x] **Recognition Engine Tuning**
    - [x] Designing improved feature extraction (Landmark Weighting)
    - [x] Implement Point-to-Point distance features
    - [x] Optimize similarity thresholding
    - [x] Test refined engine against existing dataset
    
- [ ] **Hybrid Architecture: Static ML Classifier**
    - [x] Create Python script to parse `templates/static` into tabular features (angles/distances).
    - [x] Train a lightweight Random Forest/SVM classifier on the static dataset.
    - [x] Export the trained model to C++ (via ONNX, a simple C++ header, or pure math).
    - [x] Update `scrap_receiver.cpp` to route static signs to the ML model and movement to DTW.
    - [x] Implement JSON parsing in C++ (receiver.cpp)
    - [x] Design Hierarchical Folder Structure (SECTIONING_TREE.md)
    - [x] **Sign Database**: Create `SignDatabase.hpp` to scan folders and load all templates at startup.
    - [x] Implement Wrist Displacement pruning (Static vs Movement folders).
    - [x] **Milestone 3**: Verify pruning works for all newly added static alphabet signs.
    - [x] **Deep Dataset Improvement**: Re-record problem signs (G, X, P, Q, M, N, T) 3 times for robustness.

- [ ] **Core Engine Development (Main Quest)**
    - [ ] **Step 2**: Derivative DTW (DDTW) Implementation (Prioritized)
    - [x] Implement DDTW in C++ (DtwEngine.hpp)
    - [x] **Milestone 2**: Static vs Movement - Verified 'I' vs 'J' successfully in live test!
    - [ ] **Step 3**: Exact DTW + Sakoe-Chiba Implementation

- [ ] **Sub-Quest: Sign Translator Milestones**
    - [x] **Quest A**: The "I" vs "J" Challenge (Static vs Movement)
    - [ ] **Quest B**: The "Name" vs "Weight" Challenge (Tapping vs Wiggling)
    - [ ] **Quest C**: The "Ultimate Pruning" Test (Filtering unrelated signs)

- [ ] **Sub-Quest: Continuous Fingerspelling Words**
    - [ ] Set up a separate test script to capture continuous alphabet classification.
    - [ ] Implement logic to concatenate letters into words without relying on time variables.
    - [ ] Test the fingerspelling sequence.

- [ ] **Sign Template Recorder**
    - [x] Create `recorder.py` to save reference JSON templates.
    - [/] Record "Golden" templates for 'I', 'J', 'Name', 'Weight'.
