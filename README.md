# Signs Sense: Hybrid Hand-Sign Recognition Engine

A high-performance hand-sign recognition system combining **Dynamic Time Warping (DTW)** for movement-based signs and **Machine Learning (Random Forest)** for static alphabet signs.

## Features
- **Hybrid Routing**: Automatically distinguishes between static poses (A, B, C...) and dynamic movements (HELLO, J, Z).
- **Face Context awareness**: Reconstructs absolute hand positions relative to facial landmarks (forehead, chin) for superior spatial accuracy.
- **Precision Snipper Filters**: Geometric sanity checks to prevent false positives for similar signs (e.g., U vs V spread).
- **C++ Inference Engine**: Ultra-fast predictions with a transpiled ML brain.

## File Structure
- `upgraded_receiver.cpp`: The main C++ inference engine.
- `DtwEngine.hpp`: DTW logic for dynamic signs.
- `StaticSignClassifier.hpp`: Auto-generated ML brain (58-feature Sniper Core).
- `ml_project/`: Python scripts for training and exporting the model.
- `templates/`: High-quality skeletal recording data in JSON format.
- `recorder.py`: Tool for recording new sign templates.

## How to Run
1. Start the C++ receiver: `.\upgraded_receiver.exe`
2. Start the Python capture: `python scrap_capture.py`
3. Update the engine after recording new signs with `update_engine.bat`.

---
*Created with Advanced Agentic Coding for high reliability and speed.*
