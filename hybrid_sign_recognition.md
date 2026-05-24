# 🏆 Hybrid Sign Language Recognition: The Final Technical Compendium

This is the absolute technical record of the project. It leaves zero details to the imagination. This is the blueprint for everything from the raw Python capture to the final C++ fusion logic.

---

## 🛰️ Phase 1: Python Capture (Skeletal Gen-2.5)
- **Tracking Engine**: `MediaPipe Sniper-Core`.
- **Coordinate System**: Normalized `(0.0 to 1.0)` based on frame resolution.
- **Normalization Units**:
    - **Face Unit**: Distance from Forehead to Chin.
    - **Hand Unit**: Distance from Wrist to the base of the Middle Finger.
- **Data Transport**: UDP Port `5005`. Data is serialized via `nlohmann/json` to maintain schema consistency between Python and C++.

---

## 📊 Phase 2: The 120-Feature Data DNA (The Blueprint)
Every single frame is converted into a 120-dimension vector. Here is the **exact index-by-index map** used by the C++ engine:

| Index Range | Feature Category | Description | Weight |
| :--- | :--- | :--- | :--- |
| **[0 - 59]** | **Skeletal XYZ** | 20 joint positions relative to the wrist. | 2.0x |
| **[60 - 64]** | **Finger Curls** | Extension ratios for Thumb, Index, Middle, Ring, Pinky. | 2.0x |
| **[65 - 79]** | **Joint Angles** | Cosines of the 3 joints for each of the 5 fingers. | 2.0x |
| **[80 - 86]** | **Index-to-Face** | Index tip distance to 7 Face Anchors. | **4.0x** |
| **[87 - 93]** | **Middle-to-Face** | Middle tip distance to 7 Face Anchors. | **4.0x** |
| **[94 - 100]** | **Thumb-to-Face** | Thumb tip distance to 7 Face Anchors. | **4.0x** |
| **[101]** | **Wrist-to-Head** | Distance from wrist to Forehead anchor. | **4.0x** |
| **[102]** | **Wrist-to-Chin** | Distance from wrist to Chin anchor. | **4.0x** |
| **[103 - 112]** | **Tip Matrix** | Distance between every fingertip pair (e.g. 4 to 8, 4 to 12). | 1.0x |
| **[113 - 116]** | **Thumb-Cross** | Distance from Thumb Tip to all other finger bases. | 1.0x |
| **[117 - 119]** | **Palm Normal** | The XYZ vector showing where the palm faces. | **3.0x** |

---

## 🧠 Phase 3: The AI Models (Random Forest)
The system uses two separate AI models exported as C++ logic:
- **Dynamic Model**: 80 Trees, Max Depth 10. Trained on every individual frame of your `movement/` recordings.
- **Static Model**: 100 Trees, Max Depth 12. Trained on alphabet and static gesture folders.
- **The Shortlist**: The Dynamic Model generates a confidence list. Only the top-scoring candidates are allowed into the DTW Path Judge to save CPU.

---

## ⚡ Phase 4: Physics & Rhythm (DDTW)
- **Derivative DTW**: The system calculates the velocity of all 120 features.
- **Velocity Scaling**: Differences in speed are multiplied by **8.0x**. This ensures that "Mom" (tapping) and "Eat" (moving toward) are separated by their rhythm.
- **Fusion Alpha (0.5f)**: The final path score is a 50/50 balance between **Shape Matching** (DTW) and **Rhythm Matching** (DDTW).

---

## ⚖️ Phase 5: The Final Fusion Formula
Every template in the database receives a "Total Penalty." **The template with the lowest penalty wins.**

**`Score = (0.5 * Shape_Dist) + (0.5 * Rhythm_Dist) - ML_Bonus - Cluster_Bonus + Duration_Penalty`**

- **ML_Bonus (-10.0)**: Subtracted if the AI model recognizes the hand shape.
- **Cluster_Bonus (-5.0)**: Subtracted if the sign belongs to a "Golden Template" group.
- **Duration Penalty**: Up to **1.5 points** added if the sign's length is significantly different from the template.

---

## 🛡️ Phase 6: System Safeguards
1.  **First-Frame Guard**: The system sets the "Reference Point" only when a hand is detected at a confidence > 0.5.
2.  **Hand-Count Pruner**: If the system detects two hands but the leaderboard picks a one-hand sign, it forces a re-evaluation of only two-hand templates.
3.  **Sakoe-Chiba Constraint**: Limits "Time Warping" to **15%** of the sign length.

---

## 🛠️ Phase 7: Compilation & Environment
- **Command**: `g++ -O2 scrap_receiver.cpp -o scrap_receiver.exe -lws2_32 -Wl,--stack,16777216`
- **Memory**: 16MB reserved for the AI's "If-Then" tree structures.
- **Threading**: Single-threaded, synchronous inference loop for ultra-low latency.

---

## 📈 Stage: Validation
We are currently validating the **Mouth vs. Forehead** spatial logic to ensure the 4.0x weights are correctly separating signs at different heights on the face.
