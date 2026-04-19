# Signs Sense (Hybrid Sign System Architecture)

**Version:** 2.0 (Stabilized)  
**Objective:** Maintain 100% distinction between Static Alphabet and Dynamic Movement signs.

---

## 1. The Core Routing State Machine
The system must decide which "Brain" to use based on the input buffer. 

| Sign Type | Trigger Rule | Engine Used | Path |
| :--- | :--- | :--- | :--- |
| **STATIC (Alpha)** | `max_wrist_dist < 0.15` AND `max_shape_variance < 0.05` | **Random Forest (ML)** | `StaticSignClassifier.hpp` |
| **DYNAMIC (Word)** | `max_wrist_dist >= 0.15` OR `max_shape_variance >= 0.05` OR `Hands >= 2` | **DTW + DDTW Engine** | `DtwEngine.hpp` |

> [!IMPORTANT]
> **Rule 0:** You MUST check "Shape Variance." Signs like **'J'** or **'Z'** may have zero wrist movement but high shape variance. If you only check the wrist, the routing will fail.

---

## 2. Strict Category Pruning (Memory Structure)
To prevent interference (e.g., 'Y' being confused with 'A'), the system follows **Strict Folder Isolation.**

*   **Rule 1 (The Dynamic Room):** If the routing is **DYNAMIC**, the DTW Engine is FORBIDDEN from looking at the alphabet. It must only calculate scores against templates in the `templates/movement/` folder.
*   **Rule 2 (The Static Room):** If the routing is **STATIC**, it must ONLY use the ML Model. It is FORBIDDEN from guessing a movement sign.

---

## 3. The Dual-Score Engine (DTW + DDTW)
For movement signs, we use a hybrid score to distinguish rhythm.

*   **Shape Score (DTW):** Measures the absolute position/angles (e.g., the hand location).
*   **Rhythm Score (DDTW):** Measures the **Velocity/Derivative** (e.g., is the hand wiggling fast or moving slow?).

> [!CAUTION]
> **'Name' vs 'Weight' Separation:** 
> - **'Name'** has a "Tap-Tap" velocity profile.
> - **'Weight'** has a "Steady-Shake" velocity profile.
> If they are confused, you must increase the **alpha weight** for the Rhythm Score (DDTW) to prioritize "the feel" over "the shape."

---

## 4. Feature Extraction Contract
The C++ `extract_ml_features` and Python `train_ml.py` MUST stay synchronized.

- **Angle Features:** Cosine similarity of finger joints.
- **Radial Features:** Distance from finger tips to wrist.
- **Normalization:** All landmarks must be normalized by the `hand_size` (distance between wrist-0 and knuckle-9) to remain scale-invariant.

---

## 5. Metadata & Pruning Map
The system expects the following folder structure:
```text
templates/
  ├── static/     <-- ML Model training data (A-Z)
  └── movement/   <-- DTW Template data (Name, Weight, J, etc.)
```
**Pruning Logic:** `SignDatabase` must categorize templates by their parent folder name.
