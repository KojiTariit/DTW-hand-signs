# How to Read the Signs Sense Data

Now that your C++ "Brain" is receiving text from your Python "Eyes," you need to know how to turn that text into actual numbers you can use for the 3-step classification.

## 1. The Data Structure (JSON)
Each packet sent by `signs_sense_capture.py` looks like this:

```json
{
  "label": "Right",               // "Left" or "Right" hand
  "wrist_pos": {"x": 0.5, "y": 0.8, "z": 0.0}, // Absolute screen position
  "landmarks": [                  // 21 points relative to the wrist
      {"x": 0.0, "y": 0.0, "z": 0.0}, // [0] Wrist (always 0,0,0)
      {"x": 0.1, "y": -0.2, "z": 0.0}, // [1] Thumb start...
      ...
  ],
  "hand_index": 0,                // 0 for first hand, 1 for second
  "timestamp": 17118...           // Time of capture
}
```

## 2. Landmarks Map (The Fingers)
MediaPipe numbers the 21 landmarks like this. You will use these indices to detect "Hand Shapes":

*   **Wrist**: 0
*   **Thumb**: 1 (base), 2, 3, 4 (tip)
*   **Index**: 5 (base), 6, 7, 8 (tip)
*   **Middle**: 9 (base), 10, 11, 12 (tip)
*   **Ring**: 13 (base), 14, 15, 16 (tip)
*   **Pinky**: 17 (base), 18, 19, 20 (tip)

---

## 3. How to Use it in C++ (JSON Parsing)
To read this easily in C++, I highly recommend using the **nlohmann/json** library. It is "header-only," which means you just need one file to make it work.

### Step A: Get the Library
1.  Go to the [nlohmann/json GitHub Releases](https://github.com/nlohmann/json/releases).
2.  Download the file named **`json.hpp`**.
3.  Put it in your `DTW` folder.

### Step B: The Parsing Code
Once you have `json.hpp`, you can update your C++ receiver to look like this:

```cpp
#include "json.hpp" // Now you can use this!
using json = nlohmann::json;

// Inside your main loop after receiving 'message':
json data = json::parse(message);

std::string label = data["label"];
float index_tip_y = data["landmarks"][8]["y"];

if (index_tip_y < -0.5) {
    std::cout << "Index finger is pointing UP!" << std::endl;
}
```

## 4. Why use "Relative" coordinates?
Because we subtracted the wrist position in Python, the `landmarks` array is **translation-invariant**. 
*   If you move your hand to the top-left of the camera, the `landmarks` stay the same.
*   Only the `wrist_pos` changes.
This makes your Step 1 (Heuristic) and Step 3 (DTW) much easier because they only care about the **SHAPE** of the hand, not where it is on the screen.

---

### What's next?
Next, we will implement the **Heuristic Step**. We will check things like:
- "Is the index finger higher than the palm?"
- "Are the thumb and index finger touching?"
- "Are two hands stacked?" (By comparing the two `wrist_pos` values).
