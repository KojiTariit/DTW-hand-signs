# Signs Sense - Sectioning Logic (The Decision Tree)

To handle 500 signs without lagging, we use a **Hierarchical Pruning** system. Instead of the C++ engine looking at everything, it "filters" the signals through folders.

### 🌳 The "Ghost" Directory Tree

This is how we should organize the `templates/` folder. The C++ engine will only load the folder that matches your current hand-state.

```text
templates/
├── 1_hand/
│   ├── pinky_up/           <-- [Test A results here]
│   │   ├── I.json
│   │   └── J.json
│   ├── index_up/
│   │   ├── point.json
│   │   └── look.json
│   ├── fist/
│   │   ├── A.json
│   │   └── S.json
│   └── open_palm/
│       ├── 5.json
│       └── hello.json
│
└── 2_hands/
    ├── stacked/            <-- [Test B results here]
    │   ├── name.json
    │   └── weight.json
    ├── fingertips_touch/
    │   ├── more.json
    │   └── house.json
    └── mirrored/
        ├── family.json
        └── class.json
```

---

### 🧠 How Step 1 (The Heuristic) works in C++

When you start signing, the C++ code won't run DTW immediately. Instead, it runs these "Cheap" checks first:

1.  **Check 0: Hand Count**
    - If `multi_hand_landmarks.size() == 1` -> Enter `1_hand/`
    - If `multi_hand_landmarks.size() == 2` -> Enter `2_hands/`

2.  **Check 1: Finger State (The Big Filter)**
    - Is only the pinky extended? -> Load `pinky_up/`
    - Are the two wrists within 0.1 units of each other? -> Load `stacked/`

3.  **Check 2: The "Winner" Hunt**
    - Now that the list is tiny (maybe 2-5 signs), run **DDTW** (Step 2) and **Exact DTW** (Step 3).

### 💡 Why this is great for you:
As you add the rest of the alphabet, you just drop them into the right folder. 
- To add 'D', put it in `index_up`. 
- To add 'B', put it in `open_palm` (flat hand).

**What do you think of this folder structure? Does it match how you visualize the categories?**
