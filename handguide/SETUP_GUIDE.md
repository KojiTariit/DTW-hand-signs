# Signs Sense - Setup & Run Guide

This guide explains the two main "Modes" of your project: **Recording Mode** (to build your template dataset) and **Live Translation Mode** (to actually translate signs).

---

## 🏗️ Mode 1: Recording Mode (Building Your Dataset)
Before the C++ "Brain" can translate a sign, it needs an example (a JSON template) to compare against.

**Which file to run:** `recorder.py`
**How to run it (in PowerShell):**
```powershell
& "c:\Users\USER\Desktop\DTW\venv\Scripts\python.exe" "c:\Users\USER\Desktop\DTW\recorder.py"
```

**How to use it:**
1. Your webcam opens. Get your hand into the starting position.
2. Press **'R'** on your keyboard to start recording.
3. Perform the sign perfectly (e.g., hold 'I' still, or draw a hook for 'J').
4. Press **'R'** to stop recording.
5. Press **'S'** to save.
6. The terminal will ask you for a **Category** (e.g., `static`, `movement`, `stacking`). Type it and press Enter.
7. The terminal will ask for a **Sign Name** (e.g., `I` or `J`). Type it and press Enter.

*Note: The script will automatically create the folders and save the file to `DTW/templates/[Category]/[SignName].json`.*

---

## 🚀 Mode 2: Live Translation Mode
Once you have your templates saved, you can run the actual real-time translation pipeline. This involves running the Python "Eyes" and C++ "Brain" at the same time.

### Step 1: Start the "Brain" (C++)
The C++ program must be running first so it can listen for data.
*If you haven't compiled it yet, run this first:*
```powershell
g++ c:\Users\USER\Desktop\DTW\receiver.cpp -o c:\Users\USER\Desktop\DTW\receiver.exe -lws2_32
```
**How to run it:**
```powershell
& "c:\Users\USER\Desktop\DTW\receiver.exe"
```
*(Leave this terminal window open!)*

### Step 2: Start the "Eyes" (Python Capture)
Open a **new** terminal tab or window.
**Which file to run:** `signs_sense_capture.py`
**How to run it:**
```powershell
& "c:\Users\USER\Desktop\DTW\venv\Scripts\python.exe" "c:\Users\USER\Desktop\DTW\signs_sense_capture.py"
```

**How it works:**
The Python script will extract your hand data and stream it via UDP directly to the `receiver.exe` window, where the C++ DTW Engine will classify it against the templates you recorded.
