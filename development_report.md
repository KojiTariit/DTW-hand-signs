# Development Report: Sign Language Translation Mobile App Enhancements
**Date:** May 24, 2026

This report provides a detailed record of the enhancements, bug fixes, and feature integrations made to the Sign Language Translation Mobile App system today.

---

## 1. System Architecture Alignment & Routing Correction

### Background & Issue
The browser-side JavaScript translation engine was showing lower recognition accuracy compared to the local C++ system (`scrap_receiver.cpp`). We analyzed the codebase and found a routing logic mismatch:
* In `camera.js`, static signs with two hands (like the letter "W" or "B") were being forced into the dynamic DTW engine because of a `maxHands >= 2` override.
* The C++ engine routed purely based on frame length and wrist/hand shape variance.

### Resolution
* Modified [camera.js](file:///c:/Users/USER/Desktop/DTW/frontend_app/src/camera.js) to align its routing logic with [scrap_receiver.cpp](file:///c:/Users/USER/Desktop/DTW/scrap_receiver.cpp).
* Removed the `|| maxHands >= 2` override from the `isDynamic` check. Now, static two-handed signs route correctly to the Random Forest classifier rather than the dynamic DTW processor.

---

## 2. Camera Input Mirroring Alignment

### Background & Issue
Coordinates extracted from the browser's webcam were mirrored compared to the training templates.
* In Python's `scrap_capture.py` (which recorded the templates), frames were flipped horizontally using `cv2.flip(image, 1)` *before* being processed by MediaPipe Holistic.
* In JavaScript's `camera.js`, raw unflipped video frames were sent to MediaPipe, causing the $X$-coordinates to be inverted (`1.0 - x`) and left/right hands to be swapped.

### Resolution
* **Offscreen Canvas Pre-Flipping**: Added an offscreen canvas at 320x240 in the constructor of `CameraEngine`. In the `onFrame` callback, we draw the raw video frame mirror-flipped horizontally using 2D context transforms:
  ```javascript
  this.offscreenCtx.translate(320, 0);
  this.offscreenCtx.scale(-1, 1);
  this.offscreenCtx.drawImage(this.videoElement, 0, 0, 320, 240);
  ```
  We now send `this.offscreenCanvas` to MediaPipe instead of `this.videoElement`.
* **Skeletal Display Sync**: Since the coordinates returned by MediaPipe are now in the flipped coordinate space, drawing them directly matches the mirrored video display. We synchronized the mirror CSS class (`-scale-x-100`) between the video and overlay canvas so they are opposites. This keeps the skeletal drawing aligned perfectly in both mirrored and normal camera modes.

---

## 3. Dynamic Language Selection & Translation Settings

### Feature Implementation
Implemented a dual-tab Language Selection Modal for selecting input sign languages and target spoken languages.
* **Dual-Tab UI**: Users can toggle between "Sign Language" and "Spoken Language" inside the modal. The tabs transition with a smooth `border-color` and `color` change, and the lists fade in/out using a `150ms` opacity transition.
* **Search Filtering**: Added a real-time search box (`#lang-search`) that filters options instantly as the user types.
* **Sign Language (Source)**: Options include `ASL`, `TSL`, `CSL`, and `BSL`. These serve as dummy settings for now (the UI updates but the templates remain unchanged).
* **Spoken Language (Target)**: Options include `English`, `Thai`, `Chinese`, `Spanish`, and `Japanese`.
* **Dynamic Translation Prompt**: Modified the Gemini API request inside `translateSentence` in `camera.js`. If a target language other than English is selected, Gemini is instructed to:
  1. Construct a polished English sentence from the raw recognized words.
  2. Translate that polished English sentence into the selected target language.
  3. Output only the translated sentence.
* **State Persistence**: Language choices are stored in `localStorage` (`selectedSourceLang` / `selectedTargetLang`) and persist across sessions.

---

## 4. C++ Style Prediction Analysis Scoreboard

### Feature Implementation
Redesigned the Translation History Detail panel to output a professional, C++-style Candidate Prediction Analysis scoreboard.
* **Opaque AI Sentence Block**: Positioned at the top with a premium gradient background (`bg-gradient-to-r from-violet-600 to-indigo-600`) to emphasize the polished output.
* **Original Sentence Block**: Displays the raw sign sentence exactly as it was recognized (in quotes and italics, e.g. *“hello thank you eat”*) before NLP processing.
* **Scoreboard Analysis Cards**:
  * Shows a card for each word detailing the classification route (`Dynamic (DTW + RF)` vs `Static (Random Forest)`).
  * Lists the Top 3 lattice candidates with hand count tags (`[1H]` / `[2H]`) for dynamic routes.
  * Prints exact C++ scores: Fused scores and DTW distances for dynamic, or classification confidence percentage for static.
  * Highlights the selected winner candidate with a colored border and tinted background.

---

## 5. Settings Panel Upgrades & Toggles

### Feature Implementation
* **Theme Switcher**: Added a theme toggle supporting smooth light-to-dark and dark-to-light transitions.
  * Added transition properties in [style.css](file:///c:/Users/USER/Desktop/DTW/frontend_app/style.css) with `0.5s` easing.
  * Configured a dark-slate theme palette (`body.dark`) and custom brand color adjustments for dark mode.
* **Model Complexity Toggle**: Allowed toggling MediaPipe Holistic model complexity between **Low** (complexity `1`) and **High** (complexity `2`) to control performance on mobile devices.
* **State Persistence**: Both the theme mode (`themeMode`) and model complexity (`modelComplexity`) are saved to `localStorage`.

---

## 6. Security & Credential Management

### Issues Fixed
* Detected a hardcoded Gemini API key inside `ai_polisher.py` line 9.
* **Action Taken**: 
  1. Replaced the hardcoded key with an environment variable lookup:
     ```python
     GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
     ```
  2. Committed and pushed the removal immediately to GitHub to minimize exposure.
  3. Advised the user to delete/revoke the compromised key in their Google AI Studio console and create a new one.

---

## 7. Build and Verification Results
* Ran `npm run build` inside `frontend_app/`. The project compiled successfully into static assets:
  ```text
  dist/index.html                 17.53 kB │ gzip:  3.90 kB
  dist/assets/index-B5Sw_3Pb.css  28.20 kB │ gzip:  5.40 kB
  dist/assets/web-kLJixoZV.js      4.09 kB │ gzip:  1.53 kB
  dist/assets/index-4I35JOBv.js   53.47 kB │ gzip: 16.37 kB
  ✓ built in 2.06s
  ```
* Created the file `vite.config.js` to configure `server.allowedHosts = true`, enabling remote connections via tunnel providers like localtunnel without hostname blocking.
* Pushed all work to the remote repository `https://github.com/KojiTariit/DTW-hand-signs.git` and tagged it with release tag `V5`.
