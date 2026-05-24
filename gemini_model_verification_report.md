# Developer Report: Dynamic Gemini Model Verification & Auto-Discovery

This document details the architecture, implementation, and verification steps for the dynamic Gemini Model Verification feature. It ensures that both the Python backend and the frontend application automatically query, verify, rank, and select the best responsive model authorized for the user's API key.

---

## 1. Problem Statement
Previously, the code was configured to use hardcoded models (e.g. `gemini-1.5-flash` in the frontend) or to select the first listed model without checking its actual responsiveness (in the Python backend). 

When API keys are rate-limited (HTTP 429), unauthorized (HTTP 403), or unavailable (HTTP 503) for specific models, the translation feature failed completely. 

---

## 2. Design & Solution Architecture
To solve this, we implemented an **active verification mechanism** that:
1.  **Queries the `/v1/models` endpoint** to list all models available for the provided API key.
2.  **Filters candidates** that explicitly support the `generateContent` method.
3.  **Concurrently checks every model's availability** by sending a lightweight `"hi"` post request.
4.  **Rank-orders successful models (HTTP 200)** based on performance/tier preferences:
    *   `gemini-3.5-flash` (Highest priority)
    *   `gemini-3.1-flash`
    *   `gemini-2.5-flash`
    *   `gemini-2.0-flash`
    *   `gemini-1.5-flash`
    *   Any other Flash-tier model
    *   Any other Gemini model
    *   Any other supported model (e.g., Gemma)
5.  **Caches the selected model** to minimize network overhead and latency.
6.  **Invalidates the cache** automatically if the API key is modified or if a request fails.

---

## 3. Code Implementation

### Python Backend Implementation
*   **Location**: [ai_polisher.py](file:///c:/Users/USER/Desktop/DTW/ai_polisher.py)
*   **Key Logic**:
    *   Uses Python's `concurrent.futures.ThreadPoolExecutor` to test all available models in parallel.
    *   Selects and outputs detailed developer logs to the command line interface (CLI).

```python
# Helper to test a single model
def check_model_availability(model_name):
    # Sends a fast post request to verify HTTP 200
    ...

# Executor logic in polish_with_ai:
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    results = executor.map(check_model_availability, available_models)
    for name, is_working, status_str in results:
        print(f"  - {name}: {status_str}")
        if is_working:
            working_models.append(name)
```

### Frontend Web App Implementation
*   **Location**: [camera.js](file:///c:/Users/USER/Desktop/DTW/frontend_app/src/camera.js)
*   **Key Logic**:
    *   Defines [getActiveGeminiModel(apiKey)](file:///c:/Users/USER/Desktop/DTW/frontend_app/src/camera.js#L762) inside the `CameraEngine` class.
    *   Uses `Promise.all` to query all candidate models concurrently with a `5000ms` fetch timeout.
    *   Appends the check results to the **Developer Diagnostics Panel** (`#debug-content`) for real-time visualization.
    *   Caches the selection in `localStorage.setItem('geminiActiveModel', selectedModel)`.
    *   Clears the cache when the input listener detects edits to the Gemini API Key settings field.

---

## 4. Verification & Testing

### Python Backend
Running the CLI with the `--auto` flag shows the parallel test logs and selection of the target model:
```bash
python ai_polisher.py --auto
```
**Example Log Output**:
```text
[AI] Checking available models for your account...
[AI] Testing 12 candidate models in parallel...
  - models/gemini-2.5-flash: OK
  - models/gemini-2.5-pro: Failed (429: You exceeded your current quota...)
  - models/gemini-2.0-flash: Failed (429: You exceeded your current quota...)
  ...
  - models/gemini-3.5-flash: OK
[AI] Selected best working model: models/gemini-3.5-flash
[AI] Reconstructing sentence...

==================================================
      --- AI CORRECTED SENTENCE ---
==================================================
Grandma Tonkla likes to run.
```

### Frontend Website
The compiler builds successfully with zero errors:
```bash
cd frontend_app
npm run build
```
Vite outputs the static HTML/CSS/JS bundle successfully. The local dev server running at `http://localhost:5173/` has been verified to execute the active model checks, write to local storage, and display the logs inside the Diagnostics panel when clicking the header title **SignTranslate**.
