import os
import sys
import argparse
import json
import requests
import concurrent.futures

# PASTE YOUR API KEY HERE or set GEMINI_API_KEY environment variable:
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

def check_model_availability(model_name):
    test_url = f"https://generativelanguage.googleapis.com/v1/{model_name}:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{
            "parts": [{"text": "hi"}]
        }]
    }
    headers = {'Content-Type': 'application/json'}
    try:
        r = requests.post(test_url, headers=headers, data=json.dumps(payload), timeout=5)
        if r.status_code == 200:
            return model_name, True, "OK"
        else:
            err_msg = r.json().get('error', {}).get('message', f"HTTP {r.status_code}")
            return model_name, False, f"Failed ({r.status_code}: {err_msg})"
    except Exception as e:
        return model_name, False, f"Failed ({type(e).__name__}: {str(e)})"

def polish_with_ai(lattice_data):
    # --- DEBUG: SHOW THE BRACKETS ---
    print("\n" + "-"*50)
    print("      --- DEBUG: RAW WORD BRACKETS ---")
    print("-"*50)
    print(lattice_data.strip())
    print("-" * 50 + "\n")

    if GEMINI_API_KEY == "YOUR_API_KEY_HERE":
        print("\n[NOTE] No API Key found.")
        return

    print("[AI] Checking available models for your account...")
    
    list_url = f"https://generativelanguage.googleapis.com/v1/models?key={GEMINI_API_KEY}"
    try:
        r_list = requests.get(list_url)
        if r_list.status_code == 200:
            models = r_list.json().get('models', [])
            available_models = [m['name'] for m in models if 'generateContent' in m.get('supportedGenerationMethods', [])]
            
            if not available_models:
                print("[AI ERROR] No generation models found for this key.")
                return
            
            print(f"[AI] Testing {len(available_models)} candidate models in parallel...")
            working_models = []
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                results = executor.map(check_model_availability, available_models)
                for name, is_working, status_str in results:
                    print(f"  - {name}: {status_str}")
                    if is_working:
                        working_models.append(name)
            
            if not working_models:
                print("[AI ERROR] Checked all models but none of them are working/authorized.")
                return

            def get_model_rank(model_name):
                model_name_lower = model_name.lower()
                if "gemini-3.5-flash" in model_name_lower:
                    return 0
                elif "gemini-3.1-flash" in model_name_lower:
                    return 1
                elif "gemini-2.5-flash" in model_name_lower:
                    return 2
                elif "gemini-2.0-flash" in model_name_lower:
                    return 3
                elif "gemini-1.5-flash" in model_name_lower:
                    return 4
                elif "flash" in model_name_lower:
                    return 5
                elif "gemini" in model_name_lower:
                    return 6
                else:
                    return 7

            working_models.sort(key=get_model_rank)
            target_model = working_models[0]
            print(f"[AI] Selected best working model: {target_model}")
        else:
            print(f"[AI ERROR] Could not list models: {r_list.status_code}")
            return
    except Exception as e:
        print(f"[AI ERROR] Discovery failed: {e}")
        return

    print("[AI] Reconstructing sentence...")
    
    url = f"https://generativelanguage.googleapis.com/v1/{target_model}:generateContent?key={GEMINI_API_KEY}"
    
    prompt = f"""
INSTRUCTION: You are a professional Sign Language Translation Assistant.
You will receive a list of "Word Brackets" from a deaf person using a sign language app.
Each bracket has 3 words. The FIRST word is the top priority from our engine.

YOUR PRIORITY RULES:
1. The first word is the most prioritized. If it makes sense in a sentence, use it.
2. If the second word makes sense as well alongside the first, you can put the first and second word (e.g. "I [eat/drink] chicken").
3. If the second or third words don't make any sense in context, CROSS THEM OUT (discard them).
4. If the first word is clearly the best and only logical choice, choose it and discard all others.
5. Fix the overall grammar into a natural, flowing sentence.

LATTICE DATA (The signing sequence):
{lattice_data}

OUTPUT ONLY THE FINAL POLISHED SENTENCE.
"""

    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }
    
    headers = {'Content-Type': 'application/json'}

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        result = response.json()
        
        if response.status_code == 200:
            if 'candidates' in result and len(result['candidates']) > 0:
                text = result['candidates'][0]['content']['parts'][0]['text']
                print("\n" + "="*50)
                print("      --- AI CORRECTED SENTENCE ---")
                print("="*50)
                print(f"\n{text.strip()}")
                print("\n" + "="*50)
            else:
                print(f"[AI ERROR] No candidates in response: {result}")
        else:
            print(f"[AI ERROR] {response.status_code}: {result.get('error', {}).get('message', 'Unknown Error')}")

    except Exception as e:
        print(f"[AI ERROR] {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", action="store_true", help="Read from bridge_lattice.txt")
    args = parser.parse_args()

    if args.auto:
        if os.path.exists("bridge_lattice.txt"):
            with open("bridge_lattice.txt", "r") as f:
                lattice_data = f.read()
            polish_with_ai(lattice_data)
        else:
            print("[ERROR] bridge_lattice.txt not found.")
    else:
        print("Paste your Lattice (Word 1: { ... }) and press Ctrl+Z (Windows) then Enter:")
        data = sys.stdin.read()
        if data.strip():
            polish_with_ai(data)
