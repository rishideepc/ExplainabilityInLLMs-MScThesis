import sys
import os
project_root = os.path.abspath('..')
sys.path.append(project_root)
import json
import requests

# === CONFIGURATION ===
USE_OLLAMA = True
MODEL = "qwen"  
INPUT_FILE = "generators/truthful_claims_dataset.json"


# === Load JSON claims file ===
def load_claim_dataset(path):
    with open(path, "r") as f:
        return json.load(f)


# === Prompt generator ===
def make_cot_prompt(claim: str) -> str:
    prompt = f"""
Claim: {claim}

State if the above claim true or false, based on your reasoning.


Simply answer "true" or "false" (in one word, all lowercase, no punctuations) and generate NOTHING ELSE as part of your response.
"""
    return prompt.strip()


# === Generate using Ollama ===
def generate_with_ollama(prompt, model=MODEL):
    try:
        res = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False}
        )
        res_json = res.json()

        if "response" not in res_json:
            print("Ollama API returned unexpected structure:")
            print(res_json)
            return "[ERROR - OLLAMA] Response key missing"

        return res_json["response"]

    except Exception as e:
        return f"[ERROR - OLLAMA Exception] {str(e)}"


# === Generate CoT explanations ===
def generate_answer(data, model=MODEL):
    results = []
    # c=0
    for entry in data:
        claim = entry["claim"]
        label = entry["label"]
        prompt = make_cot_prompt(claim)
        

        if USE_OLLAMA:
            answer = generate_with_ollama(prompt, model)
        else:
            answer = "[ERROR] Non-Ollama models not yet supported."

        results.append({
            "id": entry.get("id"),
            "claim": claim,
            "actual label": label,
            "generated label": answer.replace(" ", "").replace(".", "").lower()
        })

        # c+=1
        # if c==2:
        #     break

    print(results)
    return results



