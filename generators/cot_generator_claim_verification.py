import sys
import os
project_root = os.path.abspath('..')
sys.path.append(project_root)
import json
import requests

# === CONFIGURATION ===
USE_OLLAMA = True
MODEL = "qwen3:4b"  
INPUT_FILE = "generators/med_claims_dataset.json"
OUTPUT_FILE = f"results/generation/medclaim_qwen/cot_outputs_claims.jsonl"


# === Load JSON claims file ===
def load_claim_dataset(path):
    with open(path, "r") as f:
        return json.load(f)


# === Prompt generator ===
def make_cot_prompt(claim: str, label: str) -> str:
    prompt = f"""
Claim: {claim}
Q: Is the above claim true? Let's think step-by-step to ensure each part of our reasoning connects clearly to the final answer. 
Generate your explanation slightly elaborately! Conclude with a single-sentence verdict beginning with 'Conclusion:'.
A:"""
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
def generate_cot_explanations(data, model=MODEL):
    results = []
    # c=0
    for entry in data:
        claim = entry["claim"]
        label = entry["label"]  # either "true" or "false"
        prompt = make_cot_prompt(claim, label)
        

        if USE_OLLAMA:
            explanation = generate_with_ollama(prompt, model)
        else:
            explanation = "[ERROR] Non-Ollama models not yet supported."

        results.append({
            "id": entry.get("id"),
            "claim": claim,
            "label": label,
            "question": entry.get("question"),
            "cot_explanation": explanation
        })

        # c+=1
        # if c==2:
        #     break

    return results


# === MAIN ===
if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    claims_data = load_claim_dataset(INPUT_FILE)
    generated_results = generate_cot_explanations(claims_data)

    with open(OUTPUT_FILE, "w") as f:
        for item in generated_results:
            json.dump(item, f)
            f.write("\n")

    print(f"CoT explanations saved to {OUTPUT_FILE}")
