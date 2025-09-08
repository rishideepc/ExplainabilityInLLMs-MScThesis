import sys
import os
project_root = os.path.abspath('..')
sys.path.append(project_root)
import json
import requests

USE_OLLAMA = True
MODEL = "llama3:latest"
INPUT_FILE = "generators/truthful_claims_dataset.json"
OUTPUT_FILE = f"results/generation/truthfulclaim_llama/cot_outputs_claims.jsonl"

def load_claim_dataset(path):
    """
    Loads entire claim dataset from JSON file

    @params: input JSON file path
    
    @returns: list of claim entries
    """
    with open(path, "r") as f:
        return json.load(f)

def make_cot_prompt(claim: str, label: str) -> str:
    """
    Builds a few-shot Chain-of-Thought prompt to elicit step-by-step explanations

    @params: input claim
    
    @returns: formatted prompt string
    """
    prompt = f"""
Claim: {claim}
Q: Is the above claim true? Let's think step-by-step to ensure each part of our reasoning connects clearly to the final answer. 
Generate your explanation slightly elaborately! Conclude with a single-sentence verdict beginning with 'Conclusion:'.
A:"""
    return prompt.strip()

def generate_with_ollama(prompt, model=MODEL):
    """
    Passes prompt to a local Ollama model and returns the LLM-generated response

    @params: prompt string, model name
    
    @returns: response string from LLM
    """
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

def generate_cot_explanations(data, model=MODEL):
    """
    Generates CoT explanations for a list of questions

    @params: questions, model name

    @returns: list of dictionaries with 'question' and 'cot_explanation'
    """
    results = []
    for entry in data:
        claim = entry["claim"]
        label = entry["label"]
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
    return results

if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    claims_data = load_claim_dataset(INPUT_FILE)
    generated_results = generate_cot_explanations(claims_data)
    with open(OUTPUT_FILE, "w") as f:
        for item in generated_results:
            json.dump(item, f)
            f.write("\n")
    print(f"CoT explanations saved to {OUTPUT_FILE}")
