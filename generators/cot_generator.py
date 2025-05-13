from datasets import load_dataset
import openai
import json, os
import requests

# Flags
USE_OLLAMA = True

if not USE_OLLAMA:
    pass

# Load initial experiment questions
dataset = load_dataset("truthful_qa", "generation")
questions = dataset["validation"]["question"][:20]

# Prompt generator for zero-shot CoT
def make_cot_prompt(q):
    return f"Q: {q}\nLet's think step-by-step.\nA:"

###### Model 1: Local LLM via Ollama ######
def generate_with_ollama(prompt, model="mistral"):
    try:
        res = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False}
        )
        res_json = res.json()

        # Log full response for debugging
        if "response" not in res_json:
            print("Ollama API returned unexpected structure:")
            print(res_json)
            return "[ERROR - OLLAMA] Response key missing"

        return res_json["response"]

    except Exception as e:
        return f"[ERROR - OLLAMA Exception] {str(e)}"


###### Model 2: Boilerplate for any other open-access model ######
##################################################################

# Method for CoT prompting explanations
def generate_cot_explanations(questions, model="mistral"):
    results = []
    for q in questions:
        prompt = make_cot_prompt(q)
        if USE_OLLAMA:
            explanation = generate_with_ollama(prompt, model="mistral")  # can be: llama2, gemma
        else:
            pass
        results.append({"question": q, "cot_explanation": explanation})
    return results

# Main function
if __name__ == "__main__":
    results = generate_cot_explanations(questions)
    if USE_OLLAMA:
        output_file = "results/generation/cot_outputs_ollama.jsonl"
    else:
        pass
    with open(output_file, "w") as f:
        for item in results:
            json.dump(item, f)
            f.write("\n")
    print(f"CoT outputs saved to {output_file}")
