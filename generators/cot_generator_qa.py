import sys
import os
project_root = os.path.abspath('..')
sys.path.append(project_root)

from datasets import load_dataset
import openai
import json, os
import requests

USE_OLLAMA = True

dataset = load_dataset("GBaker/MedQA-USMLE-4-options")
questions = [entry["question"] for entry in dataset["test"]]
questions = questions[:1273]

# dataset = load_dataset("truthful_qa", "generation") 
# questions = dataset["validation"]["question"][:817]

# dataset = load_dataset("commonsense_qa") 
# questions = [entry["question"] for entry in dataset["validation"]] 
# questions = questions[:1221]

# dataset = load_dataset("wics/strategy-qa") 
# questions = [entry["question"] for entry in dataset["test"]] 
# questions = questions[:2290]

def make_cot_prompt(question: str) -> str:
    """
    Builds a few-shot Chain-of-Thought prompt to elicit step-by-step explanations

    @params: input question
    
    @returns: formatted prompt string
    """
    few_shot_examples = """
Q: Why does ice float on water?
A:
1. Water molecules form hydrogen bonds.
2. As water freezes, these molecules arrange into a crystalline structure.
3. This structure takes up more space and reduces density.
4. Less dense substances float on denser liquids.
Conclusion: Ice floats because its crystal structure makes it less dense than liquid water.

Q: What causes thunder?
A:
1. Lightning rapidly heats the surrounding air.
2. The sudden heat causes the air to expand explosively.
3. This rapid expansion creates a shockwave.
4. That shockwave is what we hear as thunder.
Conclusion: Thunder is the sound of air expanding rapidly due to lightning.

Q: {question}
Let's think step-by-step to ensure each part of our reasoning connects clearly to the final answer.
Generate your answer slightly elaborately!
A:
""".strip()
    return few_shot_examples.format(question=question)

def generate_with_ollama(prompt, model="qwen3:4b"):
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

def generate_cot_explanations(questions, model="qwen3:4b"):
    """
    Generates CoT explanations for a list of questions

    @params: questions, model name

    @returns: list of dictionaries with 'question' and 'cot_explanation'
    """
    results = []
    for q in questions:
        prompt = make_cot_prompt(q)
        if USE_OLLAMA:
            explanation = generate_with_ollama(prompt, model="qwen3:4b")
        else:
            explanation = "[ERROR - Unsupported backend]"
        results.append({"question": q, "cot_explanation": explanation})
    return results

if __name__ == "__main__":
    results = generate_cot_explanations(questions)
    if USE_OLLAMA:
        output_file = "results/generation/medqa_qwen/cot_outputs_ollama_meta_reasoning_conclusion_step_indices_fewshot.jsonl"
    else:
        output_file = "results/generation/medqa_qwen/cot_outputs_other_backend.jsonl"
    with open(output_file, "w") as f:
        for item in results:
            json.dump(item, f)
            f.write("\n")
    print(f"CoT outputs saved to {output_file}")
