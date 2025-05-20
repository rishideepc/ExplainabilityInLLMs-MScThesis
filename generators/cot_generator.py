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
# def make_cot_prompt(q):
#     # return f"Q: {q}\nLet’s think step-by-step. \nA:" # basic zero-shot CoT method
#     # return f"Q: {q}\nLet’s think step-by-step to ensure each part of our reasoning connects clearly to the final answer. Generate your answer slightly elaborately!\nA:" # adding meta-reasoning instruction + explanation elaboration sub-prompt
#     return f"""Q: {q}
#     Let's think step-by-step to ensure each part of our reasoning connects clearly to the final answer.
#     Generate your answer slightly elaborately and finish with a single-sentence conclusion beginning with 'Conclusion:'.
#     A:""" # adding meta-reasoning instruction + explanation elaboration sub-prompt + explicit 'conclusion' sub-prompt


# def make_cot_prompt(q):
#     return f"""Q: {q}
#     Let's think step-by-step to ensure each part of our reasoning connects clearly to the final answer.
#     Generate your answer slightly elaborately!
#     A:
#     Step 1:
#     Step 2:
#     Step 3:
#     Step 4:
#     Conclusion:"""
#     # adding meta-reasoning instruction + explanation elaboration sub-prompt + explicit 'conclusion' sub-prompt + step-indices annotation


def make_cot_prompt(question: str) -> str:
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
        output_file = "results/generation/cot_outputs_ollama_meta_reasoning_conclusion_step_indices_fewshot.jsonl"
    else:
        pass
    with open(output_file, "w") as f:
        for item in results:
            json.dump(item, f)
            f.write("\n")
    print(f"CoT outputs saved to {output_file}")
