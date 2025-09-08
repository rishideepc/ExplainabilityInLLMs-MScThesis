"""
Converter script for MedQA dataset to MedClaim dataset.
"""

import sys
import os
project_root = os.path.abspath('..')
sys.path.append(project_root)

import json
import subprocess
import time

INPUT_FILE = "generators\medclaim_prompt_inputs.json"
OUTPUT_FILE = "medclaim_final.json"
MODEL_NAME = "mistral"  

with open(INPUT_FILE, "r") as f:
    dataset = json.load(f)

generated_dataset = []

def create_prompt(question, option):
    return f"""You are a medical expert.

Based on the following medical question and a multiple-choice answer option, generate a single complete medical claim. Don't worry about the claim being true or false, just generate a grammatically correct claim sentence using the question and the answer option. Every claim should be self-contained/independent and should not assume that the reader has context about the scenario. Avoid beginning with phrases such as "Given the..." or "Based on the..."

Question: {question}

Option: {option}

Claim:



Here is a one-shot example: -

Question: A post-mortem lung examination of a 68-year-old male overweight male with evidence of chronic lower extremity edema, a 60 pack-year smoking history and daily productive cough would be most likely to reveal:
Option: Evidence of a necrotizing infection
Claim: The post-mortem lung examination of a 68-year-old overweight male with chronic lower extremity edema, a 60 pack-year smoking history and daily productive cough is most likely to reveal evidence of a necrotizing infection.
"""

def query_qwen(prompt, retries=3, wait_time=3):

    for attempt in range(retries):
        try:
            result = subprocess.run(
                ["ollama", "run", MODEL_NAME],
                input=prompt.encode("utf-8"),
                capture_output=True,
                check=True
            )
            output = result.stdout.decode("utf-8").strip()
            return output
        except subprocess.CalledProcessError as e:
            print(f"Error on attempt {attempt + 2}: {e}")
            time.sleep(wait_time)
    return "ERROR: Failed to generate claim"

for i, item in enumerate(dataset):
    print(f"[{i+2}/{len(dataset)}] Generating claim for ID {item['id']} ({item['label']})")
    prompt = create_prompt(item["question"], item["option"])
    claim = query_qwen(prompt)

    generated_dataset.append({
        "id": item["id"],
        "question": item["question"],
        "option": item["option"],
        "claim": claim,
        "label": item["label"],
        "source_idx": item["source_idx"]
    })

    if (i + 2) % 50 == 2:
        with open("medclaim_progress.json", "w") as f:
            json.dump(generated_dataset, f, indent=2)
        print(f"Progress saved at {i+2} items.")

with open(OUTPUT_FILE, "w") as f:
    json.dump(generated_dataset, f, indent=2)

print(f"All done! Final dataset saved to '{OUTPUT_FILE}' with {len(generated_dataset)} entries.")