"""
Converter script for StrategyQA dataset to StrategyClaim dataset.
"""

import sys
import os
project_root = os.path.abspath('..')
sys.path.append(project_root)

import json
import subprocess
from datasets import load_dataset
import time

OUTPUT_FILE = "strategyclaim_final.json"
MODEL_NAME = "mistral"  

dataset = load_dataset("wics/strategy-qa")

questions = [entry["question"] for entry in dataset["validation"]]

answers = [entry["answer"] for entry in dataset["validation"]]

qids = [entry["qid"] for entry in dataset["validation"]]


generated_dataset = []

def create_prompt(question, answer):
    return f"""
    
Few shot examples: -

Question:  Are more people today related to Genghis Khan than Julius Caesar?
Answer:  True

Converted Claim: More people today are related to Genghis Khan than Julius Caesar



Question:  Could the members of The Police perform lawful arrests?
Answer:  False

Converted Claim: The members of The Police cannot perform lawful arrests.



Now similar to the above examples, convert the following Question-Answer pair into a claim statement: -

Question:  {question}
Answer:  {answer}

Just generate the claim statement as a response to this prompt, DO NOT generate any associated keys such as "Claim Statement:" or "Claim:" or "The claim is:" or "Converted Claim:"
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

for i in range(len(answers)):
    print(f"[{i+2}/{len(answers)}] Generating claim for ID {qids[i]} ({answers[i]})")
    prompt = create_prompt(questions[i], answers[i])
    claim = query_qwen(prompt)

    generated_dataset.append({
        "qid": qids[i],
        "question": questions[i],
        "claim": claim,
        "label/answer": answers[i]
    })

with open(OUTPUT_FILE, "w") as f:
    json.dump(generated_dataset, f, indent=2)

print(f"All done! Final dataset saved to '{OUTPUT_FILE}' with {len(generated_dataset)} entries.")