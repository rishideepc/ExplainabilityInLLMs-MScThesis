import sys
import os
project_root = os.path.abspath('..')
sys.path.append(project_root)

import json
import random
import subprocess
from datasets import load_dataset
from tqdm import tqdm

# Seed for reproducibility
random.seed(42)

# Load the CommonsenseQA validation split
dataset = load_dataset("commonsense_qa")["validation"]

# Helper to map answerKey (e.g., "A") to the corresponding text
def get_option_text(entry, key):
    idx = entry["choices"]["label"].index(key)
    return entry["choices"]["text"][idx]

# Helper to get a random incorrect answer
def get_random_incorrect_option(entry, correct_key):
    labels = entry["choices"]["label"]
    texts = entry["choices"]["text"]
    incorrect = [(l, t) for l, t in zip(labels, texts) if l != correct_key]
    return random.choice(incorrect)[1]

# LLM prompt function using Mistral via Ollama CLI
def generate_claim_with_qwen(question, option):
    prompt = f"""
Few shot examples: -

Question: A revolving door is convenient for two direction travel, but it also serves as a security measure at a what?
Answer: bank
Converted Claim: A revolving door is convenient for two direction travel, but it also serves as a security measure at a bank.


Question: What do people aim to do at work?
Answer: complete job
Converted Claim: People aim to complete the job at work.



Now similar to the above examples, convert the following Question-Answer pair into a claim statement: -

Question: {question}
Answer: {option}

Just generate the claim statement as a response to this prompt, DO NOT generate any associated keys such as "Claim Statement:" or "Claim:" or "The claim is:" or "Converted Claim:"
"""
    result = subprocess.run(
        ["ollama", "run", "mistral"],
        input=prompt.encode("utf-8"),
        capture_output=True
    )
    response = result.stdout.decode("utf-8").strip()
    # Strip prompt echoes and artifacts if present
    return response.split("Claim:")[-1].strip()

# Build dataset entries
output_data = []

print("\n🟢 Generating TRUE claims (correct answers)...")
for i in tqdm(range(611)):
    entry = dataset[i]
    question = entry["question"]
    correct_key = entry["answerKey"]
    option = get_option_text(entry, correct_key)
    claim = generate_claim_with_qwen(question, option)
    output_data.append({
        "id": i + 1,
        "question": question,
        "option": option,
        "claim": claim,
        "label": "true",
        "source_id": entry["id"]
    })

print("\n🔴 Generating FALSE claims (random wrong answers)...")
for i in tqdm(range(611, 1221)):
    entry = dataset[i]
    question = entry["question"]
    correct_key = entry["answerKey"]
    option = get_random_incorrect_option(entry, correct_key)
    claim = generate_claim_with_qwen(question, option)
    output_data.append({
        "id": i + 1,
        "question": question,
        "option": option,
        "claim": claim,
        "label": "false",
        "source_id": entry["id"]
    })

# Shuffle the final dataset
print("\n🔀 Shuffling final dataset...")
random.shuffle(output_data)

# Save to JSON
output_path = "commonsenseclaim.json"
with open(output_path, "w") as f:
    json.dump(output_data, f, indent=2)

print(f"\n✅ CommonsenseClaim dataset saved to '{output_path}' with {len(output_data)} entries.")
