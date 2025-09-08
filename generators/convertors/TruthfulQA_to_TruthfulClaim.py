"""
Converter script for TruthfulQA dataset to TruthfulClaim dataset.
"""

from datasets import load_dataset
import random
import json
import time
import requests

desired_true = 409
desired_false = 408
OLLAMA_MODEL = "mistral"
DELAY_BETWEEN_CALLS = 0.5 

dataset = load_dataset("truthful_qa", "generation")
data_items = list(dataset["validation"])
random.shuffle(data_items) 

def generate_claim_with_mistral(question, answer):
    prompt = f"""Rephrase the following question and answer as a complete factual claim:

Q: {question}
A: {answer}

Claim:"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            }
        )
        response.raise_for_status()
        claim = response.json().get("response", "").strip()
        return claim
    except Exception as e:
        print(f"Error generating claim: {e}")
        return None

true_claims = []
false_claims = []

for item in data_items:
    question = item["question"]
    correct_answers = item["correct_answers"]
    incorrect_answers = item["incorrect_answers"]
    source = item["source"]

    label_choice = random.choice(["true", "false"])

    success = False  

    for label in [label_choice, "false" if label_choice == "true" else "true"]:
        if label == "true" and len(true_claims) < desired_true and correct_answers:
            answer = random.choice(correct_answers)
            claim = generate_claim_with_mistral(question, answer)
            if claim:
                true_claims.append({
                    "claim": claim,
                    "label": "true",
                    "question": question,
                    "source": source
                })
                print(f"True claim #{len(true_claims)}: {claim}")
                success = True
                break

        elif label == "false" and len(false_claims) < desired_false and incorrect_answers:
            answer = random.choice(incorrect_answers)
            claim = generate_claim_with_mistral(question, answer)
            if claim:
                false_claims.append({
                    "claim": claim,
                    "label": "false",
                    "question": question,
                    "source": source
                })
                print(f"False claim #{len(false_claims)}: {claim}")
                success = True
                break

    if success:
        time.sleep(DELAY_BETWEEN_CALLS)

    if len(true_claims) >= desired_true and len(false_claims) >= desired_false:
        break

final_dataset = true_claims + false_claims
random.shuffle(final_dataset)

for idx, item in enumerate(final_dataset, start=2):
    item["id"] = idx

output_path = "truthful_claims_dataset.json"
with open(output_path, "w") as f:
    json.dump(final_dataset, f, indent=2)

print(f"\nDataset saved to {output_path} with {len(final_dataset)} items.")