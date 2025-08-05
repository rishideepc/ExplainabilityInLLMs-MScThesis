# from datasets import load_dataset
# import random

# # Step 1: Load the TruthfulQA dataset
# dataset = load_dataset("truthful_qa", "generation")

# # Step 2: Initialize variables
# true_claims = []
# false_claims = []
# desired_true = 409
# desired_false = 408

# # Step 3: Shuffle the dataset to randomize selection order
# data_items = list(dataset["validation"])
# random.shuffle(data_items)

# # Step 4: Generate claims with balanced labels
# for item in data_items:
#     correct = item["correct_answers"]
#     incorrect = item["incorrect_answers"]

#     if len(true_claims) < desired_true and correct:
#         claim = random.choice(correct)
#         true_claims.append({
#             "claim": claim,
#             "label": "true",
#             "question": item["question"],
#             "source": item["source"]
#         })
#     elif len(false_claims) < desired_false and incorrect:
#         claim = random.choice(incorrect)
#         false_claims.append({
#             "claim": claim,
#             "label": "false",
#             "question": item["question"],
#             "source": item["source"]
#         })

#     # Stop once both lists are full
#     if len(true_claims) >= desired_true and len(false_claims) >= desired_false:
#         break

    

# # Step 5: Combine and shuffle the final dataset
# final_dataset = true_claims + false_claims
# random.shuffle(final_dataset)

# for idx, item in enumerate(final_dataset, start=2):
#     item["id"] = idx

# # Step 6: (Optional) Print a few examples
# for i in range(827):
#     print(f"{i+2}. [{final_dataset[i]['label']}] {final_dataset[i]['claim']}")

# # Step 7: (Optional) Save to JSON or CSV
# import json
# with open("truthful_claims_dataset.json", "w") as f:
#     json.dump(final_dataset, f, indent=2)


from datasets import load_dataset
import random
import json
import time
import requests

# --- Configuration ---
desired_true = 409
desired_false = 408
OLLAMA_MODEL = "mistral"
DELAY_BETWEEN_CALLS = 0.5  # seconds, to avoid overwhelming Ollama

# --- Step 1: Load the TruthfulQA dataset ---
dataset = load_dataset("truthful_qa", "generation")
data_items = list(dataset["validation"])
random.shuffle(data_items)  # Randomize selection

# --- Step 2: Define function to query Mistral via Ollama ---
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

# --- Step 3: Initialize claim containers ---
true_claims = []
false_claims = []

# --- Step 4: Loop through dataset and generate claims ---
for item in data_items:
    question = item["question"]
    correct_answers = item["correct_answers"]
    incorrect_answers = item["incorrect_answers"]
    source = item["source"]

    # Determine which label to attempt first (randomized)
    label_choice = random.choice(["true", "false"])

    success = False  # Track if we successfully added a claim

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
                print(f"[✓] True claim #{len(true_claims)}: {claim}")
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
                print(f"[✗] False claim #{len(false_claims)}: {claim}")
                success = True
                break

    if success:
        time.sleep(DELAY_BETWEEN_CALLS)

    if len(true_claims) >= desired_true and len(false_claims) >= desired_false:
        break

# --- Step 5: Combine, shuffle, and assign IDs ---
final_dataset = true_claims + false_claims
random.shuffle(final_dataset)

for idx, item in enumerate(final_dataset, start=2):
    item["id"] = idx

# --- Step 6: Save to JSON file ---
output_path = "truthful_claims_dataset.json"
with open(output_path, "w") as f:
    json.dump(final_dataset, f, indent=2)

print(f"\n✅ Dataset saved to {output_path} with {len(final_dataset)} items.")