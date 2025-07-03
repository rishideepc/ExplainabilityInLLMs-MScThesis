from datasets import load_dataset
import random

# Step 1: Load the TruthfulQA dataset
dataset = load_dataset("truthful_qa", "generation")

# Step 2: Initialize variables
true_claims = []
false_claims = []
desired_true = 409
desired_false = 408

# Step 3: Shuffle the dataset to randomize selection order
data_items = list(dataset["validation"])
random.shuffle(data_items)

# Step 4: Generate claims with balanced labels
for item in data_items:
    correct = item["correct_answers"]
    incorrect = item["incorrect_answers"]

    if len(true_claims) < desired_true and correct:
        claim = random.choice(correct)
        true_claims.append({
            "claim": claim,
            "label": "true",
            "question": item["question"],
            "source": item["source"]
        })
    elif len(false_claims) < desired_false and incorrect:
        claim = random.choice(incorrect)
        false_claims.append({
            "claim": claim,
            "label": "false",
            "question": item["question"],
            "source": item["source"]
        })

    # Stop once both lists are full
    if len(true_claims) >= desired_true and len(false_claims) >= desired_false:
        break

    

# Step 5: Combine and shuffle the final dataset
final_dataset = true_claims + false_claims
random.shuffle(final_dataset)

for idx, item in enumerate(final_dataset, start=2):
    item["id"] = idx

# Step 6: (Optional) Print a few examples
for i in range(827):
    print(f"{i+2}. [{final_dataset[i]['label']}] {final_dataset[i]['claim']}")

# Step 7: (Optional) Save to JSON or CSV
import json
with open("truthful_claims_dataset.json", "w") as f:
    json.dump(final_dataset, f, indent=2)