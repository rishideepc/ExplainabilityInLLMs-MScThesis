# evaluation/run_example_pipeline.py

import csv
import random
import numpy as np
from datasets import load_dataset
from faithfulness_pipeline import FaithMultiPipeline
from faithfulness_metrics import compute_faith_multi
from perturbation_utils import compute_causal_impact

# Load and split TruthfulQA
dataset = load_dataset("truthful_qa", "generation")["validation"]
questions = [example["question"] for example in dataset]
random.seed(42)
random.shuffle(questions)
split_idx = int(0.7 * len(questions))
test_questions = questions[split_idx:]
print(f"✅ Using {len(test_questions)} test samples")

# Initialize pipeline
pipeline = FaithMultiPipeline(model="llama3")

# Prepare CSV output
csv_path = "faith_test_results.csv"
fieldnames = ["sample_id", "question", "FAITH_ATTRIB", "FAITH_CAUSAL", "FAITH_SUFF", "ALIGN_CROSS", "FAITH_MULTI"]
rows = []

for idx, text in enumerate(test_questions):
    print(f"\n🔬 Test Sample {idx + 1}: {text.strip()[:60]}")
    
    try:
        label, full_conf = pipeline.get_prediction_and_confidence(text)
        explanation = pipeline.generate_explanation(text)
        expl_tokens = pipeline.tokenize_explanation(explanation)
        attrib_tokens = pipeline.get_attribution_tokens(text)
        causal_score = compute_causal_impact(pipeline, text, attrib_tokens, label, full_conf)
        suff_score = pipeline.get_suff_confidence(expl_tokens, label)

        scores = compute_faith_multi(
            full_conf=full_conf,
            explanation_tokens=expl_tokens,
            erased_confs=[full_conf - causal_score] * len(attrib_tokens),
            suff_conf=suff_score,
            attrib_tokens=attrib_tokens,
            causal_tokens=attrib_tokens,
            suff_tokens=expl_tokens
        )

        print("📊 FAITH_MULTI Breakdown:")
        for k, v in scores.items():
            print(f"{k}: {v:.4f}")

        row = {
            "sample_id": idx + 1,
            "question": text.strip(),
            **{k: round(v, 4) for k, v in scores.items()}
        }
        rows.append(row)

    except Exception as e:
        print(f"⚠️ Failed test sample: {e}")
        continue

# Compute averages
if rows:
    avg_row = {
        "sample_id": "AVG",
        "question": "Average over all test samples",
    }
    for key in fieldnames[2:]:  # Skip sample_id and question
        avg_value = np.mean([row[key] for row in rows])
        avg_row[key] = round(avg_value, 4)
    rows.append(avg_row)

# Write to CSV
with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"\n✅ Results written to: {csv_path}")
