# evaluation/run_shake_test.py

import os
os.environ["HF_HOME"] = "/vol/bitbucket/rc1124/hf_cache"

import csv
import json
from shake_pipeline import ShakePipeline
from attention_perturbation_llama import run_with_attention_perturbation
from shake_score import compute_shake_score

# Configuration
VISUALIZE_ATTENTION = False  # Set to True to enable visualization for specific samples
VISUALIZE_SAMPLE_IDS = [4]  # Which samples to visualize (1-indexed)

with open("/vol/bitbucket/rc1124/MSc_Individual_Project/ExplainabilityInLLMs-MScThesis/generators/truthful_claims_dataset.json", "r") as f:
    data = json.load(f)

samples = data[:817]
pipeline = ShakePipeline()
results = []

for idx, sample in enumerate(samples):
    claim = sample["claim"]
    ground_truth = sample["label"]
    ground_truth = str(ground_truth).upper()
    print("\n" + "="*80)
    print(f"🔬 Sample {idx+1}: {claim}")

    try:
        label_orig, conf_orig = pipeline.get_label_and_confidence(claim)
        rationale_tokens = pipeline.generate_rationale_tokens(claim)
        
        # Check if we should visualize this sample
        # should_visualize = VISUALIZE_ATTENTION and (idx + 1) in VISUALIZE_SAMPLE_IDS
        should_visualize = VISUALIZE_ATTENTION

        save_path = f"attention_sample_{idx+1}.png" if should_visualize else None
        
        label_pert, conf_pert = run_with_attention_perturbation(
            pipeline.model, 
            pipeline.tokenizer, 
            claim, 
            rationale_tokens,
            visualize=should_visualize,
            save_path=save_path
        )
        
        shake_score = compute_shake_score(label_orig, conf_orig, label_pert, conf_pert, ground_truth)
        print(f"Final SHAKE_SCORE = {shake_score:.4f}")

        results.append({
            "sample_id": sample["id"],
            "claim": claim,
            "label_orig": label_orig,
            "conf_orig": round(conf_orig, 3),
            "label_pert": label_pert,
            "conf_pert": round(conf_pert, 3),
            "shake_score": round(shake_score, 4),
            "ground_truth": ground_truth
        })

    except Exception as e:
        print(f"Error on sample {idx+1}: {e}")
        continue

csv_path = "shake_score_results_truthfulclaim.csv"
if results:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"SHAKE test results saved to: {csv_path}")
else:
    print("No results were generated due to errors in all samples.")
