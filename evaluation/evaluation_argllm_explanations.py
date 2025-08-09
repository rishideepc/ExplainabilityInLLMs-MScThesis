import sys
import os
project_root = os.path.abspath('..')
sys.path.append(project_root)

import json
import csv
import os
from tqdm import tqdm

from Evaluating_Explanations.src.metrics.argumentative_metrics import (
    compute_circularity,
    compute_dialectical_acceptability,
    compute_dialectical_faithfulness, 
    compute_dialectical_acceptability_v2
)

dataset_model = "truthfulclaim_mistral_temp_3"  # Using dataset and model
# === CONFIG ===
INPUT_FILE = f"/vol/bitbucket/rc1124/MSc_Individual_Project/ExplainabilityInLLMs-MScThesis/results/generation/argLLM_generation/{dataset_model}/argllm_outputs_ollama.jsonl"
OUTPUT_CSV = f"/vol/bitbucket/rc1124/MSc_Individual_Project/ExplainabilityInLLMs-MScThesis/results/temp/argllm_eval_scores_{dataset_model}.csv"
CONFIDENCE_LEVELS = ["top", "high", "low"]


def evaluate_explanation(sample_id, sample_json):
    try:
        # Extract argument dictionary
        bag = sample_json["base"]["bag"]
        arguments_dict = bag["arguments"]
        # attack_relations = bag.get("attacks", [])
        # support_relations = bag.get("supports", [])
        attack_relations = bag.get("attacks", [])
        support_relations_raw = bag.get("supports", [])
        prediction_confidence = sample_json["base"]["prediction"]

        support_relations_for_circ = [
            (supported, supporter) for (supporter, supported) in support_relations_raw
        ]

        # support_relations_for_circ = [
        #     (supporter, supported) for (supporter, supported) in support_relations_raw
        # ]
        # === ARGUMENT NAMES ===
        args = list(arguments_dict.keys())

        # === ASSUMPTION ===: central claim = db0
        target_arg = "db0"
        args_y_hat = [target_arg] if target_arg in args else []

        # === Compute Metrics ===
        circ_score = compute_circularity(args, attack_relations, support_relations_for_circ)
        # acc_score = compute_dialectical_acceptability(args, attack_relations, args_y_hat)
        acc_score = compute_dialectical_acceptability_v2(
    args=args,
    attack_relations=attack_relations,
    support_relations=support_relations_raw,   # this one expects (supporter, supported)
    args_y_hat=args_y_hat,
    normalize_by="yhat",                       # <-- divide by #y_hat instead of num_nodes
    implicit_support_defense=False              # <-- treat supporters as implicit defenders
)
        # faithfulness_scores = {
        #     level: compute_dialectical_faithfulness(
        #         args,
        #         attack_relations,
        #         support_relations,
        #         args_y_hat,
        #         confidence_level=level
        #     ) for level in CONFIDENCE_LEVELS
        # }

        faithfulness_scores = {
    level: compute_dialectical_faithfulness(
        args,
        attack_relations,
        support_relations_raw,  # faithfulness expects (supporter, supported)
        args_y_hat,
        confidence_level=level
    ) for level in CONFIDENCE_LEVELS
}

        return {
            "id": sample_id,
            "circ_score": circ_score,
            "acceptability_score": acc_score,
            "faithfulness_top": faithfulness_scores["top"],
            "faithfulness_high": faithfulness_scores["high"],
            "faithfulness_low": faithfulness_scores["low"],
            "prediction_confidence": prediction_confidence,
            "num_args": len(args),
            "num_attacks": len(attack_relations),
            "num_supports": len(support_relations_for_circ),
        }

    except Exception as e:
        print(f"[⚠️ ERROR] Sample ID {sample_id} failed: {e}")
        return None


def load_jsonl(filepath):
    with open(filepath, 'r') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def save_to_csv(results, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = [
        "id", "circ_score", "acceptability_score",
        "faithfulness_top", "faithfulness_high", "faithfulness_low",
        "prediction_confidence", "num_args", "num_attacks", "num_supports"
    ]
    with open(output_path, "w", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            if row is not None:
                writer.writerow(row)


if __name__ == "__main__":
    print("🔍 Loading explanations...")
    data = load_jsonl(INPUT_FILE)

    print("🚀 Evaluating explanations...")
    results = []
    for idx, sample in tqdm(enumerate(data), total=len(data)):
        result = evaluate_explanation(sample_id=idx, sample_json=sample)
        print(result, "\n")
        results.append(result)

    print("💾 Saving results to CSV...")
    save_to_csv(results, OUTPUT_CSV)
    print(f"✅ Done! Results saved to: {OUTPUT_CSV}")
