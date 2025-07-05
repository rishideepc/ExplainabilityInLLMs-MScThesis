import json
from typing import List, Dict
import sys
import os
project_root = os.path.abspath('...')
sys.path.append(project_root)

from evaluation.Evaluating_Explanations.src.metrics.argumentative_metrics import compute_circularity, compute_dialectical_acceptability
from evaluation.Evaluating_Explanations.src.metrics.deductive_metrics import compute_redundancy, compute_strong_relevance, compute_weak_relevance
# from evaluation.EvaluatingExplanations.src.metrics.freeform_metrics import CoherenceEvaluator

# === Load jsonl data ===
def load_jsonl(filepath: str) -> List[Dict]:
    with open(filepath, "r") as f:
        return [json.loads(line.strip()) for line in f.readlines()]

##### === Evaluation for Chain-of-Thought explanations (Deductive) === #####

def evaluate_cot_metrics(entry: Dict, coherence_model=None) -> Dict[str, float]:
    explanation = entry.get("cot_explanation", "")
    steps = [s.strip() for s in explanation.split("\n") if s.strip()]

    matrix = {step: [steps[i+1]] if i < len(steps)-1 else [] for i, step in enumerate(steps)}
    propositions = steps
    y_hat = steps[-1]

    metrics = {
        "redundancy": compute_redundancy(matrix, propositions, y_hat),
        "weak_relevance": compute_weak_relevance(matrix, propositions, y_hat),
        "strong_relevance": compute_strong_relevance(matrix, propositions, y_hat)
    }

    # Optional coherence model
    if coherence_model:
        try:
            metrics["coherence"] = coherence_model.coherence_metric(steps, y_hat)
        except Exception:
            metrics["coherence"] = -1.0
    else:
        metrics["coherence"] = -1.0

    return metrics


##### === Evaluation for argLLM explanations (Argumentative) === #####

def evaluate_argllm_metrics(entry: Dict) -> Dict[str, float]:
    base_bag = entry.get("base", {}).get("bag", {})
    args = list(base_bag.get("arguments", {}).keys())
    attacks = base_bag.get("attacks", [])
    supports = base_bag.get("supports", [])

    # Assume conclusion is db0 for base tree
    y_hat = ["db0"] if "db0" in args else []

    return {
        "circularity": compute_circularity(args, attacks, supports),
        "acceptability": compute_dialectical_acceptability(args, attacks, y_hat)
    }

# === Helper evaluation function ===

def evaluate_all_cot(filepath: str) -> List[Dict]:
    data = load_jsonl(filepath)
    coherence_model = None
    return [dict(q=item["question"], **evaluate_cot_metrics(item, coherence_model)) for item in data]

def evaluate_all_argllm(filepath: str) -> List[Dict]:
    data = load_jsonl(filepath)
    return [dict(q=item["question"], **evaluate_argllm_metrics(item)) for item in data]

