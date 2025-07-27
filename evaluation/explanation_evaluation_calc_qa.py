# evaluation/explanation_evaluation_calc.py

import json
from typing import List, Dict
import torch
import sys
import os

from transformers import AutoModelForSequenceClassification, AutoTokenizer

project_root = os.path.abspath('...')
sys.path.append(project_root)

from evaluation.Evaluating_Explanations.src.metrics.argumentative_metrics import (
    compute_circularity, compute_dialectical_acceptability
)
from evaluation.Evaluating_Explanations.src.metrics.deductive_metrics import (
    compute_redundancy, compute_strong_relevance, compute_weak_relevance
)

# === Load model + tokenizer ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "roberta-large-mnli"
MAX_LEN = 512

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
model.eval()

# === Load jsonl data ===
def load_jsonl(filepath: str) -> List[Dict]:
    with open(filepath, "r") as f:
        return [json.loads(line.strip()) for line in f.readlines()]

def batch_entailment(pairs: List[tuple], threshold: float = 0.85) -> List[bool]:
    """
    Manual batch NLI inference using RoBERTa-large-MNLI.
    """
    results = []
    batch_size = 8
    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            premises, hypotheses = zip(*batch)

            encodings = tokenizer.batch_encode_plus(
                list(zip(premises, hypotheses)),
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors='pt'
            )

            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            entail_probs = probs[:, 2]  # ENT label is index 2 in RoBERTa-MNLI

            for prob in entail_probs:
                results.append(prob.item() >= threshold)

    return results

def build_nli_graph(propositions: List[str]) -> Dict[str, List[str]]:
    """Construct adjacency matrix using batch entailment calls."""
    matrix = {p: [] for p in propositions}
    pairs = []
    index_map = []

    for i, p_i in enumerate(propositions):
        for j, p_j in enumerate(propositions):
            if i != j:
                pairs.append((p_i, p_j))
                index_map.append((p_i, p_j))

    try:
        entailment_flags = batch_entailment(pairs)
    except Exception as e:
        print("[NLI ERROR] Skipping entry due to failure in entailment batch.")
        return matrix  # Return empty matrix (no edges)

    for idx, (p_i, p_j) in enumerate(index_map):
        if entailment_flags[idx]:
            matrix[p_i].append(p_j)

    return matrix

def evaluate_cot_metrics(entry: Dict, coherence_model=None) -> Dict[str, float]:
    explanation = entry.get("cot_explanation", "")
    steps = [s.strip() for s in explanation.split("\n") if s.strip()]
    if len(steps) < 2:
        return {"redundancy": 0.0, "weak_relevance": 0.0, "strong_relevance": 0.0, "coherence": -1.0}

    propositions = steps
    y_hat = steps[-1]

    matrix = build_nli_graph(propositions)

    metrics = {
        "redundancy": compute_redundancy(matrix, propositions, y_hat),
        "weak_relevance": compute_weak_relevance(matrix, propositions, y_hat),
        "strong_relevance": compute_strong_relevance(matrix, propositions, y_hat)
    }

    if coherence_model:
        try:
            metrics["coherence"] = coherence_model.coherence_metric(steps, y_hat)
        except Exception:
            metrics["coherence"] = -1.0
    else:
        metrics["coherence"] = -1.0

    return metrics

def evaluate_argllm_metrics(entry: Dict) -> Dict[str, float]:
    base_bag = entry.get("base", {}).get("bag", {})
    args = list(base_bag.get("arguments", {}).keys())
    attacks = base_bag.get("attacks", [])
    supports = base_bag.get("supports", [])

    y_hat = ["db0"] if "db0" in args else []

    return {
        "circularity": compute_circularity(args, attacks, supports),
        "acceptability": compute_dialectical_acceptability(args, attacks, y_hat)
    }

def evaluate_all_cot(filepath: str) -> List[Dict]:
    data = load_jsonl(filepath)
    coherence_model = None
    results = []
    for item in data:
        try:
            result = dict(q=item["question"], **evaluate_cot_metrics(item, coherence_model))
            results.append(result)
        except Exception as e:
            print(f"[WARN] Skipped an entry due to error: {e}")
    return results

def evaluate_all_argllm(filepath: str) -> List[Dict]:
    data = load_jsonl(filepath)
    return [dict(q=item["question"], **evaluate_argllm_metrics(item)) for item in data]
