import json
import csv
import glob
import os
import re
from collections import defaultdict

DATASETS = ["Truthfulclaim", "Strategyclaim", "Medclaim"]
MODELS   = ["Mistral", "Llama", "Qwen"]
SEARCH_ROOT = "/vol/bitbucket/rc1124/MSc_Individual_Project/ExplainabilityInLLMs-MScThesis/results/generation/"
OUT_CSV = "/vol/bitbucket/rc1124/MSc_Individual_Project/ExplainabilityInLLMs-MScThesis/results/generation/cot_method/cot_accuracy_by_dataset_model.csv"
PREFERRED_SUFFIX = "_with_preds.jsonl"


def find_candidate_files(dataset: str, model: str) -> list[str]:
    """
    Finds matching JSONL files under root directory

    @params: 
        dataset name, model name 
        
    @returns: list of candidate file paths
    """
    dl = dataset.lower()
    ml = model.lower()
    candidates = glob.glob(os.path.join(SEARCH_ROOT, "**", "*.jsonl"), recursive=True)
    filtered = []
    for path in candidates:
        low = path.lower()
        if dl in low and ml in low:
            filtered.append(path)

    def score(p):
        base = os.path.basename(p).lower()
        pref = base.endswith(PREFERRED_SUFFIX)
        return (0 if pref else 1, len(p))
    filtered.sort(key=score)
    return filtered


def compute_parsed_accuracy(jsonl_path: str):
    """
    Computes accuracy over parsed files

    @params: path to JSONL files 
    
    @returns: accuracy, correct count, incorrect count, unknown count, total count
    """
    correct = incorrect = unknown = total = 0
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            rec = json.loads(line)
            pred = rec.get("pred_label", None)
            is_corr = rec.get("is_correct", None)
            if pred is None or is_corr is None:
                unknown += 1
            else:
                if is_corr:
                    correct += 1
                else:
                    incorrect += 1
    parsed = correct + incorrect
    acc = (correct / parsed) if parsed else None
    return acc, correct, incorrect, unknown, total


def format_acc(acc):
    """
    Formats accuracy scores or returns empty string
    """
    return "" if acc is None else f"{acc:.4f}"


def main():
    """
    Builds dataset+model CoT accuracy table and writes CSV.
    """
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    table = defaultdict(dict)
    missing = []
    low_coverage_notes = []

    for ds in DATASETS:
        for md in MODELS:
            files = find_candidate_files(ds, md)
            if not files:
                missing.append((ds, md))
                table[ds][md] = None
                continue

            chosen = files[0]
            acc, c, ic, unk, tot = compute_parsed_accuracy(chosen)
            table[ds][md] = acc

            parsed = c + ic
            if tot > 0 and parsed / tot < 0.4:
                low_coverage_notes.append(f"{ds}-{md}: parsed {parsed}/{tot} ({parsed/tot:.1%}) -> {os.path.relpath(chosen)}")

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + MODELS)
        for ds in DATASETS:
            row = [ds] + [format_acc(table[ds].get(md)) for md in MODELS]
            writer.writerow(row)

    print(f"Wrote CSV to: {OUT_CSV}")
    if missing:
        print("\nMissing (dataset, model) pairs (no matching files found):")
        for ds, md in missing:
            print(f"  - {ds:>14s} × {md:<8s}")
    if low_coverage_notes:
        print("\nLow parsed coverage warnings (parsed/total < 40%):")
        for note in low_coverage_notes:
            print("  -", note)


if __name__ == "__main__":
    main()
