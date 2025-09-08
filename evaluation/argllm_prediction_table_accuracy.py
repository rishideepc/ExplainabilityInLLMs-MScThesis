import json, csv, glob, os, re
from collections import defaultdict

DATASETS = ["Truthfulclaim", "Strategyclaim", "Medclaim"]
MODEL_ALIASES = {
    "Mistral": ["mistral"],
    "Llama":   ["llama", "meta-llama", "llama3", "llama-3"],
    "Qwen":    ["qwen", "qwen2", "qwen3"],
}
SEARCH_ROOTS = [
    "/vol/bitbucket/rc1124/MSc_Individual_Project/ExplainabilityInLLMs-MScThesis/results/generation/argLLM_generation"
]
OUT_CSV = "/vol/bitbucket/rc1124/MSc_Individual_Project/ExplainabilityInLLMs-MScThesis/results/generation/argLLM_generation/argLLM_generation/cv_argllm_accuracy_by_dataset_model.csv"
PREFERRED_SUFFIX = "_with_preds.jsonl"
EST_THRESHOLD = 0.5
BASE_THRESHOLD = 0.5


def find_all_jsonl(search_roots):
    """
    Helper function; finds all JSONL files under given root directories

    @params: list of root directories 
    
    @returns: list of relevant JSONL file paths
    """
    files = []
    for root in search_roots:
        root_abs = os.path.abspath(root)
        hits = glob.glob(os.path.join(root_abs, "**", "*.jsonl"), recursive=True)
        print(f"[scan] {root_abs} -> {len(hits)} .jsonl files")
        files.extend(hits)
    seen, uniq = set(), []
    for p in files:
        if p not in seen:
            uniq.append(p); seen.add(p)
    return uniq


def matches_token(path: str, token_aliases: list[str]) -> bool:
    """
    Helper function; checks for aliases in paths

    @params: path string, list of aliases 

    @returns: True/False
    """
    low = path.lower()
    low = low.replace("ollama", "")
    segments = re.split(r"[\\/._\-\s]+", low)
    segset = set(s for s in segments if s)

    for alias in token_aliases:
        a = alias.lower()
        if a in segset:
            return True
        if re.search(rf"(?<!o){re.escape(a)}(?![a-z0-9])", low):
            return True
    return False


def filter_candidates(all_files, dataset: str, model: str):
    """
    Helper function; filters files that match dataset and model tokens

    @params: list of files, dataset name, model name
    
    @returns: sorted list of candidates
    """
    dl = dataset.lower()
    ds_aliases = [dl]
    model_aliases = MODEL_ALIASES[model]

    cands = []
    for p in all_files:
        if matches_token(p, ds_aliases) and matches_token(p, model_aliases):
            cands.append(p)

    if not cands:
        return []

    def score(path):
        base = os.path.basename(path).lower()
        prefers = base.endswith(PREFERRED_SUFFIX)
        return (0 if prefers else 1, len(path))
    cands.sort(key=score)
    return cands


def to_label_from_strength(x, tau):
    """
    Helper function; thresholds a numeric strength into a 'true'/'false' label

    @params: value x (any), threshold tau

    @returns: 'true'/'false'/None labels
    """
    if x is None: return None
    try:
        return "true" if float(x) >= tau else "false"
    except Exception:
        return None


def compute_accuracy(jsonl_path):
    """
    Computes accuracy for the JSONL file

    @params: path to JSONL 
    
    @return: accuracy score, number of correct predictions, number of incorrect predictions, number of unknown predictions, 
             total, number of parsed, source used)
    """
    correct = incorrect = unknown = total = 0
    used_sources = set()

    with open(jsonl_path, "r") as f:
        for line in f:
            s = line.strip()
            if not s: continue
            total += 1
            rec = json.loads(s)

            gt = str(rec.get("label", "")).lower()

            pred = rec.get("est_pred_label")
            if pred is not None:
                used_sources.add("est_label")
            else:
                pred = to_label_from_strength(rec.get("est_pred_strength"), EST_THRESHOLD)
                if pred is not None:
                    used_sources.add("est_strength")
                else:
                    pred = rec.get("base_pred_label")
                    if pred is not None:
                        used_sources.add("base_label")
                    else:
                        pred = to_label_from_strength(rec.get("base_pred_strength"), BASE_THRESHOLD)
                        if pred is not None:
                            used_sources.add("base_strength")

            if pred is None:
                unknown += 1
            else:
                if pred == gt:
                    correct += 1
                else:
                    incorrect += 1

    parsed = correct + incorrect
    acc = (correct / parsed) if parsed else None

    if not used_sources:
        src = "none"
    elif len(used_sources) == 1:
        src = next(iter(used_sources))
    else:
        src = "mixed"

    return acc, correct, incorrect, unknown, total, parsed, src


def main():
    """
    Builds dataset+model accuracy table from JSONLs and writes CSV
    """
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    all_jsonls = find_all_jsonl(SEARCH_ROOTS)
    print(f"[info] total unique .jsonl considered: {len(all_jsonls)}")

    table = defaultdict(dict)
    missing = []
    low_cov = []

    for ds in DATASETS:
        for md in MODEL_ALIASES.keys():
            cands = filter_candidates(all_jsonls, ds, md)
            if not cands:
                missing.append((ds, md))
                table[ds][md] = None
                continue

            chosen = cands[0]
            acc, c, ic, unk, tot, parsed, src = compute_accuracy(chosen)
            table[ds][md] = acc

            print(f"[pick] {ds:>13s} × {md:<7s} -> {os.path.relpath(chosen)} | "
                  f"acc={'' if acc is None else f'{acc:.4f}'} parsed={parsed}/{tot} unk={unk} src={src}")

            if tot and parsed / tot < 0.4:
                low_cov.append(f"{ds}-{md}: parsed {parsed}/{tot} ({parsed/tot:.1%}) -> {os.path.relpath(chosen)}")

    models = list(MODEL_ALIASES.keys())
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + models)
        for ds in DATASETS:
            row = [ds] + [("" if table[ds][m] is None else f"{table[ds][m]:.4f}") for m in models]
            w.writerow(row)

    print("\nWrote CSV to:", OUT_CSV)
    if missing:
        print("\nMissing (dataset × model) pairs (no safe match under provided roots):")
        for ds, md in missing:
            print(f"  - {ds:>13s} × {md:<7s}")
    if low_cov:
        print("\nLow parsed coverage warnings (parsed/total < 40%):")
        for note in low_cov:
            print("  -", note)


if __name__ == "__main__":
    main()
