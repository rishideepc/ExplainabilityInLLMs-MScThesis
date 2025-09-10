import os
os.environ["HF_HOME"] = "/vol/bitbucket/rc1124/hf_cache"

import csv
import json
import random
import argparse
import pandas as pd
import torch

from shake_expl_pipeline import ShakeExplPipeline
from expl_attention_perturbation_llama import (
    run_with_attention_perturbation_on_prompt,
    logits_label_on_last_token
)

DEFAULT_DATA_JSON_PATH = os.environ.get(
    "SHAKE_DATA_JSON",
    "/vol/bitbucket/rc1124/MSc_Individual_Project/ExplainabilityInLLMs-MScThesis/generators/truthful_claims_dataset.json"
)
DEFAULT_CSV_PATH  = os.environ.get("SHAKE_OUT_CSV",  "shake_expl_results_llama.csv")
DEFAULT_HTML_PATH = os.environ.get("SHAKE_OUT_HTML", "shake_expl_results_llama.html")
DEFAULT_LAST_N_LAYERS = int(os.environ.get("SHAKE_LAST_N_LAYERS", "12"))
DEFAULT_MAX_KW = int(os.environ.get("SHAKE_MAX_KW", "32"))
DEFAULT_MAX_SAMPLES = os.environ.get("SHAKE_MAX_SAMPLES")
FREEZE_LEFT_COLS = 3

def compute_bfs(label_base: str, label_pert: str) -> float:
    """
    Computes the post-perturbation Binary Flip Score (BFS)

    @params: 
        baseline (unperturbed) label, perturbed label (strings 'TRUE' or 'FALSE')
        
    @returns: 
        1.0 if flipped; else 0.0
    """
    lo = (label_base or "").upper()
    lp = (label_pert or "").upper()
    if lo in ("TRUE", "FALSE") and lp in ("TRUE", "FALSE") and lo != lp:
        return 1.0
    return 0.0

def build_html(df: pd.DataFrame, html_path: str, freeze_left_cols: int = 2, default_order_col: str = "bfs_expl"):
    """
    Converts a Pandas DataFrame to an interactive HTML table

    @params: 
        results DataFrame, output file path, number of left columns to freeze, default column to sort by
    
    @returns: 
        HTML file with interactive table, written to disk
    """
    table_id = "shakeExplTable"
    table_html = df.to_html(index=False, table_id=table_id, classes="display compact nowrap")
    try:
        order_idx = df.columns.get_loc(default_order_col)
    except Exception:
        order_idx = 0

    css = """
    <style>
      body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 20px; }
      h1 { font-size: 20px; margin-bottom: 12px; }
      .dt-buttons { margin-bottom: 8px; }
      table.dataTable tbody th, table.dataTable tbody td { white-space: nowrap; }
      .dataTables_wrapper .dataTables_scroll div.dataTables_scrollBody { border: 1px solid #ddd; }
    </style>
    """

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>SHAKE Explanation-Ablation Results</title>
{css}
<link rel="stylesheet" href="https://cdn.datatables.net/1.13.8/css/jquery.dataTables.min.css">
<link rel="stylesheet" href="https://cdn.datatables.net/buttons/2.4.2/css/buttons.dataTables.min.css">
<link rel="stylesheet" href="https://cdn.datatables.net/fixedcolumns/4.3.0/css/fixedColumns.dataTables.min.css">
<link rel="stylesheet" href="https://cdn.datatables.net/colreorder/1.7.0/css/colReorder.dataTables.min.css">
<script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
<script src="https://cdn.datatables.net/1.13.8/js/jquery.dataTables.min.js"></script>
<script src="https://cdn.datatables.net/buttons/2.4.2/js/dataTables.buttons.min.js"></script>
<script src="https://cdn.datatables.net/buttons/2.4.2/js/buttons.colVis.min.js"></script>
<script src="https://cdn.datatables.net/fixedcolumns/4.3.0/js/dataTables.fixedColumns.min.js"></script>
<script src="https://cdn.datatables.net/colreorder/1.7.0/js/dataTables.colReorder.min.js"></script>
</head>
<body>
  <h1>SHAKE Explanation-Ablation Results (interactive)</h1>
  <p>
    • Left columns are frozen; drag to reorder; use Column visibility for toggles; sort and search as usual.
  </p>
  <div style="width: 100%; overflow: hidden;">
    {table_html}
  </div>
  <script>
    $(document).ready(function() {{
      var table = $('#{table_id}').DataTable({{
        scrollX: true,
        scrollY: '70vh',
        scrollCollapse: true,
        paging: true,
        pageLength: 25,
        lengthMenu: [[10, 25, 50, 100, -1], [10, 25, 50, 100, 'All']],
        colReorder: true,
        fixedColumns: {{
          leftColumns: {freeze_left_cols}
        }},
        dom: 'Bfrtip',
        buttons: [
          'colvis',
          'pageLength'
        ],
        order: [[{order_idx}, 'desc']]
      }});
    }});
  </script>
</body>
</html>
    """
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[HTML] Interactive report saved to: {html_path}")

def _pick_sample_id(sample, default_idx):
    """
    Chooses a sample ID depending on the dataset format

    @params: 
        sample dict, fallback index 
        
    @returns: 
        ID from common keys or fallback index
    """
    if "id" in sample:
        return sample["id"]
    if "qid" in sample:
        return sample["qid"]
    if "source_id" in sample:
        return sample["source_id"]
    return default_idx

def _pick_ground_truth(sample):
    """
    Normalizes ground-truth label to either 'TRUE' or 'FALSE'

    @params: 
        sample dict with varied label fields. 
        
    @returns: 
        normalized label string or empty string if not found
    """
    if "label/answer" in sample:
        val = sample["label/answer"]
        if isinstance(val, bool):
            return "TRUE" if val else "FALSE"
        return "TRUE" if str(val).lower() in {"true", "1"} else "FALSE"
    if "label" in sample:
        return "TRUE" if str(sample["label"]).lower() in {"true", "1"} else "FALSE"
    return ""

def _select_samples(data, mode: str, size: int, seed: int, max_cap: int | None):
    """
    Selects proper subsets from dataset for preliminary testing
    
    @params: 
        full data, mode ('all'/'random'), size, seed, optional cap 
        
    @returns: 
        selected list of samples (subset)
    """
    n = len(data)
    if mode == "random":
        k = min(size, n)
        rng = random.Random(seed)
        idxs = rng.sample(range(n), k)
        selected = [data[i] for i in idxs]
        print(f"[SAMPLE] mode=random | requested={size} available={n} -> selected={k} (seed={seed})")
    else:
        selected = list(data)
        print(f"[SAMPLE] mode=all | selected={n}")

    if max_cap is not None:
        cap = min(int(max_cap), len(selected))
        selected = selected[:cap]
        print(f"[SAMPLE] post-cap with MAX_SAMPLES={max_cap} -> using={cap}")

    return selected

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHAKE explanation-ablation runner with sampling toggle")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_JSON_PATH,
                        help="Path to dataset JSON (any of truthful/med/strategy/commonsense formats).")
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV_PATH, help="Output CSV path.")
    parser.add_argument("--html", type=str, default=DEFAULT_HTML_PATH, help="Output HTML path.")
    parser.add_argument("--sample-mode", choices=["all", "random"], default="all",
                        help="Use entire dataset or a random sample.")
    parser.add_argument("--sample-size", type=int, default=500,
                        help="Random sample size (only used when --sample-mode=random).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--last-n-layers", type=int, default=DEFAULT_LAST_N_LAYERS,
                        help="How many final layers to perturb.")
    parser.add_argument("--max-kw", type=int, default=DEFAULT_MAX_KW,
                        help="Max #keywords to extract from explanation (K).")
    parser.add_argument("--max-samples", type=int, default=int(DEFAULT_MAX_SAMPLES) if DEFAULT_MAX_SAMPLES else None,
                        help="Optional cap after selection (keeps compatibility with previous env).")
    args = parser.parse_args()

    with open(args.data, "r") as f:
        data = json.load(f)

    samples = _select_samples(
        data=data,
        mode=args.sample_mode,
        size=args.sample_size,
        seed=args.seed,
        max_cap=args.max_samples
    )

    pipe = ShakeExplPipeline()
    tok  = pipe.tokenizer
    mdl  = pipe.model

    results = []

    for i, sample in enumerate(samples):
        claim = sample["claim"]
        gt    = _pick_ground_truth(sample)
        sid   = _pick_sample_id(sample, i)

        print("\n" + "="*100)
        print(f"🔬 Sample {i+1}/{len(samples)} | id={sid}")
        print(f"Claim: {claim}")

        try:
            expl = pipe.explain_only(claim)
            prompt_base = pipe.build_prompt_classify_with_expl(claim, expl)

            inputs = tok(prompt_base, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(mdl.device) for k, v in inputs.items()}
            for k in inputs:
                if inputs[k].dtype == torch.float:
                    inputs[k] = inputs[k].to(dtype=torch.float16)
            with torch.no_grad():
                out_base = mdl(**inputs, output_attentions=False)
            label_base, p_true_base, p_false_base = logits_label_on_last_token(tok, out_base.logits)

            print(f"[BASE] label={label_base} | p_true={p_true_base:.4f}")

            kws = pipe.keywords_from_expl(expl, max_k=args.max_kw)
            target_indices = pipe.indices_for_keywords_in_expl_span(prompt_base, kws)
            if not target_indices:
                print("[WARN] No target indices found inside explanation span. Mark as UNKNOWN perturb.")
                label_pert, p_true_pert = label_base, p_true_base
                bfs_expl = 0.0
            else:
                label_pert, p_true_pert, p_false_pert = run_with_attention_perturbation_on_prompt(
                    mdl, tok, prompt_base, target_indices, last_n_layers=args.last_n_layers
                )
                bfs_expl = compute_bfs(label_base, label_pert)

            print(f"[PERT] label={label_pert} | p_true={p_true_pert:.4f} | BFS_expl={bfs_expl:.1f}")

            results.append({
                "sample_id": sid,
                "claim": claim,
                "ground_truth": gt,
                "explanation": expl,
                "keywords_expl": " ".join(kws),
                "num_keywords": len(kws),
                "label_base_withE": label_base,
                "p_true_base": round(p_true_base, 6),
                "label_pert_withE": label_pert,
                "p_true_pert": round(p_true_pert, 6),
                "bfs_expl": bfs_expl,
                "target_indices": str(target_indices),
            })

        except Exception as e:
            print(f"[ERROR] Sample {i+1} id={sid} :: {e}")
            continue

    if results:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"[CSV] Saved: {args.csv}")

        df = pd.DataFrame(results)
        preferred = [
            "sample_id","claim","ground_truth","explanation","keywords_expl","num_keywords",
            "label_base_withE","p_true_base","label_pert_withE","p_true_pert",
            "bfs_expl","target_indices"
        ]
        cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
        df = df[cols]
        build_html(df, args.html, freeze_left_cols=FREEZE_LEFT_COLS, default_order_col="bfs_expl")
    else:
        print("[INFO] No results collected.")
