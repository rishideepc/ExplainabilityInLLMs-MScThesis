# evaluation/run_shake_test.py

import os
os.environ["HF_HOME"] = "/vol/bitbucket/rc1124/hf_cache"
os.environ["HF_TOKEN"] = "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"

import csv
import json
from typing import Tuple
import pandas as pd

from shake_pipeline import ShakePipeline
from attention_perturbation_llama import run_with_attention_perturbation
from shake_score import compute_shake_score

# =========================
# Configuration
# =========================
VISUALIZE_ATTENTION = False  # Set to True to enable visualization for specific samples
VISUALIZE_SAMPLE_IDS = [4]   # Which samples to visualize (1-indexed)
MAX_SAMPLES = 50             # For quick runs; set to 817 for full dataset
MAX_K = None                 # None = use all available rationale tokens; or set an int cap (e.g., 4)

CSV_PATH = "shake_score_results_truthfulclaim.csv"
HTML_PATH = "shake_score_results_truthfulclaim.html"
FREEZE_LEFT_COLS = 3         # Number of left columns to freeze in the HTML table

DATA_JSON_PATH = "/vol/bitbucket/rc1124/MSc_Individual_Project/ExplainabilityInLLMs-MScThesis/generators/truthful_claims_dataset.json"

# =========================
# Helpers (flip-only metrics)
# =========================
def compute_bfs(label_orig: str, label_pert: str) -> float:
    """
    Binary Flip Score (BFS): 1.0 if label flips TRUE<->FALSE, else 0.0.
    'UNKNOWN' is ignored (treated as no flip).
    """
    lo = (label_orig or "").upper()
    lp = (label_pert or "").upper()
    if lo in ("TRUE", "FALSE") and lp in ("TRUE", "FALSE") and lo != lp:
        return 1.0
    return 0.0

def compute_mbf_norm(
    label_orig: str,
    model,
    tokenizer,
    claim: str,
    rationale_tokens: list,
    max_k: int | None = None
) -> Tuple[float, int, int]:
    """
    Normalized MBF (flip-only):
      - Iterate k = 1..K (K = len(rationale_tokens) or max_k cap).
      - Perturb using the FIRST k rationale tokens.
      - Let k* be the smallest k that flips TRUE<->FALSE.
      - Return:
          score = 1 - (k* - 1)/K    (in (1/K .. 1]) if a flip occurs
                  0.0               (if no flip occurs)
          k_star = k* (0 if no flip)
          K_used = K

    Notes:
      - 'UNKNOWN' is ignored (does not count as a flip).
      - Requires label_orig to be 'TRUE' or 'FALSE' for flip logic.
    """
    if not rationale_tokens:
        return 0.0, 0, 0

    lo = (label_orig or "").upper()
    if lo not in ("TRUE", "FALSE"):
        return 0.0, 0, 0

    K = len(rationale_tokens) if max_k is None else min(len(rationale_tokens), max_k)
    if K <= 0:
        return 0.0, 0, 0

    for k in range(1, K + 1):
        subset = rationale_tokens[:k]
        label_k, _ = run_with_attention_perturbation(
            model,
            tokenizer,
            claim,
            subset,
            visualize=False,
            save_path=None
        )
        lk = (label_k or "").upper()
        if lk in ("TRUE", "FALSE") and lk != lo:
            # First k that flips
            score = 1.0 - (k - 1) / K
            return float(score), k, K

    # No flip observed
    return 0.0, 0, K

def build_html(df: pd.DataFrame, html_path: str, freeze_left_cols: int = 2, default_order_col: str = "mbf_norm"):
    """
    Write an interactive HTML report using DataTables (CDN) with:
      - Sort, search, pagination
      - Column hide/show (Buttons -> colvis)
      - Column reordering (drag & drop)
      - Frozen left columns (FixedColumns)
      - Horizontal scroll (for many columns)
    """
    table_id = "shakeTable"
    # Render the base HTML table
    table_html = df.to_html(index=False, table_id=table_id, classes="display compact nowrap")

    # Determine default order column index (fallback to first column if not found)
    try:
        order_idx = df.columns.get_loc(default_order_col)
    except Exception:
        order_idx = 0

    # Minimal CSS to make it nice
    css = """
    <style>
      body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 20px; }
      h1 { font-size: 20px; margin-bottom: 12px; }
      .dt-buttons { margin-bottom: 8px; }
      table.dataTable tbody th, table.dataTable tbody td { white-space: nowrap; }
      .dataTables_wrapper .dataTables_scroll div.dataTables_scrollBody { border: 1px solid #ddd; }
    </style>
    """

    # Full HTML with DataTables + extensions via CDN
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>SHAKE Results</title>
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
  <h1>SHAKE Results (interactive)</h1>
  <p>
    • Drag columns to reorder (left-most <b>{freeze_left_cols}</b> are frozen).<br/>
    • Use <b>Column visibility</b> button to hide/show columns.<br/>
    • Sort by clicking column headers; use the search box for filtering.
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
    print(f"Interactive HTML report saved to: {html_path}")

# =========================
# Main
# =========================
with open(DATA_JSON_PATH, "r") as f:
    data = json.load(f)

samples = data[:MAX_SAMPLES]
pipeline = ShakePipeline()
results = []

for idx, sample in enumerate(samples):
    claim = sample["claim"]
    ground_truth = str(sample["label"]).upper()
    print("\n" + "="*80)
    print(f"🔬 Sample {idx+1}: {claim}")

    try:
        # Base (unperturbed) prediction
        label_orig, conf_orig = pipeline.get_label_and_confidence(claim)

        # Rationale token extraction
        rationale_tokens = pipeline.generate_rationale_tokens(claim)

        # Should we visualize the attention for this sample?
        # should_visualize = VISUALIZE_ATTENTION and (idx + 1) in VISUALIZE_SAMPLE_IDS
        should_visualize = VISUALIZE_ATTENTION
        save_path = f"attention_sample_{idx+1}.png" if should_visualize else None

        # Perturbed prediction using ALL rationale tokens (kept for continuity with your current CSV)
        label_pert, conf_pert = run_with_attention_perturbation(
            pipeline.model,
            pipeline.tokenizer,
            claim,
            rationale_tokens,
            visualize=should_visualize,
            save_path=save_path
        )

        # Existing SHAKE score (unchanged; you can ignore later if you adopt pure flip metrics)
        shake_score = compute_shake_score(label_orig, conf_orig, label_pert, conf_pert, ground_truth)
        print(f"Final SHAKE_SCORE = {shake_score:.4f}")

        # New: BFS (pure flip)
        bfs = compute_bfs(label_orig, label_pert)
        print(f"BFS (pure flip): {bfs:.4f}")

        # New: MBF (normalized)
        mbf_norm, mbf_k_star, mbf_K = compute_mbf_norm(
            label_orig=label_orig,
            model=pipeline.model,
            tokenizer=pipeline.tokenizer,
            claim=claim,
            rationale_tokens=rationale_tokens,
            max_k=MAX_K
        )
        if mbf_k_star > 0:
            print(f"MBF_norm: {mbf_norm:.4f} (k*={mbf_k_star}, K={mbf_K})")
        else:
            print(f"MBF_norm: {mbf_norm:.4f} (no flip, K={mbf_K})")

        # Record
        results.append({
            "sample_id": sample["id"],
            "claim": claim,
            "label_orig": label_orig,
            "conf_orig": round(conf_orig, 3),
            "label_pert": label_pert,
            "conf_pert": round(conf_pert, 3),
            "shake_score": round(shake_score, 4),
            "ground_truth": ground_truth,
            # Audit fields
            "rationale_tokens": " ".join(rationale_tokens),
            # New metrics
            "bfs": round(bfs, 4),
            "mbf_norm": round(mbf_norm, 4),
            "mbf_k_star": mbf_k_star,
            "mbf_K": mbf_K,
        })

    except Exception as e:
        print(f"Error on sample {idx+1}: {e}")
        continue

# =========================
# Output: CSV + HTML
# =========================
if results:
    # CSV
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"SHAKE test results saved to: {CSV_PATH}")

    # HTML
    df = pd.DataFrame(results)
    # Optional: set nicer column order (freeze-left columns show first)
    preferred_order = [
        "sample_id", "claim", "ground_truth",
        "label_orig", "label_pert", "bfs",
        "mbf_norm", "mbf_k_star", "mbf_K",
        "conf_orig", "conf_pert", "shake_score",
        "rationale_tokens"
    ]
    cols = [c for c in preferred_order if c in df.columns] + [c for c in df.columns if c not in preferred_order]
    df = df[cols]
    build_html(df, HTML_PATH, freeze_left_cols=FREEZE_LEFT_COLS, default_order_col="mbf_norm")
else:
    print("No results were generated due to errors in all samples.")
