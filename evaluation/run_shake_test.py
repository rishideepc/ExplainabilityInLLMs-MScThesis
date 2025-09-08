import os
os.environ["HF_HOME"] = "/vol/bitbucket/rc1124/hf_cache"
os.environ["HF_TOKEN"] = "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"

import csv
import json
from typing import Tuple
import pandas as pd

from shake_pipeline import ShakePipeline
from attention_perturbation_llama import run_with_attention_perturbation

VISUALIZE_ATTENTION = True
MAX_SAMPLES = 5

CSV_PATH = "shake_score_results_truthfulclaim_llama_temp.csv"
HTML_PATH = "shake_score_results_truthfulclaim_llama_temp.html"
FREEZE_LEFT_COLS = 3

DATA_JSON_PATH = "/vol/bitbucket/rc1124/MSc_Individual_Project/ExplainabilityInLLMs-MScThesis/generators/truthful_claims_dataset.json"

def compute_bfs(label_orig: str, label_pert: str) -> float:
    """
    Computes the Binary Flip Score (BFS)
    
    @params: original (unperturbed) and perturbed labels 
    
    @returns: 1.0 if TRUE flips to FALSE (or vice versa); else 0.0
    """
    lo = (label_orig or "").upper()
    lp = (label_pert or "").upper()
    if lo in ("TRUE", "FALSE") and lp in ("TRUE", "FALSE") and lo != lp:
        return 1.0
    return 0.0

def build_html(df: pd.DataFrame, html_path: str, freeze_left_cols: int = 2, default_order_col: str = "bfs"):
    """
    Converts Pandas DataFrame to interactive HTML tables
    
    @params: DataFrame, html path, number of left columns to freeze, default column to order by
    
    @returns: saved HTML file
    """
    table_id = "shakeTable"
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
        label_orig = pipeline.get_label(claim)
        rationale_tokens = pipeline.generate_rationale_tokens(claim)

        label_pert = run_with_attention_perturbation(
            pipeline.model,
            pipeline.tokenizer,
            claim,
            rationale_tokens,
            visualize=VISUALIZE_ATTENTION,
            save_path=f"attention_sample_{idx+1}.png" if VISUALIZE_ATTENTION else None
        )

        bfs = compute_bfs(label_orig, label_pert)
        print(f"BFS (pure flip): {bfs:.4f}")

        results.append({
            "sample_id": sample["id"],
            "claim": claim,
            "label_orig": label_orig,
            "label_pert": label_pert,
            "ground_truth": ground_truth,
            "rationale_tokens": " ".join(rationale_tokens),
            "bfs": round(bfs, 4),
        })

    except Exception as e:
        print(f"Error on sample {idx+1}: {e}")
        continue

if results:
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "sample_id","claim","label_orig","label_pert","ground_truth","rationale_tokens","bfs"
        ])
        writer.writeheader()
        writer.writerows(results)
    print(f"SHAKE test results saved to: {CSV_PATH}")

    df = pd.DataFrame(results)
    preferred_order = [
        "sample_id","claim","ground_truth",
        "label_orig","label_pert","bfs","rationale_tokens"
    ]
    cols = [c for c in preferred_order if c in df.columns] + [c for c in df.columns if c not in preferred_order]
    df = df[cols]
    build_html(df, HTML_PATH, freeze_left_cols=FREEZE_LEFT_COLS, default_order_col="bfs")
else:
    print("No results were generated due to errors in all samples.")
