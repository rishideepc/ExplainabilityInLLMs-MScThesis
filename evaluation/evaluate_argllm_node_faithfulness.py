# evaluate_argllm_node_faithfulness.py
"""
CLI to compute LOO node faithfulness for ArgLLM outputs.

Usage:
  python scripts/evaluate_argllm_node_faithfulness.py \
      --inputs results/generation/argLLM_generation/truthfulclaim_mistral_temp_2/argllm_outputs_ollama.jsonl \
      --which estimated \
      --out_csv results/analysis/node_faithfulness_loo.csv \
      --out_jsonl results/analysis/node_faithfulness_loo_details.jsonl \
      --out_html results/analysis/node_faithfulness_loo.html \
      --out_pkl  results/analysis/node_faithfulness_loo.pkl
"""

from __future__ import annotations
import argparse
import json
import os
from typing import Any, Dict, Optional, List
import sys
import csv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)
from evaluation.argllm_node_faithfulness import evaluate_bag_loo

def _pick_bag(sample: Dict[str, Any], which: str) -> Optional[Dict[str, Any]]:
    if which == "base":
        return ((sample.get("base") or {}).get("bag") or None)
    elif which == "estimated":
        return ((sample.get("estimated") or {}).get("bag") or None)
    elif which == "both":
        return ((sample.get("estimated") or {}).get("bag")) or ((sample.get("base") or {}).get("bag") or None)
    else:
        return None

def _generate_html(out_html: str, columns: List[str], rows: List[List[Any]]) -> None:
    """
    Build an interactive HTML table (DataTables + Buttons + FixedHeader).
    """
    import html
    import json as _json

    # Embedding data as JSON to avoid filesystem restrictions
    data_json = _json.dumps(rows, ensure_ascii=False)
    columns_json = _json.dumps(columns, ensure_ascii=False)

    html_str = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>ArgLLM LOO Node Faithfulness</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>

  <!-- DataTables CSS/JS via CDN -->
  <link rel="stylesheet" href="https://cdn.datatables.net/1.13.8/css/jquery.dataTables.min.css"/>
  <link rel="stylesheet" href="https://cdn.datatables.net/buttons/2.4.2/css/buttons.dataTables.min.css"/>
  <link rel="stylesheet" href="https://cdn.datatables.net/fixedheader/3.4.0/css/fixedHeader.dataTables.min.css"/>

  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 20px; }}
    table.dataTable thead th {{ position: sticky; top: 0; background: #fff; }}
    .dt-buttons {{ margin-bottom: 10px; }}
    .container {{ max-width: 98vw; overflow-x: auto; }}
  </style>
</head>
<body>
  <h2>ArgLLM LOO Node Faithfulness</h2>
  <p>Search, sort, hide columns (Column visibility), and export. Header is sticky. </p>
  <div class="container">
    <table id="tbl" class="display compact" style="width:100%"></table>
  </div>

  <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
  <script src="https://cdn.datatables.net/1.13.8/js/jquery.dataTables.min.js"></script>
  <script src="https://cdn.datatables.net/buttons/2.4.2/js/dataTables.buttons.min.js"></script>
  <script src="https://cdn.datatables.net/buttons/2.4.2/js/buttons.colVis.min.js"></script>
  <script src="https://cdn.datatables.net/buttons/2.4.2/js/buttons.html5.min.js"></script>
  <script src="https://cdn.datatables.net/fixedheader/3.4.0/js/dataTables.fixedHeader.min.js"></script>

  <script>
    const columns = {columns_json}.map(c => ({{ title: c }}));
    const data = {data_json};

    $(document).ready(function() {{
      $('#tbl').DataTable({{
        data: data,
        columns: columns,
        dom: 'Bfrtip',
        buttons: [
          'colvis', 'copyHtml5', 'csvHtml5'
        ],
        fixedHeader: true,
        pageLength: 25,
        scrollX: true
      }});
    }});
  </script>
</body>
</html>
"""
    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    with open(out_html, "w", encoding="utf-8") as fh:
        fh.write(html_str)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help="Path to ArgLLM JSONL with 'base'/'estimated' bags.")
    ap.add_argument("--which", default="estimated", choices=["base", "estimated", "both"],
                    help="Which bag to use for evaluation.")
    ap.add_argument("--out_csv", required=True, help="CSV file for per-node metrics.")
    ap.add_argument("--out_jsonl", default=None, help="Optional JSONL with per-sample details.")
    ap.add_argument("--out_html", default=None, help="Optional HTML interactive report.")
    ap.add_argument("--out_pkl", default=None, help="Optional pandas DataFrame pickle (.pkl).")
    ap.add_argument("--claim_id", default="db0")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    if args.out_jsonl:
        os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    if args.out_html:
        os.makedirs(os.path.dirname(args.out_html), exist_ok=True)
    if args.out_pkl:
        os.makedirs(os.path.dirname(args.out_pkl), exist_ok=True)

    n_samples = 0
    n_nodes = 0

    # CSV header (NEW: add loo_label_after_ablation)
    columns = [
        "sample_idx","id","question","claim","label",
        "bag_type","baseline_claim_strength","baseline_label",
        "arg_id","role","tau","arg_baseline_strength",
        "loo_claim_strength","loo_label_after_ablation","claim_strength_delta",
        "node_influence_aligned","label_flip",
        "num_args","num_attacks","num_supports"
    ]

    with open(args.out_csv, "w", encoding="utf-8", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(columns)

    if args.out_jsonl:
        fout_jsonl = open(args.out_jsonl, "w", encoding="utf-8")
    else:
        fout_jsonl = None

    all_rows_for_html: List[List[Any]] = []

    with open(args.inputs, "r", encoding="utf-8") as fin, open(args.out_csv, "a", encoding="utf-8", newline="") as fcsv:
        writer = csv.writer(fcsv)
        for idx, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except Exception:
                continue

            bag = _pick_bag(sample, args.which)
            bag_type = args.which
            if bag is None and args.which == "both":
                continue
            if bag is None:
                continue

            num_args = len((bag.get("arguments") or {}))
            num_attacks = len((bag.get("attacks") or []))
            num_supports = len((bag.get("supports") or []))

            res = evaluate_bag_loo(bag, claim_id=args.claim_id, threshold=args.threshold)
            s0 = res["baseline_strength"]
            y0 = res["baseline_label"]

            if fout_jsonl is not None:
                out_detail = {
                    "meta": {
                        "sample_idx": idx,
                        "id": sample.get("id"),
                        "bag_type": bag_type,
                        "num_args": num_args,
                        "num_attacks": num_attacks,
                        "num_supports": num_supports
                    },
                    "input": {
                        "question": sample.get("question"),
                        "claim": sample.get("claim"),
                        "label": sample.get("label"),
                    },
                    "baseline": {
                        "claim_strength": s0,
                        "claim_label": y0
                    },
                    "nodes": res["nodes"]
                }
                fout_jsonl.write(json.dumps(out_detail) + "\n")

            for arg_id, node_metrics in res["nodes"].items():
                row = [
                    idx,
                    sample.get("id"),
                    sample.get("question"),
                    sample.get("claim"),
                    sample.get("label"),
                    bag_type,
                    f"{s0:.6f}",
                    y0,
                    arg_id,
                    node_metrics["role"],
                    f"{node_metrics['tau']:.6f}",
                    f"{node_metrics['baseline_strength']:.6f}",
                    f"{node_metrics['loo_claim_strength']:.6f}",
                    node_metrics["loo_label"],  # NEW
                    f"{node_metrics['claim_strength_delta']:.6f}",
                    f"{node_metrics['node_influence_aligned']:.6f}",
                    node_metrics["label_flip"],
                    num_args,
                    num_attacks,
                    num_supports
                ]
                writer.writerow(row)
                all_rows_for_html.append(row)
                n_nodes += 1

            n_samples += 1

    if fout_jsonl is not None:
        fout_jsonl.close()

    # Optional: pandas pickle for immediate analysis
    if args.out_pkl:
        try:
            import pandas as pd
            df = pd.DataFrame(all_rows_for_html, columns=columns)
            df.to_pickle(args.out_pkl)
            print(f"[LOO] Wrote DataFrame pickle to {args.out_pkl}")
        except Exception as e:
            print(f"[WARN] Could not write pandas pickle: {e}")

    # Optional: interactive HTML
    if args.out_html:
        _generate_html(args.out_html, columns, all_rows_for_html)
        print(f"[LOO] Wrote interactive HTML report to {args.out_html}")

    print(f"[LOO] Processed {n_samples} samples; wrote {n_nodes} node rows to {args.out_csv}")
    if args.out_jsonl:
        print(f"[LOO] Wrote detailed per-sample JSONL to {args.out_jsonl}")

if __name__ == "__main__":
    main()
