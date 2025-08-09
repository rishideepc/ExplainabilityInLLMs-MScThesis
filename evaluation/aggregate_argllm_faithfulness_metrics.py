# evaluation/aggregate_faithfulness_metrics.py
"""
Aggregate faithfulness metrics from node-level LOO CSV produced by
`evaluation/evaluate_argllm_node_faithfulness.py`.

Metrics:
  - Option 1 (binary): label_flip rate per sample = (# label flips) / (# nodes)
  - Option 2 (continuous):
        * mean_abs_delta = mean(|claim_strength_delta|)
        * norm_sum_abs_delta = (sum |claim_strength_delta|) / (N * max_possible_change)
          where max_possible_change = max(s0, 1 - s0), s0 = baseline claim strength (per sample)

Outputs:
  - Per-sample metrics CSV
  - Optional JSONL (per-sample)
  - Optional overall summary JSON
  - Optional HTML (interactive table)
  - Console summary

Usage:
  python evaluation/aggregate_faithfulness_metrics.py \
      --node_csv results/analysis/node_faithfulness_loo.csv \
      --out_samples_csv results/analysis/faithfulness_per_sample.csv \
      --out_summary_json results/analysis/faithfulness_summary.json \
      --out_html results/analysis/faithfulness_per_sample.html
"""

from __future__ import annotations
import argparse
import json
import os
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np

# Columns expected from node-level CSV
EXPECTED_COLUMNS = [
    "sample_idx","id","question","claim","label",
    "bag_type","baseline_claim_strength","baseline_label",
    "arg_id","role","tau","arg_baseline_strength",
    "loo_claim_strength","loo_label_after_ablation","claim_strength_delta",
    "node_influence_aligned","label_flip",
    "num_args","num_attacks","num_supports"
]

GROUP_KEYS = ["sample_idx", "id", "bag_type"]

def _safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default

def compute_per_sample_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group node-level rows into per-sample metrics.
    """
    # Ensure numeric types where needed
    df = df.copy()
    df["baseline_claim_strength"] = df["baseline_claim_strength"].apply(_safe_float)
    df["claim_strength_delta"] = df["claim_strength_delta"].apply(_safe_float)
    df["label_flip"] = pd.to_numeric(df["label_flip"], errors="coerce").fillna(0).astype(int)
    df["baseline_label"] = pd.to_numeric(df["baseline_label"], errors="coerce").fillna(0).astype(int)
    df["loo_label_after_ablation"] = pd.to_numeric(df["loo_label_after_ablation"], errors="coerce").fillna(0).astype(int)

    # For convenience: abs delta
    df["abs_delta"] = df["claim_strength_delta"].abs()

    # Aggregate function for a group (one sample)
    rows: List[Dict[str, Any]] = []
    for keys, grp in df.groupby(GROUP_KEYS, dropna=False):
        sample_idx, sid, bag_type = keys

        # invariant per sample (take first non-null)
        s0 = grp["baseline_claim_strength"].dropna().astype(float)
        s0 = float(s0.iloc[0]) if len(s0) else np.nan

        n_nodes = len(grp)
        flips = int(grp["label_flip"].sum())
        label_flip_rate = flips / n_nodes if n_nodes > 0 else np.nan

        mean_abs_delta = float(grp["abs_delta"].mean()) if n_nodes > 0 else np.nan
        sum_abs_delta = float(grp["abs_delta"].sum()) if n_nodes > 0 else np.nan

        # normalization: max possible change for this sample = max(s0, 1 - s0)
        max_possible_change = max(s0, 1.0 - s0) if not np.isnan(s0) else np.nan
        denom = n_nodes * max_possible_change if (n_nodes > 0 and not np.isnan(max_possible_change) and max_possible_change > 0) else np.nan
        norm_sum_abs_delta = (sum_abs_delta / denom) if denom and not np.isnan(denom) else np.nan

        # attach some context fields
        question = grp["question"].iloc[0] if "question" in grp.columns else None
        claim = grp["claim"].iloc[0] if "claim" in grp.columns else None
        label = grp["label"].iloc[0] if "label" in grp.columns else None
        baseline_label = int(grp["baseline_label"].iloc[0]) if "baseline_label" in grp.columns else None
        num_args = int(grp["num_args"].iloc[0]) if "num_args" in grp.columns else n_nodes
        num_attacks = int(grp["num_attacks"].iloc[0]) if "num_attacks" in grp.columns else None
        num_supports = int(grp["num_supports"].iloc[0]) if "num_supports" in grp.columns else None

        rows.append({
            "sample_idx": sample_idx,
            "id": sid,
            "bag_type": bag_type,
            "question": question,
            "claim": claim,
            "label": label,

            # context from graph
            "num_nodes": n_nodes,
            "num_args": num_args,
            "num_attacks": num_attacks,
            "num_supports": num_supports,

            # baseline strength/label
            "baseline_claim_strength": s0,
            "baseline_label": baseline_label,

            # Option 1 (binary)
            "metric_label_flip_rate": label_flip_rate,

            # Option 2 (continuous)
            "metric_mean_abs_delta": mean_abs_delta,
            "metric_sum_abs_delta": sum_abs_delta,
            "metric_norm_sum_abs_delta": norm_sum_abs_delta,   # preferred normalized continuous metric
            "max_possible_change_per_node": max_possible_change,
        })

    return pd.DataFrame(rows)

def compute_corpus_summary(df_samples: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute overall/corpus means and some breakouts.
    """
    summary: Dict[str, Any] = {}

    def _mean(col: str) -> float:
        return float(pd.to_numeric(df_samples[col], errors="coerce").dropna().mean()) if col in df_samples.columns else float("nan")

    summary["n_samples"] = int(len(df_samples))
    summary["mean_label_flip_rate"] = _mean("metric_label_flip_rate")
    summary["mean_mean_abs_delta"] = _mean("metric_mean_abs_delta")
    summary["mean_norm_sum_abs_delta"] = _mean("metric_norm_sum_abs_delta")

    # Breakout by bag_type (base/estimated/both)
    by_bag = {}
    for bag_type, g in df_samples.groupby("bag_type"):
        by_bag[bag_type] = {
            "n_samples": int(len(g)),
            "mean_label_flip_rate": float(pd.to_numeric(g["metric_label_flip_rate"], errors="coerce").dropna().mean()),
            "mean_mean_abs_delta": float(pd.to_numeric(g["metric_mean_abs_delta"], errors="coerce").dropna().mean()),
            "mean_norm_sum_abs_delta": float(pd.to_numeric(g["metric_norm_sum_abs_delta"], errors="coerce").dropna().mean()),
        }
    summary["by_bag_type"] = by_bag

    # Breakout by baseline_label (0/1)
    if "baseline_label" in df_samples.columns:
        by_y = {}
        for y, g in df_samples.groupby("baseline_label"):
            by_y[int(y)] = {
                "n_samples": int(len(g)),
                "mean_label_flip_rate": float(pd.to_numeric(g["metric_label_flip_rate"], errors="coerce").dropna().mean()),
                "mean_mean_abs_delta": float(pd.to_numeric(g["metric_mean_abs_delta"], errors="coerce").dropna().mean()),
                "mean_norm_sum_abs_delta": float(pd.to_numeric(g["metric_norm_sum_abs_delta"], errors="coerce").dropna().mean()),
            }
        summary["by_baseline_label"] = by_y

    return summary

def _write_html(out_html: str, df_samples: pd.DataFrame, summary: Dict[str, Any]) -> None:
    """
    Simple interactive HTML with DataTables and a JSON summary block.
    """
    import json as _json

    # Build table data/columns
    columns = list(df_samples.columns)
    data = df_samples.astype(object).where(pd.notnull(df_samples), None).values.tolist()

    html_str = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>ArgLLM Faithfulness (Per-sample)</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>

  <link rel="stylesheet" href="https://cdn.datatables.net/1.13.8/css/jquery.dataTables.min.css"/>
  <link rel="stylesheet" href="https://cdn.datatables.net/buttons/2.4.2/css/buttons.dataTables.min.css"/>
  <link rel="stylesheet" href="https://cdn.datatables.net/fixedheader/3.4.0/css/fixedHeader.dataTables.min.css"/>

  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 20px; }}
    pre {{ background: #f6f8fa; padding: 12px; border-radius: 8px; max-width: 100%; overflow-x: auto; }}
    .container {{ max-width: 98vw; overflow-x: auto; }}
    table.dataTable thead th {{ position: sticky; top: 0; background: #fff; }}
  </style>
</head>
<body>
  <h2>ArgLLM Faithfulness (Per-sample)</h2>
  <p>
    <b>Option 1</b>: <code>metric_label_flip_rate</code> (fraction of nodes whose removal flips the label).<br/>
    <b>Option 2</b>: <code>metric_norm_sum_abs_delta</code> (sum |Δ| normalized by N*max_possible_change); we also provide <code>metric_mean_abs_delta</code>.
  </p>

  <h3>Corpus Summary</h3>
  <pre>{_json.dumps(summary, indent=2)}</pre>

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
    const columns = {json.dumps(columns)}.map(c => ({{ title: c }}));
    const data = {json.dumps(data, ensure_ascii=False)};
    $(document).ready(function() {{
      $('#tbl').DataTable({{
        data: data,
        columns: columns,
        dom: 'Bfrtip',
        buttons: ['colvis', 'copyHtml5', 'csvHtml5'],
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
    ap.add_argument("--node_csv", required=True, help="Node-level LOO CSV produced by evaluate_argllm_node_faithfulness.py")
    ap.add_argument("--out_samples_csv", required=True, help="Per-sample metrics CSV")
    ap.add_argument("--out_samples_jsonl", default=None, help="Optional per-sample JSONL")
    ap.add_argument("--out_summary_json", default=None, help="Optional overall summary JSON")
    ap.add_argument("--out_html", default=None, help="Optional interactive HTML")
    args = ap.parse_args()

    # Load node CSV
    df_nodes = pd.read_csv(args.node_csv)
    missing = [c for c in EXPECTED_COLUMNS if c not in df_nodes.columns]
    if missing:
        print(f"[WARN] node_csv is missing expected columns: {missing}")

    # Compute per-sample
    df_samples = compute_per_sample_metrics(df_nodes)

    # Save per-sample CSV
    os.makedirs(os.path.dirname(args.out_samples_csv), exist_ok=True)
    df_samples.to_csv(args.out_samples_csv, index=False)
    print(f"[OK] Wrote per-sample metrics to: {args.out_samples_csv}")

    # Optional per-sample JSONL
    if args.out_samples_jsonl:
        os.makedirs(os.path.dirname(args.out_samples_jsonl), exist_ok=True)
        with open(args.out_samples_jsonl, "w", encoding="utf-8") as fout:
            for _, row in df_samples.iterrows():
                rec = {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
                fout.write(json.dumps(rec) + "\n")
        print(f"[OK] Wrote per-sample JSONL to: {args.out_samples_jsonl}")

    # Corpus summary
    summary = compute_corpus_summary(df_samples)

    if args.out_summary_json:
        os.makedirs(os.path.dirname(args.out_summary_json), exist_ok=True)
        with open(args.out_summary_json, "w", encoding="utf-8") as fsum:
            json.dump(summary, fsum, indent=2)
        print(f"[OK] Wrote summary JSON to: {args.out_summary_json}")

    # Optional HTML
    if args.out_html:
        _write_html(args.out_html, df_samples, summary)
        print(f"[OK] Wrote interactive HTML to: {args.out_html}")

    # Console summary
    print("\n=== Corpus Summary ===")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
