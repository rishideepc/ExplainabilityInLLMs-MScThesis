#!/usr/bin/env python3
# viz_cot_results.py
# Visualize CoT evaluation results for Claim Verification & Question Answering
# Usage:
#   python viz_cot_results.py --out ./cot_viz --no-show
#   python viz_cot_results.py  (defaults to ./cot_viz and shows nothing)
#
# Requires: pandas, numpy, matplotlib

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from textwrap import wrap

# -------------------------
# Embedded data (from your message)
# -------------------------
CLAIM_ROWS = [
    ["Truthfulclaim", 0.9391, 0.2680, 0.2325, 0.9951, 0.1755, 0.1533, 0.9978, 0.1734, 0.1384],
    ["Strategyclaim", 0.9478, 0.2582, 0.2360, 0.9906, 0.1581, 0.1462, 0.9994, 0.1364, 0.1154],
    ["Medclaim",      0.9814, 0.2282, 0.2073, 0.9966, 0.1465, 0.1399, 1.0000, 0.1297, 0.1115],
]

QA_ROWS = [
    ["TruthfulQA", 0.9189, 0.3481, 0.2942, 0.9792, 0.2178, 0.1994, 1.0000, 0.1490, 0.1198],
    ["StrategyQA", 0.9234, 0.3049, 0.2770, 0.9706, 0.2044, 0.1928, 1.0000, 0.1287, 0.1092],
    ["MedQA",      0.9878, 0.2528, 0.2243, 0.9884, 0.1706, 0.1582, 0.9993, 0.1380, 0.1183],
]

COLS = [
    "Dataset",
    "Mistral_Redundancy","Mistral_WeakRel","Mistral_StrongRel",
    "Llama_Redundancy","Llama_WeakRel","Llama_StrongRel",
    "Qwen_Redundancy","Qwen_WeakRel","Qwen_StrongRel",
]

SUITES = ["Claim Verification", "Question Answering"]
MODELS = ["Mistral", "Llama", "Qwen"]
METRICS = ["Redundancy", "WeakRel", "StrongRel"]  # use "WeakRel"/"StrongRel" not "Weak Relevance" for brevity


# -------------------------
# Data preparation
# -------------------------
def build_frames():
    df_claim = pd.DataFrame(CLAIM_ROWS, columns=COLS)
    df_qa = pd.DataFrame(QA_ROWS, columns=COLS)
    return df_claim, df_qa

def to_tidy(df: pd.DataFrame, suite_label: str) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        dset = r["Dataset"]
        for c in df.columns:
            if c == "Dataset":
                continue
            model, metric = c.split("_")
            rows.append({
                "Suite": suite_label,
                "Dataset": dset,
                "Model": model,
                "Metric": metric,
                "Value": float(r[c]),
            })
    return pd.DataFrame(rows)


# -------------------------
# Plotting helpers (matplotlib; one figure per chart)
# -------------------------
def savefig(fig, out_dir: Path, name: str, images: list):
    path = out_dir / name
    fig.savefig(path, bbox_inches="tight", dpi=180)
    plt.close(fig)
    images.append(path)
    return path

def grouped_bar_by_dataset(tidy_df, metric, suite_label, out_dir, images):
    sub = tidy_df[(tidy_df["Suite"] == suite_label) & (tidy_df["Metric"] == metric)]
    datasets = list(sub["Dataset"].unique())
    x = np.arange(len(datasets))
    width = 0.25

    fig = plt.figure(figsize=(10, 5.5))
    ax = fig.add_subplot(111)
    for i, m in enumerate(MODELS):
        vals = [sub[(sub["Dataset"] == d) & (sub["Model"] == m)]["Value"].values[0] for d in datasets]
        ax.bar(x + (i - 1) * width, vals, width, label=m)
        for xi, v in zip(x + (i - 1) * width, vals):
            ax.text(xi, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_title(f"{suite_label} — {metric} by Dataset and Model")
    ax.set_ylabel(metric)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fname = f"{suite_label.replace(' ','_').lower()}_{metric.lower()}_grouped_by_dataset.png"
    return savefig(fig, out_dir, fname, images)

def heatmap_dataset_vs_model(tidy_df, metric, suite_label, out_dir, images):
    sub = tidy_df[(tidy_df["Suite"] == suite_label) & (tidy_df["Metric"] == metric)]
    pivot = sub.pivot(index="Dataset", columns="Model", values="Value").reindex(columns=MODELS)
    fig = plt.figure(figsize=(6.6, 4.6))
    ax = fig.add_subplot(111)
    im = ax.imshow(pivot.values, aspect="auto")
    ax.set_title(f"{suite_label} — {metric} (Heatmap)")
    ax.set_xticks(np.arange(pivot.shape[1])); ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(pivot.shape[0])); ax.set_yticklabels(pivot.index)

    # value labels
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.3f}", ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax, shrink=0.85, label=metric)
    fname = f"{suite_label.replace(' ','_').lower()}_{metric.lower()}_heatmap.png"
    return savefig(fig, out_dir, fname, images)

def line_per_model_across_datasets(tidy_df, metric, suite_label, out_dir, images):
    sub = tidy_df[(tidy_df["Suite"] == suite_label) & (tidy_df["Metric"] == metric)]
    datasets = list(sub["Dataset"].unique())

    fig = plt.figure(figsize=(10, 5.5))
    ax = fig.add_subplot(111)
    for m in MODELS:
        vals = [sub[(sub["Dataset"] == d) & (sub["Model"] == m)]["Value"].values[0] for d in datasets]
        ax.plot(datasets, vals, marker="o", label=m)
        for xd, v in zip(datasets, vals):
            ax.text(xd, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_title(f"{suite_label} — {metric} across Datasets (per Model)")
    ax.set_ylabel(metric)
    ax.set_xlabel("Dataset")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fname = f"{suite_label.replace(' ','_').lower()}_{metric.lower()}_lines_per_model.png"
    return savefig(fig, out_dir, fname, images)

def radar_per_model(suite_df, model, out_dir, images):
    # Average across datasets for each metric, per model
    metrics = METRICS
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    vals = [suite_df[(suite_df["Model"] == model) & (suite_df["Metric"] == m)]["Value"].mean()
            for m in metrics]
    vals += vals[:1]

    fig = plt.figure(figsize=(6.2, 6.2))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, vals, marker="o")
    ax.fill(angles, vals, alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(["\n".join(wrap(lbl, 8)) for lbl in metrics])
    ax.set_title(f"Average across datasets — {model}")
    ax.set_rlabel_position(0)
    ax.grid(True, linestyle="--", alpha=0.4)

    fname = f"radar_{model.lower()}.png"
    return savefig(fig, out_dir, fname, images)

def radar_set_for_suite(tidy_df, suite_label, out_dir, images):
    sub = tidy_df[tidy_df["Suite"] == suite_label]
    for m in MODELS:
        radar_per_model(sub, m, out_dir, images)


# -------------------------
# Tables & summaries
# -------------------------
def best_model_table(tidy_df):
    def best_row(g):
        idx = g["Value"].idxmax()
        return g.loc[idx, ["Suite","Dataset","Metric","Model","Value"]]
    return tidy_df.groupby(["Suite","Dataset","Metric"]).apply(best_row).reset_index(drop=True)


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Visualize CoT evaluation results.")
    parser.add_argument("--out", type=str, default="./cot_viz", help="Output directory")
    parser.add_argument("--show", dest="show", action="store_true", help="Show figures interactively")
    parser.add_argument("--no-show", dest="show", action="store_false", help="Do not show figures")
    parser.set_defaults(show=False)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build data
    df_claim, df_qa = build_frames()
    tidy_claim = to_tidy(df_claim, "Claim Verification")
    tidy_qa = to_tidy(df_qa, "Question Answering")
    tidy_all = pd.concat([tidy_claim, tidy_qa], ignore_index=True)

    # Save CSVs
    tidy_all.to_csv(out_dir / "tidy_all.csv", index=False)
    pd.DataFrame(CLAIM_ROWS, columns=COLS).to_csv(out_dir / "claim_raw.csv", index=False)
    pd.DataFrame(QA_ROWS, columns=COLS).to_csv(out_dir / "qa_raw.csv", index=False)

    # Best models CSV
    best = best_model_table(tidy_all)
    best.to_csv(out_dir / "best_models.csv", index=False)

    # Wide pivots for each suite (optional scan)
    wide_claim = tidy_claim.pivot_table(index="Dataset", columns=["Model","Metric"], values="Value")
    wide_qa = tidy_qa.pivot_table(index="Dataset", columns=["Model","Metric"], values="Value")
    wide_claim.to_csv(out_dir / "wide_claim.csv")
    wide_qa.to_csv(out_dir / "wide_qa.csv")

    # Figures
    images = []
    for suite in ["Claim Verification", "Question Answering"]:
        for metric in METRICS:
            grouped_bar_by_dataset(tidy_all, metric, suite, out_dir, images)
            heatmap_dataset_vs_model(tidy_all, metric, suite, out_dir, images)
            line_per_model_across_datasets(tidy_all, metric, suite, out_dir, images)
        radar_set_for_suite(tidy_all, suite, out_dir, images)

    # Combined PDF
    pdf_path = out_dir / "cot_visualizations_all.pdf"
    with PdfPages(pdf_path) as pdf:
        for p in images:
            fig = plt.figure(figsize=(11, 7.5))
            ax = fig.add_subplot(111)
            ax.axis("off")
            im = plt.imread(p)
            ax.imshow(im)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"[OK] Saved {len(images)} images to: {out_dir}")
    print(f"[OK] Combined PDF: {pdf_path}")
    print(f"[OK] CSVs: tidy_all.csv, best_models.csv, wide_claim.csv, wide_qa.csv")

    if args.show:
        # If you want to see one example interactively:
        plt.figure()
        plt.title("Preview only — open the saved PNG/PDFs for the full set")
        plt.text(0.1, 0.5, "Figures saved to disk.\nUse --show to preview custom plots.", fontsize=12)
        plt.axis("off")
        plt.show()

if __name__ == "__main__":
    main()
