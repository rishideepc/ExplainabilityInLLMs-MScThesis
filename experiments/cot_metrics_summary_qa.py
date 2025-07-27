# cot_metrics_summary.py

import os
import sys
import json
from collections import defaultdict
import pandas as pd
from IPython.display import display, HTML

project_root = os.path.abspath("..")
sys.path.append(project_root)

from evaluation.explanation_evaluation_calc_qa import evaluate_all_cot

datasets = ["truthfulqa", "strategyqa", "medqa"]
models = ["mistral", "llama", "qwen"]
metric_keys = ["redundancy", "weak_relevance", "strong_relevance"]

base_path = os.path.join(project_root, "results", "generation")
summary = defaultdict(dict)

for dataset in datasets:
    for model in models:
        subfolder = f"{dataset}_{model}"
        folder_path = os.path.join(base_path, subfolder)

        if not os.path.exists(folder_path):
            print(f"[SKIPPED] Missing folder: {folder_path}")
            continue

        jsonl_files = [f for f in os.listdir(folder_path) if f.endswith(".jsonl")]
        if not jsonl_files:
            print(f"[SKIPPED] No .jsonl file in {folder_path}")
            continue

        filepath = os.path.join(folder_path, jsonl_files[0])
        print(f"[✓] Evaluating: {filepath}")

        try:
            results = evaluate_all_cot(filepath)
            print(f"    → Processed {len(results)} explanations.")
        except Exception as e:
            print(f"[ERROR] Failed to evaluate: {filepath}")
            print(e)
            continue

        for metric in metric_keys:
            try:
                avg_score = sum(entry[metric] for entry in results) / len(results)
                summary[dataset][f"{model}_{metric}"] = round(avg_score, 4)
            except Exception as e:
                summary[dataset][f"{model}_{metric}"] = "N/A"

df = pd.DataFrame.from_dict(summary, orient="index").reset_index()
df.rename(columns={"index": "Dataset"}, inplace=True)

dataset_name_map = {
    "truthfulqa": "TruthfulQA",
    "strategyqa": "StrategyQA",
    "medqa": "MedQA",
    "commonsenseqa": "CommonSenseQA"
}
df["Dataset"] = df["Dataset"].map(dataset_name_map)

column_tuples = []
new_data = {}

for model in models:
    pretty_model = model.capitalize() if model != "llama" else "LLaMA"
    for metric in metric_keys:
        pretty_metric = metric.replace("_", " ").title()
        flat_col = f"{model}_{metric}"
        multi_col = (pretty_model, pretty_metric)
        column_tuples.append(multi_col)
        new_data[multi_col] = df[flat_col]

multi_df = pd.DataFrame(new_data)
multi_df.insert(0, ("", "Dataset"), df["Dataset"])
multi_df.columns = pd.MultiIndex.from_tuples(multi_df.columns)

df = multi_df

custom_table_html = f"""
<style>
th.col_heading.level0 {{
    text-align: center !important;
    padding: 10px 12px;
    background-color: #f9f9f9;
    font-weight: bold;
}}
th.col_heading.level1 {{
    text-align: center !important;
    padding: 8px 16px;
}}
td {{
    text-align: center;
    padding: 6px 10px;
}}
</style>
<div style="max-height: 500px; overflow: auto;">
{df.to_html(index=False, escape=False, border=0)}
</div>
"""
display(HTML(custom_table_html))

output_csv = os.path.join(project_root, "results", "cot_method", "cot_metrics_summary_qa.csv")
output_html = os.path.join(project_root, "results", "cot_method", "cot_metrics_summary_cv.html")

df.to_csv(output_csv, index=False)
print(f"[✓] CSV saved to: {output_csv}")

df.to_html(output_html, index=False)
print(f"[✓] HTML saved to: {output_html}")
