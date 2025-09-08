"""
CoT metrics aggregation script for all datasets and models; 
evaluates each JSONL file to compile the average metrics into a summary table;
saves results to CSV and interactive HTML files.
"""

import os
import sys
import json
from collections import defaultdict
import pandas as pd

project_root = os.path.abspath("...")
sys.path.append(project_root)

from evaluation.explanation_evaluation_calc import evaluate_all_cot

datasets = ["truthfulqa", "strategyqa", "medqa", "commonsenseqa"]
models = ["mistral", "llama", "qwen"]
metric_keys = ["redundancy", "weak_relevance", "strong_relevance"]

base_path = os.path.join(project_root, "results", "generation")
output_csv = os.path.join(project_root, "results", "cot_metrics_summary.csv")
output_html = os.path.join(project_root, "results", "cot_metrics_summary.html")

summary = defaultdict(dict)

for dataset in datasets:
    for model in models:
        subfolder = f"{dataset}_{model}"
        folder_path = os.path.join(base_path, subfolder)
        
        if not os.path.exists(folder_path):
            print(f"[WARNING] Folder not found: {folder_path}")
            continue

        jsonl_files = [f for f in os.listdir(folder_path) if f.endswith(".jsonl")]
        if not jsonl_files:
            print(f"[WARNING] No .jsonl files in {folder_path}")
            continue

        filepath = os.path.join(folder_path, jsonl_files[0])
        print(f"Processing: {filepath}")

        try:
            results = evaluate_all_cot(filepath)
        except Exception as e:
            print(f"[ERROR] Failed to evaluate {filepath}: {e}")
            continue

        if not results:
            print(f"[WARNING] No results in file: {filepath}")
            continue

        for metric in metric_keys:
            try:
                avg = sum(entry[metric] for entry in results) / len(results)
                summary[dataset][f"{model}_{metric}"] = round(avg, 4)
            except Exception as e:
                print(f"[ERROR] {metric} failed for {dataset}_{model}: {e}")
                summary[dataset][f"{model}_{metric}"] = "N/A"

df = pd.DataFrame.from_dict(summary, orient="index")
df.index.name = "Dataset"
df = df.reset_index()

ordered_cols = ["Dataset"]
for model in models:
    for metric in metric_keys:
        ordered_cols.append(f"{model}_{metric}")
df = df[ordered_cols]

print("\n=== CoT Explanation Evaluation Summary ===\n")
print(df.to_string(index=False))

df.to_csv(output_csv, index=False)
print(f"\n[✓] Saved summary CSV to: {output_csv}")

with open(output_html, "w") as f:
    f.write("""
<html>
<head>
<style>
table {
  border-collapse: collapse;
  width: 100%;
  font-family: Arial, sans-serif;
}
th, td {
  text-align: left;
  padding: 8px;
  border-bottom: 1px solid #ddd;
}
th {
  background-color: #f2f2f2;
  cursor: pointer;
}
tr:hover {
  background-color: #f5f5f5;
}
</style>
</head>
<body>
<h2>Chain-of-Thought Explanation Evaluation Summary</h2>
""")
    f.write(df.to_html(index=False, escape=False))
    f.write("""
<script>
document.querySelectorAll("th").forEach((th, index) => {
  th.addEventListener("click", () => {
    const table = th.closest("table");
    const tbody = table.querySelector("tbody");
    Array.from(tbody.querySelectorAll("tr"))
      .sort((a, b) => {
        let valA = a.children[index].innerText.trim();
        let valB = b.children[index].innerText.trim();
        let numA = parseFloat(valA);
        let numB = parseFloat(valB);
        if (!isNaN(numA) && !isNaN(numB)) return numA - numB;
        return valA.localeCompare(valB);
      })
      .forEach(tr => tbody.appendChild(tr));
  });
});
</script>
</body>
</html>
""")

print(f"[✓] Saved interactive HTML table to: {output_html}")
