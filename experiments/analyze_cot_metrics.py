import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

project_root = os.path.abspath("...")
sys.path.append(project_root)

# === Load Data ===
summary_path = os.path.join(project_root, "results", "cot_method", "cot_metrics_summary.csv")

# Load with multi-level column header
df = pd.read_csv(summary_path, header=[0, 1])

# Flatten MultiIndex columns for analysis
df.columns = [' '.join(col).strip() if col[0] else col[1] for col in df.columns]

# Rename first column explicitly if needed
df = df.rename(columns={df.columns[0]: "Dataset"})

# Sanity check for required columns
models = ["Mistral", "LLaMA", "Qwen"]
metrics = ["Redundancy", "Weak Relevance", "Strong Relevance"]

required_columns = ["Dataset"] + [f"{model} {metric}" for model in models for metric in metrics]
missing = [col for col in required_columns if col not in df.columns]
if missing:
    raise ValueError(f"Missing expected columns: {missing}")

# === Clean column names (flatten if needed) ===
models = ["Mistral", "LLaMA", "Qwen"]
metrics = ["Redundancy", "Weak Relevance", "Strong Relevance"]

# === Textual Analysis ===
print("\n=== A. TEXTUAL ANALYSIS ===")

# 1. Ranking per dataset and metric
print("\n[1] Per-Dataset Rankings (Higher is better):")
for metric in metrics:
    print(f"\n--- {metric} ---")
    for i, row in df.iterrows():
        dataset = row["Dataset"]
        scores = {model: row[f"{model} {metric}"] for model in models}
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        print(f"{dataset}: {', '.join([f'{model} ({score})' for model, score in ranked])}")

# 2. Standard deviation of each model across datasets
print("\n[2] Standard Deviation per Model-Metric:")
for model in models:
    for metric in metrics:
        col = f"{model} {metric}"
        std = df[col].std()
        print(f"{col}: {std:.4f}")

# 3. Correlation between metrics (using all values)
print("\n[3] Correlation Between Metrics (across all values):")

# Flatten the data for all metric values across all models
metric_values = {metric: [] for metric in metrics}

for metric in metrics:
    for model in models:
        col = f"{model} {metric}"
        metric_values[metric].extend(df[col].tolist())

# Convert to DataFrame: each column is one metric
metric_df = pd.DataFrame(metric_values)

# Print correlation matrix
correlation_matrix = metric_df.corr(method="pearson").round(3)
print(correlation_matrix)

# 4. Coefficient of Variation per model
print("\n[4] Coefficient of Variation (CV) per Model-Metric:")
for model in models:
    for metric in metrics:
        col = f"{model} {metric}"
        mu = df[col].mean()
        sigma = df[col].std()
        cv = sigma / mu if mu != 0 else np.nan
        print(f"{col}: {cv:.4f}")

# 5. Composite Score (equal weights)
print("\n[5] Composite Scores per Model (avg of all metrics):")
composite_scores = {}
for model in models:
    cols = [f"{model} {metric}" for metric in metrics]
    composite = df[cols].mean().mean()
    composite_scores[model] = composite
for model, score in sorted(composite_scores.items(), key=lambda x: -x[1]):
    print(f"{model}: {score:.4f}")

# 6. Pareto Frontier (textual form)
print("\n[6] Pareto Frontier Check (Model dominance by metric):")
for metric in metrics:
    best_model = df[[f"{model} {metric}" for model in models]].mean().idxmax()
    best_model_clean = best_model.split(" ")[0]
    print(f"Metric: {metric} → Best on average: {best_model_clean}")

# === Visualizations ===
print("\n=== B. VISUAL ANALYSIS ===")

viz_dir = os.path.join(project_root, "results", "cot_method")
os.makedirs(viz_dir, exist_ok=True)

# 1. Grouped Bar Chart per Metric
for metric in metrics:
    plt.figure(figsize=(8, 5))
    for model in models:
        plt.bar(df["Dataset"] + f" ({model})", df[f"{model} {metric}"], label=model)
    plt.xticks(rotation=45, ha='right')
    plt.title(f"{metric} Scores Across Datasets")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f"bar_{metric.lower().replace(' ', '_')}.png"))
    plt.close()

# 2. Radar Plot per Model
def radar_plot(data, labels, title, filename):
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    data += data[:1]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, data, linewidth=2)
    ax.fill(angles, data, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels([])
    ax.set_title(title, y=1.1)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

for model in models:
    values = [df[f"{model} {metric}"].mean() for metric in metrics]
    radar_plot(values, metrics, f"{model} Explanation Profile", os.path.join(viz_dir, f"radar_{model.lower()}.png"))

# 3. Heatmap
heatmap_data = pd.DataFrame({
    f"{model} {metric}": df[f"{model} {metric}"]
    for model in models for metric in metrics
})
plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data.T, cmap="YlGnBu", annot=True, fmt=".2f", cbar=True)
plt.title("Explanation Quality Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(viz_dir, "heatmap_all_metrics.png"))
plt.close()

# 4. Ranking Line Plot
ranking_df = df.copy()
for metric in metrics:
    for model in models:
        col = f"{model} {metric}"
        ranking_df[f"{model} {metric}_rank"] = ranking_df[col].rank(ascending=False)
plt.figure(figsize=(10, 6))
for model in models:
    scores = []
    for metric in metrics:
        avg_rank = ranking_df[f"{model} {metric}_rank"].mean()
        scores.append(avg_rank)
    plt.plot(metrics, scores, label=model, marker="o")
plt.title("Average Model Rankings Per Metric")
plt.ylabel("Average Rank (lower is better)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(viz_dir, "lineplot_model_ranks.png"))
plt.close()

print(f"[✓] Visuals saved in: {viz_dir}")
