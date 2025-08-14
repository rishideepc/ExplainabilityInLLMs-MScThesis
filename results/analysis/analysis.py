import pandas as pd

# Load CSV
df = pd.read_csv("node_faithfulness_loo.csv")

# --- 1. Label flip percentage ---
# Ensure both columns are strings for comparison
baseline_labels = df['baseline_label'].astype(str)
loo_labels = df['loo_label_after_ablation'].astype(str)

label_flips = (baseline_labels != loo_labels).sum()
total_rows = len(df)
flip_percentage = (label_flips / total_rows) * 100
no_flip_percentage = 100 - flip_percentage

print("=== Label Flip Analysis ===")
print(f"Total rows: {total_rows}")
print(f"Label flips: {label_flips} ({flip_percentage:.2f}%)")
print(f"No label flips: {total_rows - label_flips} ({no_flip_percentage:.2f}%)\n")

# --- 2. Claim strength analysis ---
baseline_mean = df['baseline_claim_strength'].mean()
baseline_std = df['baseline_claim_strength'].std()
loo_mean = df['loo_claim_strength'].mean()
loo_std = df['loo_claim_strength'].std()
delta_mean = df['claim_strength_delta'].mean()
delta_std = df['claim_strength_delta'].std()

print("=== Claim Strength Analysis ===")
print(f"Baseline claim strength: mean={baseline_mean:.4f}, std={baseline_std:.4f}")
print(f"LOO claim strength:      mean={loo_mean:.4f}, std={loo_std:.4f}")
print(f"Claim strength delta:    mean={delta_mean:.4f}, std={delta_std:.4f}")

# Check correlation between baseline and LOO claim strengths
correlation = df['baseline_claim_strength'].corr(df['loo_claim_strength'])
print(f"Correlation between baseline and LOO claim strengths: {correlation:.4f}")

# Optional: min/max for additional insight
print("\nMin/Max values:")
print(f"Baseline claim strength: min={df['baseline_claim_strength'].min():.4f}, max={df['baseline_claim_strength'].max():.4f}")
print(f"LOO claim strength:      min={df['loo_claim_strength'].min():.4f}, max={df['loo_claim_strength'].max():.4f}")
print(f"Claim strength delta:    min={df['claim_strength_delta'].min():.4f}, max={df['claim_strength_delta'].max():.4f}")
