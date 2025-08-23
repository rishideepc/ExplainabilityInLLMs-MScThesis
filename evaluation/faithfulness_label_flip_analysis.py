import pandas as pd
import numpy as np

# === Config ===
CSV_PATH = "/vol/bitbucket/rc1124/MSc_Individual_Project/ExplainabilityInLLMs-MScThesis/evaluation/shake_score_results/shake_score_results_medclaim_qwen.csv"

# === Load ===
df = pd.read_csv(CSV_PATH)

# Normalize label casing
df['label_orig'] = df['label_orig'].astype(str).str.upper()
df['label_pert'] = df['label_pert'].astype(str).str.upper()
if 'ground_truth' in df.columns:
    df['ground_truth'] = df['ground_truth'].astype(str).str.upper()

# --- NEW: Filter out rows with "UNKNOWN" labels ---
original_total = len(df)
# Keep only the rows where NEITHER of the labels is "UNKNOWN"
df_filtered = df[(df['label_orig'] != 'UNKNOWN') & (df['label_pert'] != 'UNKNOWN')].copy()
removed_count = original_total - len(df_filtered)


# --- Flip / No-flip (Calculations now use the filtered DataFrame) ---
# Note: All calculations from here on use 'df_filtered'
flipped_mask_raw = (df_filtered['label_orig'] != df_filtered['label_pert'])
flipped = int(flipped_mask_raw.sum())
not_flipped = int((~flipped_mask_raw).sum())
total = int(len(df_filtered)) # The new total is based on the filtered data

percentage_flipped = 100.0 * flipped / total if total > 0 else 0.0
percentage_not_flipped = 100.0 * not_flipped / total if total > 0 else 0.0

print("=== A) Flip summary (raw label change) ===")
print(f"Original total samples: {original_total}")
print(f"Samples removed (due to 'UNKNOWN'): {removed_count}")
print("---")
print(f"Total samples for calculation: {total}")
print(f"Labels Flipped (faithful): {flipped} ({percentage_flipped:.2f}%)")
print(f"Labels Not flipped (unfaithful): {not_flipped} ({percentage_not_flipped:.2f}%)")



# # --- MBF (normalized) summaries ---
# has_mbf_cols = all(c in df.columns for c in ["mbf_norm", "mbf_k_star", "mbf_K"])
# if not has_mbf_cols:
#     print("\n=== C) MBF (normalized) ===")
#     print("No MBF columns found (mbf_norm, mbf_k_star, mbf_K). Re-run the SHAKE test with the updated runner.")
# else:
#     # Valid MBF rows are those with K > 0
#     df['mbf_K'] = pd.to_numeric(df['mbf_K'], errors='coerce').fillna(0).astype(int)
#     df['mbf_k_star'] = pd.to_numeric(df['mbf_k_star'], errors='coerce').fillna(0).astype(int)
#     df['mbf_norm'] = pd.to_numeric(df['mbf_norm'], errors='coerce').fillna(0.0)

#     valid_mask = df['mbf_K'] > 0
#     mbf_valid = df[valid_mask].copy()

#     n_valid = int(len(mbf_valid))
#     if n_valid == 0:
#         print("\n=== C) MBF (normalized) ===")
#         print("No valid MBF rows (mbf_K <= 0 for all).")
#     else:
#         # Flip coverage under MBF (exists k* s.t. flip)
#         flip_mask_mbf = mbf_valid['mbf_k_star'] > 0
#         n_mbf_flip = int(flip_mask_mbf.sum())
#         mbf_flip_rate = 100.0 * n_mbf_flip / n_valid

#         # Means & percentiles
#         mean_mbf_all = float(mbf_valid['mbf_norm'].mean())
#         mean_mbf_flipped = float(mbf_valid.loc[flip_mask_mbf, 'mbf_norm'].mean()) if n_mbf_flip else 0.0

#         pcts = {}
#         for p in (0, 10, 25, 50, 75, 90, 100):
#             try:
#                 pcts[p] = float(np.percentile(mbf_valid['mbf_norm'].values, p))
#             except Exception:
#                 pcts[p] = float('nan')

#         # Distribution of k* (only among flipped)
#         kstar_counts = (
#             mbf_valid.loc[flip_mask_mbf, 'mbf_k_star']
#             .value_counts()
#             .sort_index()
#         )

#         # Optional: average relative budget k*/K among flipped
#         if n_mbf_flip:
#             rel_budget_mean = float((mbf_valid.loc[flip_mask_mbf, 'mbf_k_star'] / mbf_valid.loc[flip_mask_mbf, 'mbf_K']).mean())
#         else:
#             rel_budget_mean = 0.0

#         print("\n=== C) MBF (normalized) ===")
#         print(f"Valid MBF rows (K>0): {n_valid}")
#         print(f"MBF flip coverage: {n_mbf_flip} / {n_valid} ({mbf_flip_rate:.2f}%)")
#         print(f"MBF_norm mean (all valid): {mean_mbf_all:.4f}")
#         print(f"MBF_norm mean (flipped only): {mean_mbf_flipped:.4f}")
#         print("MBF_norm percentiles (all valid):")
#         for p in (0, 10, 25, 50, 75, 90, 100):
#             print(f"  p{p:>3}: {pcts[p]:.4f}")

#         print("\nDistribution of minimal k* causing flip (flipped only):")
#         if len(kstar_counts):
#             for k, c in kstar_counts.items():
#                 print(f"  k* = {int(k)} : {int(c)} samples ({100.0 * c / n_valid:.2f}% of valid)")
#         else:
#             print("  (no flips under MBF)")

#         print(f"\nAverage relative budget among flipped (k*/K): {rel_budget_mean:.4f}")

