import pandas as pd
import numpy as np

CSV_PATH = "/vol/bitbucket/rc1124/MSc_Individual_Project/ExplainabilityInLLMs-MScThesis/evaluation/shake_test_results/rationale_token_perturbation/shake_test_results_strategyclaim_qwen.csv"

df = pd.read_csv(CSV_PATH)

df['label_orig'] = df['label_orig'].astype(str).str.upper()
df['label_pert'] = df['label_pert'].astype(str).str.upper()
if 'ground_truth' in df.columns:
    df['ground_truth'] = df['ground_truth'].astype(str).str.upper()

original_total = len(df)
df_filtered = df[(df['label_orig'] != 'UNKNOWN') & (df['label_pert'] != 'UNKNOWN')].copy()
removed_count = original_total - len(df_filtered)

flipped_mask_raw = (df_filtered['label_orig'] != df_filtered['label_pert'])
flipped = int(flipped_mask_raw.sum())
not_flipped = int((~flipped_mask_raw).sum())
total = int(len(df_filtered))

percentage_flipped = 100.0 * flipped / total if total > 0 else 0.0
percentage_not_flipped = 100.0 * not_flipped / total if total > 0 else 0.0

print("=== A) Flip summary (raw label change) ===")
print(f"Original total samples: {original_total}")
print(f"Samples removed (due to 'UNKNOWN'): {removed_count}")
print("---")
print(f"Total samples for calculation: {total}")
print(f"Labels Flipped (faithful): {flipped} ({percentage_flipped:.2f}%)")
print(f"Labels Not flipped (unfaithful): {not_flipped} ({percentage_not_flipped:.2f}%)")