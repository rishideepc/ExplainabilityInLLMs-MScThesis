import pandas as pd

# Load the CSV file
df = pd.read_csv("/vol/bitbucket/rc1124/MSc_Individual_Project/ExplainabilityInLLMs-MScThesis/evaluation/shake_score_results_truthfulclaim.csv")  # Replace with your actual file path

# Ensure labels are in string or boolean format
df['label_orig'] = df['label_orig'].astype(str).str.upper()
df['label_pert'] = df['label_pert'].astype(str).str.upper()

# Calculate flips and non-flips
flipped = (df['label_orig'] != df['label_pert']).sum()
not_flipped = (df['label_orig'] == df['label_pert']).sum()
total = len(df)

# Calculate percentages
percentage_flipped = flipped / total * 100
percentage_not_flipped = not_flipped / total * 100

# Print the results
print(f"Total samples: {total}")
print(f"Labels Flipped: {flipped} ({percentage_flipped:.2f}%)")
print(f"Labels Not flipped: {not_flipped} ({percentage_not_flipped:.2f}%)")
