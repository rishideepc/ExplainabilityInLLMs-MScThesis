import pandas as pd
import os

def analyze_shake_test(file_path):
    """
    Computes label-flip statistics from a SHAKE test result CSV 
    
    @params: CSV path (string) 
    
    @returns: dictionary summary
    """
    try:
        df = pd.read_csv(file_path)

        df['label_orig'] = df['label_orig'].astype(str).str.upper()
        df['label_pert'] = df['label_pert'].astype(str).str.upper()
        if 'ground_truth' in df.columns:
            df['ground_truth'] = df['ground_truth'].astype(str).str.upper()

        original_total = len(df)
        df_filtered = df[(df['label_orig'] != 'UNKNOWN') & (df['label_pert'] != 'UNKNOWN')].copy()
        removed_count = original_total - len(df_filtered)

        total_filtered = len(df_filtered)
        if total_filtered > 0:
            flipped = (df_filtered['label_orig'] != df_filtered['label_pert']).sum()
            not_flipped = total_filtered - flipped
            percentage_flipped = 100.0 * flipped / total_filtered
        else:
            flipped, not_flipped, percentage_flipped = 0, 0, 0.0

        filename = os.path.basename(file_path)
        model_name = filename.replace('shake_test_results_', '').replace('.csv', '')

        return {
            'Model': model_name,
            'Original Samples': original_total,
            'Removed (Unknown)': removed_count,
            'Final Samples': total_filtered,
            'Flipped': flipped,
            'Flip %': percentage_flipped,
        }
    except FileNotFoundError:
        print(f"Error: The file was not found at {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")
        return None

file_paths = [
    "/vol/bitbucket/rc1124/MSc_Individual_Project/ExplainabilityInLLMs-MScThesis/evaluation/shake_test_results/rationale_token_perturbation/shake_test_results_truthfulclaim_llama.csv",
    "/vol/bitbucket/rc1124/MSc_Individual_Project/ExplainabilityInLLMs-MScThesis/evaluation/shake_test_results/rationale_token_perturbation/shake_test_results_truthfulclaim_mistral.csv",
    "/vol/bitbucket/rc1124/MSc_Individual_Project/ExplainabilityInLLMs-MScThesis/evaluation/shake_test_results/rationale_token_perturbation/shake_test_results_truthfulclaim_qwen.csv",
    "/vol/bitbucket/rc1124/MSc_Individual_Project/ExplainabilityInLLMs-MScThesis/evaluation/shake_test_results/rationale_token_perturbation/shake_test_results_strategyclaim_llama.csv",
    "/vol/bitbucket/rc1124/MSc_Individual_Project/ExplainabilityInLLMs-MScThesis/evaluation/shake_test_results/rationale_token_perturbation/shake_test_results_strategyclaim_mistral.csv",
    "/vol/bitbucket/rc1124/MSc_Individual_Project/ExplainabilityInLLMs-MScThesis/evaluation/shake_test_results/rationale_token_perturbation/shake_test_results_strategyclaim_qwen.csv",
    "/vol/bitbucket/rc1124/MSc_Individual_Project/ExplainabilityInLLMs-MScThesis/evaluation/shake_test_results/rationale_token_perturbation/shake_test_results_medclaim_llama.csv",
    "/vol/bitbucket/rc1124/MSc_Individual_Project/ExplainabilityInLLMs-MScThesis/evaluation/shake_test_results/rationale_token_perturbation/shake_test_results_medclaim_mistral.csv",
    "/vol/bitbucket/rc1124/MSc_Individual_Project/ExplainabilityInLLMs-MScThesis/evaluation/shake_test_results/rationale_token_perturbation/shake_test_results_medclaim_qwen.csv",
    "/vol/bitbucket/rc1124/MSc_Individual_Project/ExplainabilityInLLMs-MScThesis/evaluation/shake_test_results/rationale_token_perturbation/shake_test_results_commonsenseclaim_llama.csv",
    "/vol/bitbucket/rc1124/MSc_Individual_Project/ExplainabilityInLLMs-MScThesis/evaluation/shake_test_results/rationale_token_perturbation/shake_test_results_commonsenseclaim_mistral.csv",
    "/vol/bitbucket/rc1124/MSc_Individual_Project/ExplainabilityInLLMs-MScThesis/evaluation/shake_test_results/rationale_token_perturbation/shake_test_results_commonsenseclaim_qwen.csv",
]

results_list = []

for path in file_paths:
    analysis_result = analyze_shake_test(path)
    if analysis_result:
        results_list.append(analysis_result)

if results_list:
    summary_df = pd.DataFrame(results_list)
    summary_df.set_index('Model', inplace=True)

    print("=== Shake Test Flip Summary ===")
    formatted_df = summary_df.copy()
    formatted_df['Flip %'] = formatted_df['Flip %'].map('{:.2f}%'.format)
    print(formatted_df.to_string())
else:
    print("\nNo data was processed. Please check the file paths and file contents.")
