import sys
import os
project_root = os.path.abspath('...')
sys.path.append(project_root)


from evaluation.explanation_evaluation_calc import (
    load_jsonl, 
    evaluate_all_argllm,    
    evaluate_all_cot
)

file_path = os.path.join(project_root, "results", "generation", "truthfulqa_llama", "cot_outputs_ollama_meta_reasoning_conclusion_step_indices_fewshot.jsonl")
cot_evaluation_results = evaluate_all_cot(filepath=file_path)

index= 0
for result in cot_evaluation_results:
    print(index+1, ". ", result, "\n")
    index+=1