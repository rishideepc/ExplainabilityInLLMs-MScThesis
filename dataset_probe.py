"""
Script to probe datasets for evaluation, analysis, and reporting
"""

from datasets import load_dataset

# Load initial experiment questions - TruthfulQA
dataset_1 = load_dataset("truthful_qa", "generation")
questions_1 = dataset_1["validation"]["question"][:817]

# Load initial experiment questions - CommonsenseQA
dataset_2 = load_dataset("commonsense_qa")
questions_2 = [entry["question"] for entry in dataset_2["validation"]]
questions_2 = questions_2[:1221]

# Load initial experiment questions - StrategyQA
dataset_3 = load_dataset("wics/strategy-qa")
questions_3 = [entry["question"] for entry in dataset_3["test"]]
questions_3 = questions_3[:2290]

# # Load initial experiment questions - MedQA
dataset_4 = load_dataset("GBaker/MedQA-USMLE-4-options")
questions_4 = [entry["question"] for entry in dataset_4["test"]]
questions_4 = questions_4[:1273]



print(dataset_1, dataset_2, dataset_3, dataset_4)
print(questions_1[:3], questions_2[:3], questions_3[:3], questions_4[:3])