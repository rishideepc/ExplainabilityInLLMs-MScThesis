from datasets import load_dataset

# Load initial experiment questions - TruthfulQA
# dataset = load_dataset("truthful_qa", "generation")
# questions = dataset["validation"]["question"][:817]

# Load initial experiment questions - CommonsenseQA
dataset = load_dataset("commonsense_qa")
questions = [entry["question"] for entry in dataset["validation"]]
questions = questions[:1221]

# Load initial experiment questions - StrategyQA
# dataset = load_dataset("wics/strategy-qa")
# questions = [entry["question"] for entry in dataset["test"]]
# questions = questions[:2290]

# # Load initial experiment questions - MedQA
# dataset = load_dataset("GBaker/MedQA-USMLE-4-options")
# questions = [entry["question"] for entry in dataset["test"]]
# questions = questions[:1273]



print(dataset)