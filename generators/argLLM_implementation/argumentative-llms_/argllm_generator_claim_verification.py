import os
import sys
import json
import re
import torch
from datasets import load_dataset
import time

start_time= time.time()

# Set project root and save path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.append(PROJECT_ROOT)
SAVE_PATH = os.path.join(PROJECT_ROOT, "results", "generation", "argLLM_generation", "strategyclaim_qwen", "argllm_outputs_ollama.jsonl")
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

from argument_miner import ArgumentMiner
from uncertainty_estimator import UncertaintyEstimator
from llm_managers import HuggingFaceLlmManager
import Uncertainpy.src.uncertainpy.gradual as grad
from prompt import ArgumentMiningPrompts, UncertaintyEvaluatorPrompts

# === LLM Config ===
# MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
MODEL_NAME = "Qwen/Qwen3-8B"
QUANTIZATION = "none"  # use "4bit" or "8bit" for low VRAM GPUs
USE_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if USE_CUDA else "cpu"
print(f"Device selected: {DEVICE}")

# === Dataset ===
# Load initial experiment questions - TruthfulQA
# print("Loading dataset...")
# dataset = load_dataset("truthful_qa", "generation")
# questions = dataset["validation"]["question"][:817]

# Load initial experiment questions - StrategyQA
# print("Loading dataset...")
# dataset = load_dataset("wics/strategy-qa")
# questions = [entry["question"] for entry in dataset["test"]]
# questions = questions[:2290]

# Load initial experiment questions - MedQA
# print("Loading dataset...")
# dataset = load_dataset("GBaker/MedQA-USMLE-4-options")
# questions = [entry["question"] for entry in dataset["test"]]
# questions = questions[:1273]

print("Loading claim verification dataset...")
import json

with open(os.path.join(PROJECT_ROOT, "generators", "strategy_claims_dataset.json"), "r") as f:
    dataset = json.load(f)
dataset = dataset[:2290]  # or whatever size you want


# === HuggingFace Pipeline ===
print("Initializing LLM...")
llm_manager = HuggingFaceLlmManager(
    model_name=MODEL_NAME,
    quantization=QUANTIZATION,
    cache_dir=os.path.join(PROJECT_ROOT, "cache"),
    input_device=DEVICE,
)

print("✅ LLM device:", llm_manager.input_device)

# === GPU Model Device Check ===
try:
    model_device = next(llm_manager.model.parameters()).device
    print(f"Model parameters are on device: {model_device}")
except AttributeError:
    print("[WARN] Could not access llm_manager.model to check device.")
except Exception as e:
    print(f"[ERROR] Unexpected error while checking model device: {e}")

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")

# === Certainty Formatter Patch ===
def robust_certainty_formatter(certainty):
    try:
        match = re.search(r"([01](\.\d+)?)", certainty)
        if match:
            return float(match.group(1))
        else:
            print("[WARN] No valid certainty float found. Using default 0.5")
            return 0.5
    except Exception as e:
        print(f"[ERROR] Certainty score parsing failed: {e}")
        return 0.5

# Monkey-patch prompt with safe formatter
original_prompt_fn = UncertaintyEvaluatorPrompts.chatgpt
def patched_chatgpt(statement, verbal=False, **_):
    prompt, constraints, _ = original_prompt_fn(statement, verbal=verbal)
    return prompt, constraints, robust_certainty_formatter
UncertaintyEvaluatorPrompts.chatgpt = patched_chatgpt

# === Argument Miner + Uncertainty Estimator ===
generation_args = {
    "temperature": 0.7,
    "max_new_tokens": 128,
    "top_p": 0.95,
}

print("Instantiating generators...")
am = ArgumentMiner(
    llm_manager=llm_manager,
    generate_prompt=ArgumentMiningPrompts.chatgpt,
    depth=1,
    breadth=1,
    generation_args=generation_args
)

ue = UncertaintyEstimator(
    llm_manager=llm_manager,
    generate_prompt=UncertaintyEvaluatorPrompts.chatgpt,
    verbal=False,
    generation_args=generation_args
)

# === Semantics: Aggregation + Influence Functions ===
agg_f = grad.semantics.modular.ProductAggregation()
inf_f = grad.semantics.modular.LinearInfluence(conservativeness=1)

# === Main Generation Loop ===
results = []
for idx, entry in enumerate(dataset, 1):
    try:
        # print(f"\n[{idx}/{len(questions)}] Processing: {question}")
        # t_base, t_estimated = am.generate_arguments(question, ue)

        claim = entry["claim"]
        print(f"\n[{idx}/{len(dataset)}] Processing claim: {claim}")
        t_base, t_estimated = am.generate_arguments(claim, ue)

        print("📈 Computing strength propagation...")
        grad.algorithms.computeStrengthValues(t_base, agg_f, inf_f)
        grad.algorithms.computeStrengthValues(t_estimated, agg_f, inf_f)

        results.append({
            "id": entry["qid"],
            "question": entry["question"],
            # "option": entry["option"],
            "claim": entry["claim"],
            "label": entry["label/answer"],
            "base": {
                "bag": t_base.to_dict(),
                "prediction": t_base.arguments["db0"].strength,
            },
            "estimated": {
                "bag": t_estimated.to_dict(),
                "prediction": t_estimated.arguments["db0"].strength,
            },
        })

    except Exception as e:
        print(f"❌ ERROR while processing: {question}\n{e}")

# === Save Outputs ===
print("Saving results...")
with open(SAVE_PATH, "w") as f:
    for item in results:
        json.dump(item, f)
        f.write("\n")
print(f"Saved {len(results)} entries to {SAVE_PATH}")

elapsed_time = time.time() - start_time
elapsed_hours, rem = divmod(elapsed_time, 3600)
elapsed_minutes, elapsed_seconds = divmod(rem, 60)
print(f"⏱️ Script completed in {int(elapsed_hours):02d}:{int(elapsed_minutes):02d}:{int(elapsed_seconds):02d}")

# import torch
# print("Torch version:", torch.__version__)
# print("CUDA available:", torch.cuda.is_available())

# if torch.cuda.is_available():
#     device = torch.cuda.current_device()
#     print("Device name:", torch.cuda.get_device_name(device))
#     props = torch.cuda.get_device_properties(device)
#     print(f"Total memory: {props.total_memory / (1024 ** 3):.2f} GB")
#     print(f"Memory allocated: {torch.cuda.memory_allocated(device) / (1024 ** 3):.2f} GB")
#     print(f"Memory reserved: {torch.cuda.memory_reserved(device) / (1024 ** 3):.2f} GB")
# else:
#     print("No CUDA device")


