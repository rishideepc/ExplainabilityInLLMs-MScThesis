import os
import sys
import json
import re
import torch
from datasets import load_dataset
import time

start_time= time.time()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.append(PROJECT_ROOT)
SAVE_PATH = os.path.join(PROJECT_ROOT, "results", "generation", "argLLM_generation", "truthfulclaim_mistral", "argllm_outputs_ollama.jsonl")
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

from argument_miner import ArgumentMiner
from uncertainty_estimator import UncertaintyEstimator
from llm_managers import HuggingFaceLlmManager
import Uncertainpy.src.uncertainpy.gradual as grad
from prompt import ArgumentMiningPrompts, UncertaintyEvaluatorPrompts

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
QUANTIZATION = "none"
USE_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if USE_CUDA else "cpu"
print(f"Device selected: {DEVICE}")

print("Loading claim verification dataset...")
import json
with open(os.path.join(PROJECT_ROOT, "generators", "truthful_claims_dataset.json"), "r") as f:
    dataset = json.load(f)
dataset = dataset[:50]

print("Initializing LLM...")
llm_manager = HuggingFaceLlmManager(
    model_name=MODEL_NAME,
    quantization=QUANTIZATION,
    cache_dir=os.path.join(PROJECT_ROOT, "cache"),
    input_device=DEVICE,
)
print("LLM device:", llm_manager.input_device)

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

def robust_certainty_formatter(certainty):
    """
    Parses a float certainty from text

    @params: certainty

    @returns: float value in range [0,1] or fallback 0.5
    """
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

original_prompt_fn = UncertaintyEvaluatorPrompts.chatgpt
def patched_chatgpt(statement, verbal=False, **_):
    """
    Returns uncertainty prompt + constraints + robust formatter

    @params: statement, verbal

    @returns: (prompt, constraints, formatter)
    """
    prompt, constraints, _ = original_prompt_fn(statement, verbal=verbal)
    return prompt, constraints, robust_certainty_formatter
UncertaintyEvaluatorPrompts.chatgpt = patched_chatgpt

generation_args = {
    "temperature": 0.7,
    "max_new_tokens": 128,
    "top_p": 0.95,
}

print("Instantiating generators...")
am = ArgumentMiner(
    llm_manager=llm_manager,
    generate_prompt=ArgumentMiningPrompts.chatgpt,
    depth=2,
    breadth=1,
    generation_args=generation_args
)

ue = UncertaintyEstimator(
    llm_manager=llm_manager,
    generate_prompt=UncertaintyEvaluatorPrompts.chatgpt,
    verbal=False,
    generation_args=generation_args
)

agg_f = grad.semantics.modular.ProductAggregation()
inf_f = grad.semantics.modular.LinearInfluence(conservativeness=1)

results = []
for idx, entry in enumerate(dataset, 1):
    try:
        claim = entry["claim"]
        print(f"\n[{idx}/{len(dataset)}] Processing claim: {claim}")
        t_base, t_estimated = am.generate_arguments(claim, ue)

        print("Computing strength propagation...")
        grad.algorithms.computeStrengthValues(t_base, agg_f, inf_f)
        grad.algorithms.computeStrengthValues(t_estimated, agg_f, inf_f)

        results.append({
            "id": entry["id"],
            "question": entry["question"],
            "claim": entry["claim"],
            "label": entry["label"],
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
        print(f"ERROR while processing: {question}\n{e}")

print("Saving results...")
with open(SAVE_PATH, "w") as f:
    for item in results:
        json.dump(item, f)
        f.write("\n")
print(f"Saved {len(results)} entries to {SAVE_PATH}")

elapsed_time = time.time() - start_time
elapsed_hours, rem = divmod(elapsed_time, 3600)
elapsed_minutes, elapsed_seconds = divmod(rem, 60)
print(f"Script completed in {int(elapsed_hours):02d}:{int(elapsed_minutes):02d}:{int(elapsed_seconds):02d}")
