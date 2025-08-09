# argllm_generator_claim_verification.py

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
SAVE_PATH = os.path.join(PROJECT_ROOT, "results", "generation", "argLLM_generation", "truthfulclaim_mistral_temp_2", "argllm_outputs_ollama.jsonl")
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

from argument_miner import ArgumentMiner
from uncertainty_estimator import UncertaintyEstimator
from llm_managers import HuggingFaceLlmManager
import Uncertainpy.src.uncertainpy.gradual as grad
from prompt import ArgumentMiningPrompts, UncertaintyEvaluatorPrompts

# === LLM Config ===
# MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
# MODEL_NAME = "Qwen/Qwen3-8B"
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

with open(os.path.join(PROJECT_ROOT, "generators", "truthful_claims_dataset.json"), "r") as f:
    dataset = json.load(f)
dataset = dataset[:50]  # or whatever size you want


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
            "id": entry["id"],
            "question": entry["question"],
            # "option": entry["option"],
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

# # import torch
# # print("Torch version:", torch.__version__)
# # print("CUDA available:", torch.cuda.is_available())

# # if torch.cuda.is_available():
# #     device = torch.cuda.current_device()
# #     print("Device name:", torch.cuda.get_device_name(device))
# #     props = torch.cuda.get_device_properties(device)
# #     print(f"Total memory: {props.total_memory / (1024 ** 3):.2f} GB")
# #     print(f"Memory allocated: {torch.cuda.memory_allocated(device) / (1024 ** 3):.2f} GB")
# #     print(f"Memory reserved: {torch.cuda.memory_reserved(device) / (1024 ** 3):.2f} GB")
# # else:
# #     print("No CUDA device")





# import os
# import sys
# import json
# import re
# import torch
# from datasets import load_dataset
# import time

# start_time = time.time()

# # Set project root and save path
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
# sys.path.append(PROJECT_ROOT)
# SAVE_PATH = os.path.join(PROJECT_ROOT, "results", "generation", "argLLM_generation", "truthfulclaim_mistral_temp_3", "argllm_outputs_ollama.jsonl")
# os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

# from argument_miner import ArgumentMiner
# from uncertainty_estimator import UncertaintyEstimator
# from llm_managers import HuggingFaceLlmManager
# import Uncertainpy.src.uncertainpy.gradual as grad
# from prompt import ArgumentMiningPrompts, UncertaintyEvaluatorPrompts

# # ================================================================
# # Rebuttal Heuristic: add simple counter-attacks (supporter -> attacker)
# # ================================================================
# def add_rebuttals_if_missing(bag, k_defenders=1, min_strength_gap=0.05, enable=True):
#     """
#     Add simple counter-attacks: let the strongest supporters of 'db0' rebut each attacker of 'db0',
#     if the attacker has no incoming attacks yet.

#     Heuristic:
#       - Identify attackers of target 'db0'
#       - Sort supporters of 'db0' by their 'strength'
#       - For each attacker with no incoming attacks, add up to k_defenders edges: supporter -> attacker
#       - Only add if supporter.strength >= attacker.strength + min_strength_gap

#     bag format (as produced by t_xxx.to_dict()):
#       bag['arguments']: {arg_id: {'name': ..., 'argument': ..., 'initial_weight': float, 'strength': float, ...}, ...}
#       bag['attacks']: List[[attacker, attacked], ...]
#       bag['supports']: List[[supporter, supported], ...]
#     """
#     if not enable or not bag:
#         return bag

#     arguments = bag.get("arguments", {})
#     attacks = bag.get("attacks", []) or []
#     supports = bag.get("supports", []) or []

#     # Identify attackers/supporters of db0
#     attackers_of_db0 = [a for (a, t) in attacks if t == "db0"]
#     supporters_of_db0 = [s for (s, t) in supports if t == "db0"]

#     if not attackers_of_db0 or not supporters_of_db0:
#         return bag

#     # attacked_map: attacked -> [attackers]
#     attacked_map = {}
#     for a, t in attacks:
#         attacked_map.setdefault(t, []).append(a)

#     def strength_of(arg_id):
#         meta = arguments.get(arg_id, {})
#         # prefer 'strength', fallback to 'initial_weight'
#         return float(meta.get("strength", meta.get("initial_weight", 0.0)))

#     # strongest supporters first
#     supporters_sorted = sorted(supporters_of_db0, key=lambda s: strength_of(s), reverse=True)

#     new_edges = []
#     for attacker in attackers_of_db0:
#         # skip if attacker already has at least one incoming attack (already rebutted)
#         if attacker in attacked_map:
#             continue

#         picked = 0
#         for s in supporters_sorted:
#             if picked >= k_defenders:
#                 break
#             if strength_of(s) >= strength_of(attacker) + min_strength_gap:
#                 # store as list-of-list to match existing JSON style
#                 new_edges.append([s, attacker])
#                 picked += 1

#     if new_edges:
#         attacks_extended = attacks + new_edges
#         bag["attacks"] = attacks_extended

#     return bag


# # === LLM Config ===
# # MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
# MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
# # MODEL_NAME = "Qwen/Qwen3-8B"
# QUANTIZATION = "none"  # use "4bit" or "8bit" for low VRAM GPUs
# USE_CUDA = torch.cuda.is_available()
# DEVICE = "cuda" if USE_CUDA else "cpu"
# print(f"Device selected: {DEVICE}")

# # === Dataset ===
# print("Loading claim verification dataset...")
# with open(os.path.join(PROJECT_ROOT, "generators", "truthful_claims_dataset.json"), "r") as f:
#     dataset = json.load(f)
# dataset = dataset[:50]  # adjust as needed

# # === HuggingFace Pipeline ===
# print("Initializing LLM...")
# llm_manager = HuggingFaceLlmManager(
#     model_name=MODEL_NAME,
#     quantization=QUANTIZATION,
#     cache_dir=os.path.join(PROJECT_ROOT, "cache"),
#     input_device=DEVICE,
# )

# print("✅ LLM device:", llm_manager.input_device)

# # === GPU Model Device Check ===
# try:
#     model_device = next(llm_manager.model.parameters()).device
#     print(f"Model parameters are on device: {model_device}")
# except AttributeError:
#     print("[WARN] Could not access llm_manager.model to check device.")
# except Exception as e:
#     print(f"[ERROR] Unexpected error while checking model device: {e}")

# print(f"CUDA available: {torch.cuda.is_available()}")
# if torch.cuda.is_available():
#     print(f"GPU name: {torch.cuda.get_device_name(0)}")

# # === Certainty Formatter Patch ===
# def robust_certainty_formatter(certainty):
#     try:
#         match = re.search(r"([01](\.\d+)?)", certainty)
#         if match:
#             return float(match.group(1))
#         else:
#             print("[WARN] No valid certainty float found. Using default 0.5")
#             return 0.5
#     except Exception as e:
#         print(f"[ERROR] Certainty score parsing failed: {e}")
#         return 0.5

# # Monkey-patch prompt with safe formatter
# original_prompt_fn = UncertaintyEvaluatorPrompts.chatgpt
# def patched_chatgpt(statement, verbal=False, **_):
#     prompt, constraints, _ = original_prompt_fn(statement, verbal=verbal)
#     return prompt, constraints, robust_certainty_formatter
# UncertaintyEvaluatorPrompts.chatgpt = patched_chatgpt

# # === Argument Miner + Uncertainty Estimator ===
# generation_args = {
#     "temperature": 0.7,
#     "max_new_tokens": 128,
#     "top_p": 0.95,
# }

# print("Instantiating generators...")
# am = ArgumentMiner(
#     llm_manager=llm_manager,
#     generate_prompt=ArgumentMiningPrompts.chatgpt,
#     depth=2,
#     breadth=1,
#     generation_args=generation_args
# )

# ue = UncertaintyEstimator(
#     llm_manager=llm_manager,
#     generate_prompt=UncertaintyEvaluatorPrompts.chatgpt,
#     verbal=False,
#     generation_args=generation_args
# )

# # === Semantics: Aggregation + Influence Functions ===
# agg_f = grad.semantics.modular.ProductAggregation()
# inf_f = grad.semantics.modular.LinearInfluence(conservativeness=1)

# # === Main Generation Loop ===
# results = []
# for idx, entry in enumerate(dataset, 1):
#     try:
#         claim = entry["claim"]
#         print(f"\n[{idx}/{len(dataset)}] Processing claim: {claim}")

#         # Mine base/estimated argument bags (graphs)
#         t_base, t_estimated = am.generate_arguments(claim, ue)

#         # Propagate strengths on internal graph objects
#         print("📈 Computing strength propagation...")
#         grad.algorithms.computeStrengthValues(t_base, agg_f, inf_f)
#         grad.algorithms.computeStrengthValues(t_estimated, agg_f, inf_f)

#         # Convert to JSON-serializable dicts
#         base_bag = t_base.to_dict()
#         est_bag = t_estimated.to_dict()

#         # >>> Inject rebuttals (supporter -> attacker) to enrich argumentation structure <<<
#         # This does not change previously computed strengths; it augments attacks so that downstream
#         # acceptability metrics can register defenses against attackers of db0.
#         base_bag = add_rebuttals_if_missing(base_bag, k_defenders=2, min_strength_gap=0.05, enable=True)
#         est_bag  = add_rebuttals_if_missing(est_bag,  k_defenders=2, min_strength_gap=0.05, enable=True)

#         results.append({
#             "id": entry["id"],
#             "question": entry.get("question"),
#             "claim": entry["claim"],
#             "label": entry["label"],
#             "base": {
#                 "bag": base_bag,
#                 "prediction": t_base.arguments["db0"].strength,
#             },
#             "estimated": {
#                 "bag": est_bag,
#                 "prediction": t_estimated.arguments["db0"].strength,
#             },
#         })

#     except Exception as e:
#         # Fix: reference the actual 'claim' string (question was undefined here before)
#         print(f"❌ ERROR while processing claim: {entry.get('claim')} \n{e}")

# # === Save Outputs ===
# print("Saving results...")
# with open(SAVE_PATH, "w") as f:
#     for item in results:
#         json.dump(item, f)
#         f.write("\n")
# print(f"Saved {len(results)} entries to {SAVE_PATH}")

# elapsed_time = time.time() - start_time
# elapsed_hours, rem = divmod(elapsed_time, 3600)
# elapsed_minutes, elapsed_seconds = divmod(rem, 60)
# print(f"⏱️ Script completed in {int(elapsed_hours):02d}:{int(elapsed_minutes):02d}:{int(elapsed_seconds):02d}")
