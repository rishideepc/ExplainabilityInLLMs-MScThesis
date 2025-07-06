# evaluation/perturbation_utils.py

# from copy import deepcopy
# from transformers import pipeline
# from typing import List
from attention_perturbation import compute_attention_perturbed_confidence
from faithfulness_pipeline import FaithMultiPipeline

# def generate_perturbed_inputs(text: str, tokens_to_perturb: List[str], replacement="[MASK]"):
#     perturbed = []
#     for tok in tokens_to_perturb:
#         new_text = text.replace(tok, replacement)
#         perturbed.append((tok, new_text))
#     return perturbed

def compute_causal_impact(pipeline: FaithMultiPipeline, text, tokens, expected_label, orig_conf):
    """
    Attention-based causal probing using attribution tokens.
    """
    try:
        perturbed_conf = compute_attention_perturbed_confidence(text, tokens)
        delta = abs(orig_conf - perturbed_conf)
        return delta
    except Exception as e:
        print(f"⚠️ Causal probing failed: {e}")
        return 0.0
