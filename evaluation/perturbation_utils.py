# evaluation/perturbation_utils.py

from attention_perturbation import compute_attention_perturbed_confidence
from faithfulness_pipeline import FaithMultiPipeline

def compute_causal_impact(pipeline: FaithMultiPipeline, text, tokens, expected_label, orig_conf):
    """
    Attention-based causal probing using LLaMA and rationale tokens.
    """
    try:
        perturbed_conf = compute_attention_perturbed_confidence(text, tokens)
        delta = abs(orig_conf - perturbed_conf)
        return delta
    except Exception as e:
        print(f"⚠️ Causal probing failed: {e}")
        return 0.0
