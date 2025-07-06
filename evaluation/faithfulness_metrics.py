# evaluation/faithfulness_metrics.py

import numpy as np
from unified_token_extraction import extract_all_token_sets

# load tuned weights if available
try:
    alpha, beta, gamma, delta = np.load("faith_multi_weights.npy")
except:
    alpha = beta = gamma = delta = 0.25

def compute_faith_multi(
    full_conf,
    explanation_tokens,
    erased_confs,
    suff_conf,
    attrib_tokens,
    causal_tokens,
    suff_tokens,
    alpha=0.25, beta=0.25, gamma=0.25, delta=0.25
):
    FAITH_ATTRIB = jaccard_similarity(explanation_tokens, attrib_tokens)
    FAITH_CAUSAL = sum([abs(full_conf - ec) for ec in erased_confs]) / len(erased_confs) if erased_confs else 0.0
    FAITH_SUFF = suff_conf / full_conf if full_conf > 0 else 0.0
    ALIGN_CROSS = extract_all_token_sets(explanation_tokens, attrib_tokens, causal_tokens, suff_tokens)['align_cross']

    FAITH_MULTI = (
        alpha * FAITH_ATTRIB +
        beta * FAITH_CAUSAL +
        gamma * FAITH_SUFF +
        delta * ALIGN_CROSS
    )

    return {
        "FAITH_ATTRIB": FAITH_ATTRIB,
        "FAITH_CAUSAL": FAITH_CAUSAL,
        "FAITH_SUFF": FAITH_SUFF,
        "ALIGN_CROSS": ALIGN_CROSS,
        "FAITH_MULTI": FAITH_MULTI
    }

def jaccard_similarity(t1, t2):
    set1 = set(t1)
    set2 = set(t2)
    return len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0.0
