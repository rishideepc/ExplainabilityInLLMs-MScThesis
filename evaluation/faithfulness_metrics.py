import re
import numpy as np

try:
    alpha, beta, gamma, delta = np.load("faith_multi_weights.npy")
except:
    alpha = beta = gamma = delta = 0.25

def normalize_tokens(token_list):
    return set(
        re.sub(r'\W+', '', token.replace("##", "").lower())
        for token in token_list if token.strip()
    )

def jaccard_similarity(set1, set2):
    set1_norm = normalize_tokens(set1)
    set2_norm = normalize_tokens(set2)
    intersection = len(set1_norm & set2_norm)
    union = len(set1_norm | set2_norm)
    return intersection / union if union != 0 else 0.0

def compute_faith_multi(full_conf, explanation_tokens, erased_confs,
                        suff_conf, attrib_tokens, causal_tokens, suff_tokens,
                        alpha=0.25, beta=0.25, gamma=0.25, delta=0.25):
    
    FAITH_ATTRIB = jaccard_similarity(explanation_tokens, attrib_tokens)
    FAITH_CAUSAL = sum([abs(full_conf - ec) for ec in erased_confs]) / len(erased_confs) if erased_confs else 0.0
    FAITH_SUFF = suff_conf / full_conf if full_conf > 0 else 0.0

    attrib_set = normalize_tokens(attrib_tokens)
    causal_set = normalize_tokens(causal_tokens)
    suff_set = normalize_tokens(suff_tokens)

    intersection = len(attrib_set & causal_set & suff_set)
    union = len(attrib_set | causal_set | suff_set)
    ALIGN_CROSS = intersection / union if union > 0 else 0.0

    FAITH_MULTI = alpha * FAITH_ATTRIB + beta * FAITH_CAUSAL + gamma * FAITH_SUFF + delta * ALIGN_CROSS

    return {
        "FAITH_ATTRIB": FAITH_ATTRIB,
        "FAITH_CAUSAL": FAITH_CAUSAL,
        "FAITH_SUFF": FAITH_SUFF,
        "ALIGN_CROSS": ALIGN_CROSS,
        "FAITH_MULTI": FAITH_MULTI
    }

def compute_faith_multi_tuned(full_conf, explanation_tokens, erased_confs,
                              suff_conf, attrib_tokens, causal_tokens,
                              suff_tokens):
    comps = compute_faith_multi(full_conf, explanation_tokens, erased_confs,
                                suff_conf, attrib_tokens, causal_tokens, suff_tokens)
    comps["FAITH_MULTI_TUNED"] = (
        alpha * comps["FAITH_ATTRIB"] +
        beta * comps["FAITH_CAUSAL"] +
        gamma * comps["FAITH_SUFF"] +
        delta * comps["ALIGN_CROSS"]
    )
    return comps
