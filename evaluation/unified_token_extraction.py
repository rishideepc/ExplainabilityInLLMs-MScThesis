# evaluation/unified_token_extraction.py

import re

def normalize_tokens(tokens):
    """Standardize tokens by lowercasing, removing punctuation and subword prefixes."""
    return [
        re.sub(r'\W+', '', token.replace("##", "").lower())
        for token in tokens if token.strip()
    ]

def token_overlap(set1, set2):
    s1 = set(normalize_tokens(set1))
    s2 = set(normalize_tokens(set2))
    return s1, s2, len(s1 & s2), len(s1 | s2)

def extract_all_token_sets(explanation, attribution, causal, suff):
    """Returns normalized sets and intersection/union for ALIGN_CROSS"""
    expl = normalize_tokens(explanation)
    attr = normalize_tokens(attribution)
    caus = normalize_tokens(causal)
    suff = normalize_tokens(suff)

    intersection = set(attr) & set(caus) & set(suff)
    union = set(attr) | set(caus) | set(suff)

    return {
        "explanation": expl,
        "attribution": attr,
        "causal": caus,
        "sufficiency": suff,
        "align_cross": len(intersection) / len(union) if union else 0.0
    }
