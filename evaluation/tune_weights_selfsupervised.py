# evaluation/tune_weights_selfsupervised.py

import numpy as np
from faithfulness_metrics import compute_faith_multi
from faithfulness_pipeline import FaithMultiPipeline
from perturbation_utils import compute_causal_impact

sample_texts = [
    "Water boils at 100 degrees Celsius.",
    "Elephants are the fastest land animals.",
    "The Eiffel Tower is located in Berlin.",
]

pipeline = FaithMultiPipeline()
X = []

for text in sample_texts:
    label, full_conf = pipeline.get_prediction_and_confidence(text)
    explanation = pipeline.generate_explanation(text)
    expl_tokens = pipeline.tokenize_explanation(explanation)
    attrib_tokens = pipeline.get_attribution_tokens(text)
    suff = pipeline.get_suff_confidence(expl_tokens, label)
    causal = compute_causal_impact(pipeline, text, attrib_tokens, label, full_conf)

    feat = [
        compute_faith_multi(full_conf, expl_tokens, [full_conf - causal] * len(attrib_tokens), suff,
                            attrib_tokens, attrib_tokens, expl_tokens)[m]
        for m in ["FAITH_ATTRIB", "FAITH_CAUSAL", "FAITH_SUFF", "ALIGN_CROSS"]
    ]
    X.append(feat)

X = np.array(X)
variances = np.var(X, axis=0)
inv_var = 1 / (variances + 1e-5)
weights = inv_var / np.sum(inv_var)
np.save("faith_multi_weights.npy", weights)

print("✅ Learned weights via self-supervised variance minimization:", weights)
