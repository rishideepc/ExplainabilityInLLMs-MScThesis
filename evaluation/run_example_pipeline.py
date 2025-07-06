# evaluation/run_example_pipeline.py

from faithfulness_pipeline import FaithMultiPipeline
from faithfulness_metrics import compute_faith_multi
from perturbation_utils import compute_causal_impact

text = "Fortune cookies originated in Italy"
pipeline = FaithMultiPipeline(model="llama3")

label, full_conf = pipeline.get_prediction_and_confidence(text)
print(f"✅ Prediction: {label} @ {full_conf:.2f}")

explanation = pipeline.generate_explanation(text)
expl_tokens = pipeline.tokenize_explanation(explanation)
attrib_tokens = pipeline.get_attribution_tokens(text)

# Now compute causal impact via attribution tokens
causal_score = compute_causal_impact(pipeline, text, attrib_tokens, label, full_conf)

suff_score = pipeline.get_suff_confidence(expl_tokens, label)

scores = compute_faith_multi(
    full_conf=full_conf,
    explanation_tokens=expl_tokens,
    erased_confs=[full_conf - causal_score] * len(attrib_tokens),  # simulate erased confidence
    suff_conf=suff_score,
    attrib_tokens=attrib_tokens,
    causal_tokens=attrib_tokens,
    suff_tokens=expl_tokens
)

print("🔍 FAITH_MULTI Breakdown:")
for k, v in scores.items():
    print(f"{k}: {v:.4f}")
