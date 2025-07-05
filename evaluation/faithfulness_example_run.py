from faithfulness_pipeline import FaithMultiPipeline
from faithfulness_metrics import compute_faith_multi

text = "Fortune cookies originated in Italy"
pipeline = FaithMultiPipeline(model="llama3")

label, full_conf = pipeline.get_prediction_and_confidence(text)
explanation = pipeline.generate_explanation(text)
expl_tokens = pipeline.tokenize_explanation(explanation)

erased_confs = [pipeline.erase_token_and_get_confidence(text, t, label) for t in expl_tokens]
suff_conf = pipeline.get_suff_confidence(expl_tokens, label)
attrib_tokens = pipeline.get_attribution_tokens(text)

results = compute_faith_multi(
    full_conf=full_conf,
    explanation_tokens=expl_tokens,
    erased_confs=erased_confs,
    suff_conf=suff_conf,
    attrib_tokens=attrib_tokens,
    causal_tokens=expl_tokens,
    suff_tokens=expl_tokens
)

print("\n🔍 Final FAITH_MULTI Metrics:")
for k, v in results.items():
    print(f"{k}: {v:.4f}")
