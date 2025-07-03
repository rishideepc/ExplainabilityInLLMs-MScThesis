import sys
import os
project_root = os.path.abspath('...')
sys.path.append(project_root)

from faithfulness_pipeline import FaithMultiPipeline
from faithfulness_metrics import compute_faith_multi
from generators.cot_generator import generate_with_ollama

# Usage variables
text = "Fortune cookies originated in Italy"
pipeline = FaithMultiPipeline(model="llama3")


# Get model prediction and confidence from LLM
label, full_conf = pipeline.get_prediction_and_confidence(text)
print(f"Prediction: {label} | Confidence: {full_conf:.2f}")



# Generate LLM explanation (rationale)
explanation = pipeline.generate_explanation(text)
print("Explanation:", explanation)


# Tokenize LLM explanation
expl_tokens = pipeline.tokenize_explanation(explanation)
print("Tokens:", expl_tokens)


# Compute confidence after counterfaactual test
erased_confs = [
    pipeline.erase_token_and_get_confidence(text, tok, label)
    for tok in expl_tokens
]


# Compute sufficiency confidence when only explanation is provided
suff_conf = pipeline.get_suff_confidence(expl_tokens, label)



# attrib_tokens = expl_tokens        # TODO: 1. use Inseq for highlighting highly attended tokens
attrib_tokens = pipeline.get_attribution_tokens(text, top_k=5)
print("ATtribution Tokens: ", attrib_tokens)
causal_tokens = expl_tokens        # Explanation tokens (counterfactual/causal)
suff_tokens = expl_tokens          # Explanation tokens (sufficiency)

# Weighted aggregate of metrics - FAITH_MULTI
results = compute_faith_multi(
    full_conf=full_conf,
    explanation_tokens=expl_tokens,
    erased_confs=erased_confs,
    suff_conf=suff_conf,
    attrib_tokens=attrib_tokens,
    causal_tokens=causal_tokens,
    suff_tokens=suff_tokens,
)

print("\n🔍 Final FAITH_MULTI Metrics:")
for k, v in results.items():
    print(f"{k}: {v:.4f}")



# TODO:
    # 2. incorporate trainable architecture