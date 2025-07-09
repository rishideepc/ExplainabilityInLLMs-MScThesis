# evaluation/tune_weights_selfsupervised.py

import numpy as np
from datasets import load_dataset
from faithfulness_metrics import compute_faith_multi
from faithfulness_pipeline import FaithMultiPipeline
from perturbation_utils import compute_causal_impact

# Load the TruthfulQA dataset from Hugging Face (using mc1 or mc2 variant)
dataset = load_dataset("truthful_qa", "generation")["validation"]
print(f"✅ Loaded {len(dataset)} items from TruthfulQA")

# Initialize FaithMulti pipeline (LLM-based explanation + probing)
pipeline = FaithMultiPipeline()

X = []
MAX_SAMPLES = 100  # Optional cap for quick testing/debugging

for idx, example in enumerate(dataset):
    if idx >= MAX_SAMPLES:
        break

    question = example["question"]
    print(f"\n🔎 Processing sample {idx + 1}: {question.strip()[:60]}...")

    try:
        label, full_conf = pipeline.get_prediction_and_confidence(question)
        explanation = pipeline.generate_explanation(question)
        expl_tokens = pipeline.tokenize_explanation(explanation)
        attrib_tokens = pipeline.get_attribution_tokens(question)
        suff = pipeline.get_suff_confidence(expl_tokens, label)
        causal = compute_causal_impact(pipeline, question, attrib_tokens, label, full_conf)

        feat = [
            compute_faith_multi(
                full_conf,
                expl_tokens,
                [full_conf - causal] * len(attrib_tokens),
                suff,
                attrib_tokens,
                attrib_tokens,
                expl_tokens
            )[m] for m in ["FAITH_ATTRIB", "FAITH_CAUSAL", "FAITH_SUFF", "ALIGN_CROSS"]
        ]
        X.append(feat)
    
    except Exception as e:
        print(f"⚠️ Skipping sample due to error: {e}")
        continue

X = np.array(X)
print(f"\n✅ Extracted faithfulness components for {len(X)} samples.")

# Self-supervised variance-based weight tuning
variances = np.var(X, axis=0)
inv_var = 1 / (variances + 1e-5)
weights = inv_var / np.sum(inv_var)
np.save("faith_multi_weights.npy", weights)

print("\n✅ Learned FAITH_MULTI weights via self-supervised tuning:")
print(f"  α (ATTRIB):     {weights[0]:.4f}")
print(f"  β (CAUSAL):     {weights[1]:.4f}")
print(f"  γ (SUFF):       {weights[2]:.4f}")
print(f"  δ (ALIGN_CROSS):{weights[3]:.4f}")
