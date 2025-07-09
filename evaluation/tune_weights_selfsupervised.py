# evaluation/tune_weights_selfsupervised.py

import numpy as np
import random
from datasets import load_dataset
from faithfulness_metrics import compute_faith_multi
from faithfulness_pipeline import FaithMultiPipeline
from perturbation_utils import compute_causal_impact

# Load and split TruthfulQA
dataset = load_dataset("truthful_qa", "generation")["validation"]
questions = [example["question"] for example in dataset]
random.seed(42)
random.shuffle(questions)
split_idx = int(0.7 * len(questions))
train_questions = questions[:split_idx]
test_questions = questions[split_idx:]
print(f"✅ Loaded {len(questions)} total | Train: {len(train_questions)} | Test: {len(test_questions)}")

pipeline = FaithMultiPipeline()
X_train = []

for idx, question in enumerate(train_questions):  # Optional cap for training
    print(f"\n🔧 Training sample {idx + 1}: {question.strip()[:60]}...")
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
        X_train.append(feat)

        # Update weights dynamically (just for visibility)
        X_np = np.array(X_train)
        var = np.var(X_np, axis=0)
        inv_var = 1 / (var + 1e-5)
        weights = inv_var / np.sum(inv_var)
        print(f"Updated weights: α={weights[0]:.3f}, β={weights[1]:.3f}, γ={weights[2]:.3f}, δ={weights[3]:.3f}")

    except Exception as e:
        print(f"⚠️ Skipped due to error: {e}")
        continue

# Save final learned weights
np.save("faith_multi_weights.npy", weights)
print("\n✅ Final learned weights saved.")
