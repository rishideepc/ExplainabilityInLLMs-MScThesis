import sys
import os
import json
import numpy as np
from sklearn.linear_model import LinearRegression

# Append project path
project_root = os.path.abspath('...')
sys.path.append(project_root)

from faithfulness_pipeline import FaithMultiPipeline
from faithfulness_metrics import compute_faith_multi

def load_storysumm_dataset(json_path="storysumm.json", split="val", max_samples=200):
    with open(json_path, "r") as f:
        data = json.load(f)
    
    samples = []
    for item in data.values():
        if item.get("split") == split and isinstance(item.get("summary"), list):
            samples.append(item)
        if len(samples) >= max_samples:
            break
    return samples

def prepare_data(json_path="storysumm.json", split="val", n_samples=200):
    data = load_storysumm_dataset(json_path=json_path, split=split, max_samples=n_samples)
    pipeline = FaithMultiPipeline(model="llama3")

    X, y = [], []
    for idx, sample in enumerate(data):
        try:
            story_text = sample["story"]
            summary_text = " ".join(sample["summary"])
            gold_label = sample["label"]  # 0 (unfaithful) or 1 (faithful)

            print(f"\n▶️ Sample {idx+1}:")
            print(f"Story (truncated): {story_text[:100]}...")
            print(f"Summary (truncated): {summary_text[:100]}...")
            print(f"Ground truth label: {gold_label}")

            # Run full faith pipeline
            label, full_conf = pipeline.get_prediction_and_confidence(summary_text)
            explanation = summary_text
            expl_tokens = pipeline.tokenize_explanation(explanation)
            erased = [pipeline.erase_token_and_get_confidence(summary_text, t, label) for t in expl_tokens]
            suff_conf = pipeline.get_suff_confidence(expl_tokens, label)
            attrib = pipeline.get_attribution_tokens(summary_text)

            comps = compute_faith_multi(
                full_conf, expl_tokens, erased, suff_conf, attrib,
                expl_tokens, expl_tokens
            )

            # Feature vector and label
            X.append([
                comps["FAITH_ATTRIB"],
                comps["FAITH_CAUSAL"],
                comps["FAITH_SUFF"],
                comps["ALIGN_CROSS"]
            ])
            y.append(gold_label)

        except Exception as e:
            print(f"⚠️ Skipping sample {idx} due to error: {e}")
            continue

    return np.array(X), np.array(y)

def train_weights(X, y):
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)
    print("✅ Learned weights (α, β, γ, δ):", model.coef_)
    return model.coef_

if __name__ == "__main__":
    X, y = prepare_data("storysumm.json", split="val", n_samples=200)
    weights = train_weights(X, y)
    np.save("faith_multi_weights.npy", weights)
    print("💾 Saved learned weights to faith_multi_weights.npy")
