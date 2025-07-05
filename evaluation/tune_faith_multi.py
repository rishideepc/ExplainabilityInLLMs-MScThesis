import json
import numpy as np
from sklearn.linear_model import LinearRegression
from faithfulness_pipeline import FaithMultiPipeline
from faithfulness_metrics import compute_faith_multi

def load_storysumm_dataset(json_path="storysumm.json", split="val", max_samples=200):
    with open(json_path, "r") as f:
        data = json.load(f)
    return [item for item in data.values() if item.get("split") == split][:max_samples]

def prepare_data(json_path="storysumm.json", split="val", n_samples=200):
    data = load_storysumm_dataset(json_path, split, n_samples)
    pipeline = FaithMultiPipeline(model="llama3")

    X, y = [], []
    for sample in data:
        try:
            story = sample["story"]
            summary = " ".join(sample["summary"])
            label = sample["label"]

            pred_label, conf = pipeline.get_prediction_and_confidence(summary)
            explanation = summary
            expl_toks = pipeline.tokenize_explanation(explanation)
            erased = [pipeline.erase_token_and_get_confidence(summary, t, pred_label) for t in expl_toks]
            suff_conf = pipeline.get_suff_confidence(expl_toks, pred_label)
            attrib = pipeline.get_attribution_tokens(summary)

            comps = compute_faith_multi(conf, expl_toks, erased, suff_conf, attrib, expl_toks, expl_toks)
            X.append([comps["FAITH_ATTRIB"], comps["FAITH_CAUSAL"], comps["FAITH_SUFF"], comps["ALIGN_CROSS"]])
            y.append(label)

        except Exception as e:
            print(f"⚠️ Skipping due to error: {e}")
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
    print("💾 Saved weights to faith_multi_weights.npy")
