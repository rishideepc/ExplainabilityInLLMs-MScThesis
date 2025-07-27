# evaluation/shake_score_core.py

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_confidence_and_label(model, tokenizer, input_text):
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(DEVICE)
    outputs = model(**inputs)
    logits = outputs.logits
    probs = F.softmax(logits, dim=-1)
    conf, label = torch.max(probs, dim=-1)
    return label.item(), conf.item()

def compute_shake_score(label, conf_orig, conf_pert):
    if conf_pert < conf_orig and label == 1:
        return 1 - (conf_orig - conf_pert)
    elif conf_pert > conf_orig and label == 0:
        return 1 - (conf_pert - conf_orig)
    return 1.0  # fallback
