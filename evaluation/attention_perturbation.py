import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.nn import functional as F

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

def normalize_token(token):
    return token.replace("##", "").lower().strip()

def get_token_indices(tokenizer, text, target_tokens):
    tokens = tokenizer.tokenize(text)
    norm_map = {normalize_token(tok): idx for idx, tok in enumerate(tokens)}
    return [norm_map[tok] for tok in target_tokens if tok in norm_map]

class AttentionHook:
    """
    This class is used to inject perturbed attention scores into the attention layers
    during forward pass by overriding `forward` method.
    """
    def __init__(self, target_indices):
        self.target_indices = target_indices
        self.handles = []

    def register(self, model):
        for layer in model.distilbert.transformer.layer:
            attn_module = layer.attention
            handle = attn_module.register_forward_hook(self._hook_fn)
            self.handles.append(handle)

    def _hook_fn(self, module, inputs, output):
        """
        Hook function to modify attention scores.
        inputs = (query, key, value)
        output = attention_output
        """
        attn_scores = module.attention_scores  # Shape: (batch, heads, seq_len, seq_len)
        with torch.no_grad():
            for idx in self.target_indices:
                attn_scores[:, :, :, idx] = -1e9  # Effectively zero attention after softmax
        return output

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

def compute_attention_perturbed_confidence(text, target_tokens):
    """
    Runs the model with internal attention perturbation on target_tokens
    by directly modifying attention maps via forward hooks.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, attn_implementation="eager").to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    target_indices = get_token_indices(tokenizer, text, target_tokens)

    if not target_indices:
        print("⚠️ No valid token indices to perturb.")
        return 0.0

    # Attach hook to override attention
    hook = AttentionHook(target_indices)
    hook.register(model)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        confidence = torch.max(probs).item()

    # Remove hook after pass
    hook.remove()
    return confidence
