# evaluation/attention_perturbation_llama.py

import torch
import numpy as np
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaAttention
from torch.nn import functional as F

MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"

def normalize_token(token):
    return re.sub(r'\W+', '', token.replace("▁", "").lower())

def get_token_indices(tokenizer, text, target_tokens):
    tokens = tokenizer.tokenize(text)
    norm_map = {normalize_token(tok): idx for idx, tok in enumerate(tokens)}
    return [norm_map[tok] for tok in target_tokens if tok in norm_map]

class CustomLlamaAttention(LlamaAttention):
    def __init__(self, config, target_indices=None):
        super().__init__(config)
        self.target_indices = target_indices or []

    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)
        if self.target_indices and output[1] is not None:
            attention_probs = output[1]  # shape: [batch, heads, query_len, key_len]
            for idx in self.target_indices:
                attention_probs[:, :, :, idx] = 0.0
            attention_probs = attention_probs / attention_probs.sum(dim=-1, keepdim=True)
            return output[0], attention_probs
        return output

def replace_llama_attention(model, target_indices):
    for layer in model.model.layers:
        custom_attn = CustomLlamaAttention(model.config, target_indices)
        custom_attn.load_state_dict(layer.self_attn.state_dict(), strict=False)
        layer.self_attn = custom_attn

def compute_attention_perturbed_confidence(text, target_tokens):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        output_attentions=True,
        device_map="auto",
        torch_dtype=torch.float16
    )
    model.eval()

    print(f"🧠 Model loaded with dtype: {model.dtype}")
    print(f"📍 Device map: {model.hf_device_map}")

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    token_ids = inputs["input_ids"][0]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    target_indices = get_token_indices(tokenizer, text, target_tokens)
    if not target_indices:
        print("⚠️ No valid token indices to perturb.")
        return 0.0

    replace_llama_attention(model, target_indices)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        confidence = torch.max(probs).item()
    
    return confidence
