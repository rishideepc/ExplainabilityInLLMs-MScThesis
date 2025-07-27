# evaluation/attention_perturbation_llama.py

import torch
import re
from transformers.models.llama.modeling_llama import LlamaAttention

def normalize_token(token):
    return re.sub(r'\W+', '', token.replace("▁", "").lower())

def get_token_indices(tokenizer, text, target_words):
    tokens = tokenizer.tokenize(text)
    token_strings = [token.replace("▁", "") for token in tokens]
    norm_tokens = [token.lower() for token in token_strings]
    norm_targets = [word.lower() for word in target_words]

    print(f"🧩 Tokenized words: {norm_tokens}")
    print(f"🎯 Normalized rationale targets: {norm_targets}")

    indices = [i for i, tok in enumerate(norm_tokens) if tok in norm_targets]
    print(f"📌 Matched token indices: {indices}")
    return indices

class CustomLlamaAttention(LlamaAttention):
    def __init__(self, config, layer_idx, target_indices=None):
        super().__init__(config, layer_idx)
        self.target_indices = target_indices or []

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False, **kwargs):
        output = super().forward(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=True,
            use_cache=use_cache,
            **kwargs
        )
        if self.target_indices and output[1] is not None:
            attention_probs = output[1].to(hidden_states.device)
            print(f"🔧 Modifying attention weights for indices: {self.target_indices}")
            for idx in self.target_indices:
                attention_probs[:, :, :, idx] = 0.0
            attention_probs = attention_probs / attention_probs.sum(dim=-1, keepdim=True)
            return output[0], attention_probs
        return output

def replace_llama_attention(model, target_indices, last_n_layers=1):
    device = next(model.parameters()).device
    total_layers = len(model.model.layers)
    for i in range(total_layers):
        if i >= total_layers - last_n_layers:
            custom_attn = CustomLlamaAttention(model.config, i, target_indices)
            custom_attn.load_state_dict(model.model.layers[i].self_attn.state_dict(), strict=False)
            custom_attn = custom_attn.to(device).half()
            model.model.layers[i].self_attn = custom_attn
    print("✅ Replaced final attention layer(s) with CustomLlamaAttention")

def run_with_attention_perturbation(model, tokenizer, text, target_tokens):
    model.eval()
    model = model.half()
    prompt = (
    f"Q: Is the following claim true or false?\n"
    f"Claim: \"{text}\"\n\n"
    "Return your answer only as 'TRUE' or 'FALSE' and also provide your confidence from 0 to 100.\nA:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    for key in inputs:
        if inputs[key].dtype == torch.float:
            inputs[key] = inputs[key].to(dtype=torch.float16)

    token_ids = inputs["input_ids"][0]
    print(f"📥 input_ids device: {inputs['input_ids'].device}")
    print(f"🧠 model device: {next(model.parameters()).device}")

    target_indices = get_token_indices(tokenizer, text, target_tokens)

    if not target_indices:
        print("⚠️ No matching token indices found for rationale tokens.")
        return "UNKNOWN", 0.5

    replace_llama_attention(model, target_indices, last_n_layers=1)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10)
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"🗣️ Perturbed output: {output_text}")

    label = "TRUE" if "TRUE" in output_text.upper() else "FALSE" if "FALSE" in output_text.upper() else "UNKNOWN"
    # match = re.search(r"(\d{1,3})", output_text)
    match = re.search(r"[Cc]onfidence\s*[:=]?\s*(\d{1,3})", output_text)
    print(f"🧪 Confidence match: {match.group(1) if match else 'None'}")
    confidence = float(match.group(1)) / 100 if match else 0.5

    print(f"📊 Perturbed label: {label} | confidence: {confidence:.2f}")
    return label, confidence
