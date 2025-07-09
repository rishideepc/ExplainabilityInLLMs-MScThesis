import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertConfig
from transformers.models.bert.modeling_bert import BertSelfAttention
from torch.nn import functional as F

# ✅ Use a pretrained classification model
MODEL_NAME = "textattack/bert-base-uncased-SST-2"

def normalize_token(token):
    return token.replace("##", "").lower().strip()

def get_token_indices(tokenizer, text, target_tokens):
    tokens = tokenizer.tokenize(text)
    norm_map = {normalize_token(tok): idx for idx, tok in enumerate(tokens)}
    return [norm_map[tok] for tok in target_tokens if tok in norm_map]

class CustomBertSelfAttention(BertSelfAttention):
    def __init__(self, config, target_indices=None):
        super().__init__(config)
        self.target_indices = target_indices or []

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # Call the parent forward() to get standard outputs
        output = super().forward(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions
        )

        # Now modify attention_probs AFTER they’ve been computed
        if self.target_indices and output_attentions:
            attention_probs = output[1]  # attention_probs
            for idx in self.target_indices:
                attention_probs[:, :, :, idx] = 0.0
            # renormalize if needed
            attention_probs = attention_probs / attention_probs.sum(dim=-1, keepdim=True)
            output = (output[0], attention_probs)  # replace modified attention_probs

        return output


def replace_self_attention(model, target_indices):
    """
    Monkeypatch attention modules in all BERT layers.
    """
    for layer in model.bert.encoder.layer:
        custom_attn = CustomBertSelfAttention(model.config, target_indices)
        custom_attn.load_state_dict(layer.attention.self.state_dict(), strict=False)
        layer.attention.self = custom_attn

def compute_attention_perturbed_confidence(text, target_tokens):
    """
    Replaces model self-attention to simulate internal perturbation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    target_indices = get_token_indices(tokenizer, text, target_tokens)

    if not target_indices:
        print("⚠️ No valid token indices to perturb.")
        return 0.0

    # Inject perturbed attention
    replace_self_attention(model, target_indices)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        confidence = torch.max(probs).item()

    return confidence
