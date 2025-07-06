import torch
import numpy as np
import re
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

def normalize_token(token: str) -> str:
    """Clean individual token by stripping punctuation and subword markers."""
    return re.sub(r'\W+', '', token.replace("##", "").lower())

def normalize_tokens(tokens: list) -> list:
    """Normalize a list of tokens."""
    return [normalize_token(t) for t in tokens if normalize_token(t)]

def get_top_attributed_tokens(text: str, k: int = 3) -> list:
    """
    Get top-k attributed tokens using fallback logic:
    1. Gradient-based attribution
    2. Attention-based backup
    3. Heuristic fallback
    """
    try:
        return _gradient_attribution(text, k)
    except Exception as e:
        print(f"⚠️ Gradient attribution failed: {e}")
        try:
            return _attention_based_attribution(text, k)
        except Exception as e2:
            print(f"⚠️ Attention attribution failed: {e2}")
            return _heuristic_attribution(text, k)

def _gradient_attribution(text: str, k: int) -> list:
    """
    Compute token importance using gradient norms on embeddings.
    Returns top-k tokens based on L2-norm of gradient.
    """
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    input_ids = inputs["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    embeddings = model.get_input_embeddings()(input_ids)
    embeddings.requires_grad_(True)

    outputs = model(inputs_embeds=embeddings, attention_mask=inputs["attention_mask"])
    pred_class = outputs.logits.argmax(dim=-1)
    score = outputs.logits[0, pred_class]

    gradients = torch.autograd.grad(
        outputs=score,
        inputs=embeddings,
        retain_graph=False,
        create_graph=False
    )[0]

    # L2 norm of gradients across hidden dim
    token_scores = gradients.norm(dim=-1).squeeze(0).cpu().detach().numpy()
    scored = list(zip(tokens, token_scores))
    sorted_tokens = sorted(scored, key=lambda x: x[1], reverse=True)

    top_tokens = [normalize_token(t) for t, _ in sorted_tokens if normalize_token(t)]
    return top_tokens[:k]

def _attention_based_attribution(text: str, k: int) -> list:
    """
    Use last-layer CLS attention to rank tokens by importance.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, output_attentions=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    with torch.no_grad():
        outputs = model(**inputs)

    # Extract last layer, average over heads
    attention = outputs.attentions[-1]  # shape: (1, heads, tokens, tokens)
    cls_attention = attention[0, :, 0, :].mean(dim=0)  # mean over heads

    scores = cls_attention.cpu().numpy()
    scored = list(zip(tokens, scores))
    sorted_tokens = sorted(scored, key=lambda x: x[1], reverse=True)

    top_tokens = [normalize_token(t) for t, _ in sorted_tokens if normalize_token(t)]
    return top_tokens[:k]

def _heuristic_attribution(text: str, k: int) -> list:
    """
    Heuristic importance based on:
    - Capitalization
    - Negation words
    - Token length
    """
    negation = {"not", "never", "no", "none"}
    scored = []
    for token in text.split():
        score = len(token)
        if token[0].isupper():
            score += 2
        if token.lower() in negation:
            score += 3
        scored.append((token, score))
    
    sorted_tokens = sorted(scored, key=lambda x: x[1], reverse=True)
    top_tokens = [normalize_token(t) for t, _ in sorted_tokens if normalize_token(t)]
    return top_tokens[:k]
