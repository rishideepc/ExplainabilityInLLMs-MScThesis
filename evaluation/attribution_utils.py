import re
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

def normalize_tokens(tokens):
    """Lowercase, remove subword prefixes and punctuation."""
    return [
        re.sub(r'\W+', '', token.replace("##", "").lower())
        for token in tokens if token.strip()
    ]

def get_top_attributed_tokens(text: str, k: int = 5) -> list:
    """
    Get top k attributed tokens using integrated gradients via Inseq.
    Fallbacks: gradient, attention, heuristic.
    """
    try:
        from inseq import load_model

        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        # Ensure required special tokens are set
        if tokenizer.bos_token is None:
            tokenizer.bos_token = tokenizer.cls_token
        if tokenizer.eos_token is None:
            tokenizer.eos_token = tokenizer.sep_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token

        attrib_model = load_model(
            model=model,
            tokenizer=tokenizer,
            attribution_method="integrated_gradients",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        output = attrib_model.attribute(
            text,
            step_scores=["logit"],
            internal_batch_size=1,
        )

        scores = output.attribution[0].sum(dim=-1).tolist()
        tokens = output.input_tokens[0]

        scored = list(zip(tokens, scores))
        sorted_scored = sorted(scored, key=lambda x: abs(x[1]), reverse=True)

        top_tokens = [
            token for token, _ in sorted_scored[:k]
            if token not in ["[CLS]", "[SEP]", "[PAD]", "[UNK]", tokenizer.pad_token, tokenizer.cls_token, tokenizer.sep_token]
        ]
        return normalize_tokens(top_tokens)

    except Exception as e:
        print(f"Inseq attribution failed: {e}")
        return _custom_gradient_attribution(text, k)

def _custom_gradient_attribution(text: str, k: int = 5) -> list:
    try:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        input_ids_float = inputs['input_ids'].float()
        input_ids_float.requires_grad_(True)

        embeddings = model.get_input_embeddings()(input_ids_float.round().long())
        outputs = model(inputs_embeds=embeddings, attention_mask=inputs['attention_mask'])

        predicted_class = outputs.logits.argmax(dim=-1)
        score = outputs.logits[0, predicted_class]

        gradients = torch.autograd.grad(
            outputs=score,
            inputs=input_ids_float,
            retain_graph=False,
            create_graph=False
        )[0]

        gradient_norms = gradients.abs().squeeze(0)
        token_scores = list(zip(tokens, gradient_norms.cpu().numpy()))
        sorted_tokens = sorted(token_scores, key=lambda x: x[1], reverse=True)

        top_tokens = [
            token for token, _ in sorted_tokens
            if token not in ["[CLS]", "[SEP]", "[PAD]", "[UNK]"]
        ][:k]

        return normalize_tokens(top_tokens)

    except Exception as e:
        print(f"Gradient method failed: {e}")
        return _attention_based_attribution(text, k)

def _attention_based_attribution(text: str, k: int = 5) -> list:
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            output_attentions=True,
            attn_implementation="eager"
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        with torch.no_grad():
            outputs = model(**inputs)

        attention_weights = outputs.attentions[-1]  # last layer
        cls_attention = attention_weights[0, :, 0, :].mean(dim=0)

        special_tokens = ["[CLS]", "[SEP]", "[PAD]", "[UNK]"]
        token_scores = []

        for i, (token, score) in enumerate(zip(tokens, cls_attention.cpu().numpy())):
            if token not in special_tokens:
                clean_token = token.replace("##", "").lower()
                if clean_token.strip():
                    token_scores.append((clean_token, float(score), i))

        sorted_tokens = sorted(token_scores, key=lambda x: x[1], reverse=True)

        top_tokens = []
        seen = set()
        for token, score, pos in sorted_tokens:
            if token not in seen and len(top_tokens) < k:
                top_tokens.append(token)
                seen.add(token)

        print(f"Debug - Token scores: {[(t, f'{s:.4f}') for t, s, _ in sorted_tokens[:10]]}")
        return normalize_tokens(top_tokens)

    except Exception as e:
        print(f"Attention method failed: {e}")
        return _heuristic_attribution(text, k)

def _heuristic_attribution(text: str, k: int = 5) -> list:
    tokens = text.split()
    scored_tokens = []
    for token in tokens:
        score = len(token)
        if token[0].isupper():
            score += 2
        if token.lower() in ['not', 'no', 'never', 'none']:
            score += 3
        scored_tokens.append((token, score))

    sorted_tokens = sorted(scored_tokens, key=lambda x: x[1], reverse=True)
    top_tokens = [token for token, _ in sorted_tokens[:k]]
    return normalize_tokens(top_tokens)




# TODO: 1.      Enhance generate_explanation() to Parse Token Positions (for deeper causal probing)
#               Right now, your explanations are strings. A more structured output like token indices allows:
#               Replacing tokens with synonyms
#               Position-based dropout for counterfactuals
#
#
#
#
