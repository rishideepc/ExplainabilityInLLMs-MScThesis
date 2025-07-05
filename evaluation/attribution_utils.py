import re
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

def normalize_tokens(tokens):
    return [
        re.sub(r'\W+', '', token.replace("##", "").lower())
        for token in tokens if token.strip()
    ]

def get_top_attributed_tokens_with_positions(text: str, k: int = 5):
    try:
        from inseq import load_model

        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

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

        scored = [(i, token, score) for i, (token, score) in enumerate(zip(tokens, scores))]
        sorted_scored = sorted(scored, key=lambda x: abs(x[2]), reverse=True)

        top_k_positions = [
            (i, token) for i, token, _ in sorted_scored[:k]
            if token not in ["[CLS]", "[SEP]", "[PAD]", "[UNK]", tokenizer.pad_token, tokenizer.cls_token, tokenizer.sep_token]
        ]

        return top_k_positions

    except Exception as e:
        print(f"Inseq attribution failed: {e}")
        return []
