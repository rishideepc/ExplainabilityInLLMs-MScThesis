import os
import re
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
# MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
# MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"


class ShakePipeline:
    """Minimal wrapper around an LLM for label prediction and keyword extraction"""

    def __init__(self):
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                token=os.environ["HF_TOKEN"],  # keep strict env requirement
                device_map="auto",
                torch_dtype=torch.float16,
                attn_implementation="eager",
            ).eval()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.device = next(self.model.parameters()).device

    def get_label(self, text: str) -> str:
        """
        Extracts a binary label for a factual claim via greedy decoding

        @params: input claim string 
        
        @returns: "TRUE" | "FALSE" | "UNKNOWN" labels
        """
        messages = [
            {"role": "system", "content": "You are a factual claim evaluator. Respond with TRUE or FALSE."},
            {"role": "user", "content": f'Claim: "{text}"\n\nAnswer: TRUE or FALSE'},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        for k in inputs:
            if inputs[k].dtype == torch.float:
                inputs[k] = inputs[k].to(dtype=torch.float16)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=15,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        gen = outputs[0][inputs["input_ids"].shape[1]:]
        text_out = self.tokenizer.decode(gen, skip_special_tokens=True).strip()
        print(f"Generated tokens only: {text_out}")

        u = text_out.upper()
        label = "TRUE" if "TRUE" in u else ("FALSE" if "FALSE" in u else "UNKNOWN")
        print(f"Base label: {label}")
        return label

    def generate_rationale_tokens(self, text: str) -> List[str]:
        """
        Extracts up to four keywords from the claim

        @params: input claim string 
        
        @returns: list[str] of upto 4 keywords
        """
        messages = [
            {
                "role": "system",
                "content": "You are a keyword extractor. Respond only with the requested words, nothing else.",
            },
            {
                "role": "user",
                "content": (
                    f'Claim: "{text}"\n\n'
                    "Extract 1-4 key words from this claim. Respond with only the words separated by spaces, no explanations."
                ),
            },
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        for k in inputs:
            if inputs[k].dtype == torch.float:
                inputs[k] = inputs[k].to(dtype=torch.float16)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=30, do_sample=False, temperature=1.0)

        gen = outputs[0][inputs["input_ids"].shape[1]:]
        text_out = self.tokenizer.decode(gen, skip_special_tokens=True).strip()
        print(f"Raw rationale output: {text_out}")

        # Prefer the segment after a colon, else prune boilerplate tokens.
        if ":" in text_out:
            kw_part = text_out.split(":")[-1].strip()
        else:
            words = text_out.split()
            skip = {"sure", "here", "are", "the", "key", "words", "extracted", "from", "claim"}
            kw_part = " ".join(w for w in words if w.lower() not in skip)

        # Keep only words that actually appear in the claim, dedup, cap at 4.
        claim_words = set(re.findall(r"\b[a-zA-Z]{2,}\b", text.lower()))
        picked: List[str] = []
        for w in re.findall(r"\b[a-zA-Z]{2,}\b", kw_part.lower()):
            if w in claim_words and w not in picked:
                picked.append(w)
            if len(picked) >= 4:
                break

        print(f"Extracted rationale tokens: {picked}")
        return picked
