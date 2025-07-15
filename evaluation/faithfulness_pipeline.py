# evaluation/faithfulness_pipeline.py

import re
from typing import List, Tuple
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"

class FaithMultiPipeline:
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",              # auto-map to available GPU
            torch_dtype=torch.float16       # reduced precision for speed and fit
        )
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model.eval()

        print(f"🧠 Model loaded with dtype: {self.model.dtype}")
        print(f"📍 Device map: {self.model.hf_device_map}")

    def get_prediction_and_confidence(self, text: str) -> Tuple[str, float]:
        prompt = (
            f"Q: Is the following claim true or false?\nClaim: \"{text}\"\n\n"
            "Return your answer only as 'TRUE' or 'FALSE' and also provide your confidence from 0 to 100.\nA:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=32)
        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "TRUE" in output_text.upper():
            label = "TRUE"
        elif "FALSE" in output_text.upper():
            label = "FALSE"
        else:
            label = "UNKNOWN"
        
        match = re.search(r"(\d{1,3})", output_text)
        confidence = float(match.group(1)) / 100 if match else 0.5
        return label, confidence

    def generate_explanation(self, text: str) -> str:
        prompt = (
            f"Q: {text}\n"
            f"A: Let’s explain step by step. As you explain your reasoning, *surround the most important decision-making tokens with asterisks (*).* "
            f"\nHere is an example format:\n"
            f"Example: The *sky* appears *blue* because of *Rayleigh scattering*.\n"
            f"Now answer:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=150)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def extract_rationale_tokens(self, explanation: str) -> List[str]:
        return re.findall(r"\*(.*?)\*", explanation)

    def tokenize_explanation(self, explanation: str) -> List[str]:
        return [token.lower() for token in explanation.strip().split()]

    def erase_token_and_get_confidence(self, text: str, token: str, expected_label: str) -> float:
        modified = text.replace(token, "").strip()
        if not modified:
            return 0.0
        label, conf = self.get_prediction_and_confidence(modified)
        return conf if label == expected_label else 1 - conf

    def get_suff_confidence(self, explanation_tokens: List[str], expected_label: str) -> float:
        if not explanation_tokens:
            return 0.0
        reduced_input = " ".join(explanation_tokens)
        label, conf = self.get_prediction_and_confidence(reduced_input)
        return conf if label == expected_label else 1 - conf

    def get_attribution_tokens(self, text: str, top_k: int = 5) -> List[str]:
        tokens = text.split()
        return tokens[:top_k] if len(tokens) >= top_k else tokens
