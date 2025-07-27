# evaluation/shake_pipeline.py

import re
from typing import Tuple, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"

class ShakePipeline:
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            token=os.environ["HF_TOKEN"],
            device_map="auto",
            torch_dtype=torch.float16
        ).to("cuda:0").eval()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def get_label_and_confidence(self, text: str) -> Tuple[str, float]:
        prompt = (
            f"Q: Is the following claim true or false?\n"
            f"Claim: \"{text}\"\n\n"
            "Return your answer only as 'TRUE' or 'FALSE' and also provide your confidence from 0 to 100.\nA:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.model.device)
        for key in inputs:
            if inputs[key].dtype == torch.float:
                inputs[key] = inputs[key].to(dtype=torch.float16)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=10)

        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"🧠 Base model output: {output_text}")

        label = "TRUE" if "TRUE" in output_text.upper() else "FALSE" if "FALSE" in output_text.upper() else "UNKNOWN"
        # match = re.search(r"(\d{1,3})", output_text)

        match = re.search(r"[Cc]onfidence\s*[:=]?\s*(\d{1,3})", output_text)
        print(f"🧪 Confidence match: {match.group(1) if match else 'None'}")
        confidence = float(match.group(1)) / 100 if match else 0.5

        print(f"📊 Base label: {label} | confidence: {confidence:.2f}")
        return label, confidence

    def generate_rationale_tokens(self, text: str) -> List[str]:
        prompt = (
            f"Claim: {text}\n"
            "List 1 to 4 **exact words** from the above claim that most influenced your decision.\n"
            "Respond with a space-separated list like: woman kill alley stranger\n"
            "Do not include explanations, punctuation, or numbering.\n"
            "Answer:"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.model.device)
        for key in inputs:
            if inputs[key].dtype == torch.float:
                inputs[key] = inputs[key].to(dtype=torch.float16)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=20)

        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        raw_line = output_text.strip().split("\n")[-1]

        print(f"🧠 Raw rationale string: {raw_line}")
        rationale_words = re.findall(r'\b[a-zA-Z]{3,}\b', raw_line)
        print(f"🎯 Extracted rationale tokens: {rationale_words[:4]}")
        return rationale_words[:4]
