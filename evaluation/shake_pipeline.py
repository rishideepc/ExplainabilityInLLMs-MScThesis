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
        # prompt = (
        #     f"Q: Is the following claim true or false?\n"
        #     f"Claim: \"{text}\"\n\n"
        #     "Return your answer only as 'TRUE' or 'FALSE' and also provide your confidence from 0 to 100.\nA:"
        # )
        # prompt = f'Claim: "{text}"\n\nIs this claim TRUE or FALSE? Provide your answer and confidence (0-100).\n\nAnswer:'
        # inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.model.device)

        # messages = [
        # {"role": "user", "content": f'Is this claim TRUE or FALSE? "{text}"\n\nRespond with only TRUE or FALSE and your confidence 0-100.'}
        # ]

        # messages = [
        # {"role": "user", "content": f'Evaluate this claim: "{text}"\n\nIs it TRUE or FALSE? Give your answer and confidence (0-100).'}
        # ]

        messages = [
            {"role": "system", "content": "You are a factual claim evaluator. Respond with TRUE or FALSE and your confidence level."},
            {"role": "user", "content": f'Claim: "{text}"\n\nAnswer: TRUE or FALSE (confidence 0-100)'}
        ]
        
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.model.device)
        
        for key in inputs:
            if inputs[key].dtype == torch.float:
                inputs[key] = inputs[key].to(dtype=torch.float16)

        # with torch.no_grad():
        #     outputs = self.model.generate(**inputs, max_new_tokens=10)

        # output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        # print(f"Base model output: {output_text}")

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=15, do_sample=False, temperature=1.0, pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.eos_token_id)

        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        print(f"Full output: {output_text}")

        # Get only the newly generated tokens (not the input prompt)
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
        print(f"Generated tokens only: {output_text}")

        label = "TRUE" if "TRUE" in output_text.upper() else "FALSE" if "FALSE" in output_text.upper() else "UNKNOWN"
        match = re.search(r"[Cc]onfidence\s*[:=]?\s*(\d{1,3})", output_text)
        confidence = float(match.group(1)) / 100 if match else 0.5

        print(f"Base label: {label} | confidence: {confidence:.2f}")
        return label, confidence

    def generate_rationale_tokens(self, text: str) -> List[str]:
        messages = [
            {"role": "system", "content": "You are a keyword extractor. Respond only with the requested words, nothing else."},
            {"role": "user", "content": f'Claim: "{text}"\n\nExtract 1-4 key words from this claim. Respond with only the words separated by spaces, no explanations.'}
        ]
        
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.model.device)
        
        for key in inputs:
            if inputs[key].dtype == torch.float:
                inputs[key] = inputs[key].to(dtype=torch.float16)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=30, do_sample=False, temperature=1.0)

        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        print(f"Raw rationale output: {output_text}")
    
        # Find the part after the colon or after "words"
        if ':' in output_text:
            keywords_part = output_text.split(':')[-1].strip()
        else:
            # Look for patterns like "Illuminati card game popular"
            words = output_text.split()
            # Skip conversational words and take the substantive part
            skip_words = {'sure', 'here', 'are', 'the', 'key', 'words', 'extracted', 'from', 'claim'}
            keywords_part = ' '.join([w for w in words if w.lower() not in skip_words])
        
        # Extract claim words from this part
        claim_words = set(re.findall(r'\b[a-zA-Z]{2,}\b', text.lower()))
        rationale_words = []
        
        for word in re.findall(r'\b[a-zA-Z]{2,}\b', keywords_part.lower()):
            if word in claim_words and word not in rationale_words:
                rationale_words.append(word)
        
        print(f"Extracted rationale tokens: {rationale_words[:4]}")
        return rationale_words[:4]