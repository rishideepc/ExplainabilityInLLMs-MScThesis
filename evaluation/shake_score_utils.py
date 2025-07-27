# evaluation/shake_score_utils.py

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).to(DEVICE)

def extract_rationale_tokens_from_output(generation: str) -> list:
    # Expect response like: token1 token2 token3
    match = re.findall(r'\b[\w-]+\b', generation)
    return match

def get_rationale_tokens(prompt: str) -> list:
    formatted_prompt = f"""Q: {prompt}\nAs your answer to this prompt, return only those tokens (words) that influence your decision making.
Don't include any other words in your response.
Provide the response in the form of a space-separated list of tokens, like: token1 token2 token3.\nA:"""

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50)
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        rationale_part = decoded.split("A:")[-1].strip()
        return extract_rationale_tokens_from_output(rationale_part)
