import re
from attribution_utils import get_top_attributed_tokens_with_positions
from generators.cot_generator import generate_with_ollama

class FaithMultiPipeline:
    def __init__(self, model="llama3"):
        self.model = model

    def get_prediction_and_confidence(self, text: str):
        prompt = f"""Q: Is the following claim true or false?\nClaim: "{text}"\n
        Return your answer only as 'TRUE' or 'FALSE'
        and also provide your confidence from 0 to 100 in this answer as a number.\nA:"""
        
        try:
            output = generate_with_ollama(prompt, model=self.model)
        except:
            return "UNKNOWN", 0.5

        if "TRUE" in output.upper():
            label = "TRUE"
        elif "FALSE" in output.upper():
            label = "FALSE"
        else:
            label = "UNKNOWN"

        match = re.search(r"(\d{1,3})", output)
        confidence = float(match.group(1)) / 100 if match else 0.5
        return label, confidence

    def generate_explanation(self, text: str) -> str:
        prompt = f"""Q: {text}\nAs your answer to this prompt, return only those tokens (words) that influence your decision making.
        Don't include any other words in your response.
        Return only a space-separated list of tokens.\nA:"""
        try:
            return generate_with_ollama(prompt, model=self.model)
        except:
            return " ".join(text.split()[:5])

    def tokenize_explanation(self, explanation: str):
        return [token.lower() for token in explanation.strip().replace(",", "").split()]

    def erase_token_and_get_confidence(self, text: str, token: str, expected_label: str):
        modified = text.replace(token, "").strip()
        if not modified:
            return 0.0
        try:
            label, conf = self.get_prediction_and_confidence(modified)
            return conf if label == expected_label else 1 - conf
        except:
            return 0.5

    def get_suff_confidence(self, explanation_tokens: list, expected_label: str):
        if not explanation_tokens:
            return 0.0
        reduced = " ".join(explanation_tokens)
        try:
            label, conf = self.get_prediction_and_confidence(reduced)
            return conf if label == expected_label else 1 - conf
        except:
            return 0.5

    def get_attribution_tokens(self, text: str, top_k: int = 5) -> list:
        try:
            pos_tokens = get_top_attributed_tokens_with_positions(text, k=top_k)
            return [tok for _, tok in pos_tokens]
        except:
            return text.split()[:top_k]
