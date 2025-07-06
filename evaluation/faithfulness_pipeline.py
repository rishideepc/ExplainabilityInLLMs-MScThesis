import sys
import os
import re
from typing import List, Tuple, Dict

project_root = os.path.abspath('...')
sys.path.append(project_root)

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from generators.cot_generator import generate_with_ollama
from evaluation.attribution_extraction import get_top_attributed_tokens         

class FaithMultiPipeline:
    def __init__(self, model="llama3"):
        self.model = model
    
    def get_prediction_and_confidence(self, text: str) -> Tuple[str, float]:
        """
        Prompt LLM to classify a claim as TRUE or FALSE with confidence.
        """
        prompt = f"""Q: Is the following claim true or false?\nClaim: "{text}"\n
        Return your answer only as 'TRUE' or 'FALSE'
        and also provide your confidence from 0 to 100 in this answer as a number.\nA:"""
        
        try:
            output = generate_with_ollama(prompt, model=self.model)
        except Exception as e:
            print(f"Error generating prediction: {e}")
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
        """
        Prompt LLM to generate rationale as explanation tokens only.
        """
        prompt = f"""Q: {text}\nAs your answer to this prompt, return only those tokens (words) that influence your decision making.
        Don't include any other words in your response.
        Provide the response as a space-separated list of tokens: token1 token2 token3\nA:"""
        
        try:
            return generate_with_ollama(prompt, model=self.model)
        except Exception as e:
            print(f"Error generating explanation: {e}")
            return " ".join(text.split()[:5])  # fallback

    def tokenize_explanation(self, explanation: str) -> List[str]:
        """
        Tokenize the explanation into lowercase, cleaned tokens.
        """
        return [token.lower() for token in explanation.strip().replace(",", "").split()]

    def erase_token_and_get_confidence(self, text: str, token: str, expected_label: str) -> float:
        """
        Remove a token from the input and re-check model prediction confidence.
        """
        modified_text = text.replace(token, "").strip()
        if not modified_text:
            return 0.0
        
        try:
            label, conf = self.get_prediction_and_confidence(modified_text)
            return conf if label == expected_label else 1 - conf
        except Exception as e:
            print(f"Error in token erasure for '{token}': {e}")
            return 0.5

    def get_suff_confidence(self, explanation_tokens: List[str], expected_label: str) -> float:
        """
        Compute sufficiency by feeding only explanation tokens and observing model confidence.
        """
        if not explanation_tokens:
            return 0.0
        
        reduced_input = " ".join(explanation_tokens)
        
        try:
            label, conf = self.get_prediction_and_confidence(reduced_input)
            return conf if label == expected_label else 1 - conf
        except Exception as e:
            print(f"Error in sufficiency evaluation: {e}")
            return 0.5

    def get_attribution_tokens(self, text: str, top_k: int = 5) -> List[str]:
        """
        Extract top-k attribution tokens using custom attribution_extraction module.
        """
        try:
            return get_top_attributed_tokens(text, k=top_k)
        except Exception as e:
            print(f"Error getting attribution tokens: {e}")
            tokens = text.split()
            return tokens[:top_k] if len(tokens) >= top_k else tokens
