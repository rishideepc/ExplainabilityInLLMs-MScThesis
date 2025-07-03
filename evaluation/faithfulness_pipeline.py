import sys
import os
project_root = os.path.abspath('...')
sys.path.append(project_root)

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from typing import List, Tuple, Dict
from generators.cot_generator import generate_with_ollama
from attribution_utils import get_top_attributed_tokens
import re

class FaithMultiPipeline:
    def __init__(self, model="llama3"):
        self.model = model
    
    def get_prediction_and_confidence(self, text: str):
        """Use the LLM to predict a label and a pseudo-confidence score."""
        prompt = f"""Q: Is the following claim true or false?\nClaim: "{text}"\n
        Return your answer only as 'TRUE' or 'FALSE'
        and also provide your confidence from 0 to 100 in this answer as a number.\nA:"""
        
        try:
            output = generate_with_ollama(prompt, model=self.model)
        except Exception as e:
            print(f"Error generating prediction: {e}")
            return "UNKNOWN", 0.5
        
        # Extract label
        if "TRUE" in output.upper():
            label = "TRUE"
        elif "FALSE" in output.upper():
            label = "FALSE"
        else:
            label = "UNKNOWN"
        
        # Extract confidence
        match = re.search(r"(\d{1,3})", output)
        confidence = float(match.group(1)) / 100 if match else 0.5
        
        return label, confidence
    
    def generate_explanation(self, text: str) -> str:
        """Use the LLM to generate a rationale."""
        prompt = f"""Q: {text}\nAs your answer to this prompt, return only those tokens (words) that influence your decision making.
        Don't include any other words in your response.
        Infact, provide the response in the form of a list of tokens, like: token1 token2 token3.\nA:"""
        
        try:
            return generate_with_ollama(prompt, model=self.model)
        except Exception as e:
            print(f"Error generating explanation: {e}")
            # Fallback: return key words from the text
            return " ".join(text.split()[:5])
    
    def tokenize_explanation(self, explanation: str):
        """Tokenize the LLM-generated explanation string into a list."""
        # Convert to lowercase for consistency
        return [token.lower() for token in explanation.strip().replace(",", "").split()]
    
    def erase_token_and_get_confidence(self, text: str, token: str, expected_label: str):
        """Remove a token from the input, re-predict, and get new confidence."""
        # Create modified text by removing the token
        modified_text = text.replace(token, "").strip()
        
        # Handle case where token removal results in empty text
        if not modified_text:
            return 0.0
        
        try:
            label, conf = self.get_prediction_and_confidence(modified_text)
            return conf if label == expected_label else 1 - conf
        except Exception as e:
            print(f"Error in token erasure for '{token}': {e}")
            return 0.5
    
    def get_suff_confidence(self, explanation_tokens: list, expected_label: str):
        """Evaluate confidence using only the explanation tokens."""
        if not explanation_tokens:
            return 0.0
        
        reduced_input = " ".join(explanation_tokens)
        
        try:
            label, conf = self.get_prediction_and_confidence(reduced_input)
            return conf if label == expected_label else 1 - conf
        except Exception as e:
            print(f"Error in sufficiency evaluation: {e}")
            return 0.5
    
    def get_attribution_tokens(self, text: str, top_k: int = 5) -> list:
        """Returns top_k most attributed tokens using attribution methods."""
        try:
            return get_top_attributed_tokens(text, k=top_k)
        except Exception as e:
            print(f"Error getting attribution tokens: {e}")
            # Fallback: return first k tokens from text
            tokens = text.split()
            return tokens[:top_k] if len(tokens) >= top_k else tokens   