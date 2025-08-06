# evaluation/explanation_evaluation_calc_qa.py

import json
from typing import List, Dict, Optional, Tuple
import torch
import sys
import os
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Set up logging with progress info
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

project_root = os.path.abspath('..')
sys.path.append(project_root)

from evaluation.Evaluating_Explanations.src.metrics.argumentative_metrics import (
    compute_circularity, compute_dialectical_acceptability
)
from evaluation.Evaluating_Explanations.src.metrics.deductive_metrics import (
    compute_redundancy, compute_strong_relevance, compute_weak_relevance
)

class OptimizedExplanationEvaluator:
    def __init__(self, model_name: str = "textattack/bert-base-uncased-MNLI", 
                 max_len: int = 512, max_steps: int = 20, batch_size: int = 16):
        self.model_name = model_name
        self.max_len = max_len
        self.max_steps = max_steps  # Limit number of steps to process
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Performance tracking
        self.processed_count = 0
        self.total_inference_time = 0
        self.start_time = None
        
        logger.info(f"Loading model: {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # Model-specific entailment index
        self.entailment_idx = 2  # Standard for BERT/RoBERTa MNLI models
        logger.info(f"Using max_steps={max_steps}, batch_size={batch_size}")

    def clean_and_split_explanation(self, explanation: str) -> List[str]:
        """
        Intelligently parse and clean CoT explanation into meaningful steps.
        """
        if not explanation.strip():
            return []
        
        # Remove <think> tags if present
        explanation = re.sub(r'<think>|</think>', '', explanation)
        
        # Split by sentences, but be smarter about it
        # Look for sentence boundaries, but also logical breaks
        sentences = re.split(r'[.!?]+\s+', explanation)
        
        # Clean and filter sentences
        clean_steps = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Skip very short fragments (likely incomplete)
            if len(sentence.split()) < 5:
                continue
                
            # Skip meta-commentary (common in CoT)
            if any(phrase in sentence.lower() for phrase in [
                "let me think", "wait", "hmm", "okay", "so", "but wait",
                "let's tackle", "the question is", "the options are"
            ]):
                continue
                
            clean_steps.append(sentence)
            
            # Limit total steps for performance
            if len(clean_steps) >= self.max_steps:
                break
        
        logger.debug(f"Extracted {len(clean_steps)} steps from explanation")
        return clean_steps

    def batch_entailment_optimized(self, pairs: List[Tuple[str, str]], 
                                 threshold: float = 0.85) -> List[bool]:
        """
        Highly optimized batch entailment with progress tracking.
        """
        if not pairs:
            return []
        
        start_time = time.time()
        results = []
        total_batches = len(pairs) // self.batch_size + (1 if len(pairs) % self.batch_size else 0)
        
        with torch.no_grad():
            for batch_idx in range(0, len(pairs), self.batch_size):
                current_batch = batch_idx // self.batch_size + 1
                if current_batch % 10 == 0:  # Progress every 10 batches
                    logger.info(f"Processing batch {current_batch}/{total_batches}")
                
                batch = pairs[batch_idx:batch_idx + self.batch_size]
                premises, hypotheses = zip(*batch)

                try:
                    # Optimized tokenization
                    encodings = self.tokenizer(
                        list(premises),
                        list(hypotheses),
                        padding=True,
                        truncation=True,
                        max_length=self.max_len,
                        return_tensors='pt'
                    )

                    input_ids = encodings['input_ids'].to(self.device)
                    attention_mask = encodings['attention_mask'].to(self.device)

                    # Forward pass
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    probs = torch.softmax(outputs.logits, dim=1)
                    entail_probs = probs[:, self.entailment_idx]

                    batch_results = [prob.item() >= threshold for prob in entail_probs]
                    results.extend(batch_results)
                    
                except Exception as e:
                    logger.warning(f"Error in batch {current_batch}: {e}")
                    results.extend([False] * len(batch))

        elapsed_time = time.time() - start_time
        self.total_inference_time += elapsed_time
        logger.info(f"Processed {len(pairs)} entailment pairs in {elapsed_time:.2f}s")
        
        return results

    def build_nli_graph_optimized(self, propositions: List[str]) -> Dict[str, List[str]]:
        """
        Build NLI graph with early stopping and sampling for very long explanations.
        """
        if len(propositions) <= 1:
            return {p: [] for p in propositions}
        
        # For very long explanations, use sampling
        if len(propositions) > 15:
            logger.info(f"Large explanation ({len(propositions)} steps), using strategic sampling")
            # Keep first few, last few, and sample from middle
            sampled = (propositions[:5] + 
                      propositions[-5:] + 
                      propositions[5:-5][::2])  # Every other from middle
            propositions = list(dict.fromkeys(sampled))  # Remove duplicates, preserve order
            logger.info(f"Sampled down to {len(propositions)} steps")
        
        matrix = {p: [] for p in propositions}
        pairs = []
        
        # Build pairs with focus on adjacent and key relationships
        for i, p_i in enumerate(propositions):
            for j, p_j in enumerate(propositions):
                if i != j:
                    # Prioritize adjacent steps and connections to conclusion
                    if (abs(i - j) <= 2 or  # Adjacent steps
                        j == len(propositions) - 1 or  # Connection to conclusion
                        i == 0):  # Connection from premise
                        pairs.append((p_i, p_j))

        if not pairs:
            return matrix
            
        logger.info(f"Building NLI graph with {len(pairs)} pairs for {len(propositions)} propositions")
        
        try:
            entailment_flags = self.batch_entailment_optimized(pairs)
            
            # Rebuild matrix from results
            pair_idx = 0
            for i, p_i in enumerate(propositions):
                for j, p_j in enumerate(propositions):
                    if i != j and (abs(i - j) <= 2 or j == len(propositions) - 1 or i == 0):
                        if pair_idx < len(entailment_flags) and entailment_flags[pair_idx]:
                            matrix[p_i].append(p_j)
                        pair_idx += 1
                        
        except Exception as e:
            logger.error(f"Failed to build NLI graph: {e}")
            
        return matrix

    def evaluate_cot_metrics_optimized(self, entry: Dict) -> Dict[str, float]:
        """
        Optimized CoT evaluation with progress tracking.
        """
        explanation = entry.get("cot_explanation", "")
        if not explanation:
            return {"redundancy": 0.0, "weak_relevance": 0.0, "strong_relevance": 0.0, "coherence": -1.0}

        # Smart parsing of explanation
        steps = self.clean_and_split_explanation(explanation)
        
        if len(steps) < 2:
            logger.debug("Too few meaningful steps in explanation")
            return {"redundancy": 0.0, "weak_relevance": 0.0, "strong_relevance": 0.0, "coherence": -1.0}

        logger.debug(f"Processing explanation with {len(steps)} steps")
        
        propositions = steps
        y_hat = steps[-1]  # Conclusion is typically the last step

        # Build NLI graph with optimizations
        matrix = self.build_nli_graph_optimized(propositions)

        # Compute metrics
        metrics = {}
        try:
            metrics["redundancy"] = compute_redundancy(matrix, propositions, y_hat)
            metrics["weak_relevance"] = compute_weak_relevance(matrix, propositions, y_hat)
            metrics["strong_relevance"] = compute_strong_relevance(matrix, propositions, y_hat)
            metrics["coherence"] = -1.0  # Skip coherence for now due to complexity
            
        except Exception as e:
            logger.error(f"Error computing metrics: {e}")
            metrics = {"redundancy": 0.0, "weak_relevance": 0.0, "strong_relevance": 0.0, "coherence": -1.0}

        return metrics

    def evaluate_all_cot_with_progress(self, filepath: str) -> List[Dict]:
        """
        Evaluate all CoT explanations with detailed progress tracking.
        """
        logger.info(f"Loading data from: {filepath}")
        
        try:
            with open(filepath, "r", encoding='utf-8') as f:
                data = [json.loads(line.strip()) for line in f.readlines() if line.strip()]
        except Exception as e:
            logger.error(f"Failed to load {filepath}: {e}")
            return []
            
        if not data:
            logger.warning("No data loaded")
            return []
            
        logger.info(f"Processing {len(data)} entries...")
        self.start_time = time.time()
        
        results = []
        for i, item in enumerate(data):
            if i % 10 == 0:  # Progress every 10 items
                elapsed = time.time() - self.start_time
                avg_time = elapsed / max(1, i)
                eta = avg_time * (len(data) - i)
                logger.info(f"Progress: {i}/{len(data)} ({i/len(data)*100:.1f}%) - "
                          f"Avg: {avg_time:.2f}s/item - ETA: {eta/60:.1f}min")
            
            try:
                if "question" not in item:
                    continue
                    
                result = {
                    "q": item["question"],
                    **self.evaluate_cot_metrics_optimized(item)
                }
                results.append(result)
                self.processed_count += 1
                
            except Exception as e:
                logger.warning(f"Skipped entry {i}: {e}")
                continue
        
        total_time = time.time() - self.start_time
        logger.info(f"Completed! Processed {len(results)}/{len(data)} entries in {total_time/60:.2f}min")
        logger.info(f"Average time per entry: {total_time/len(results):.2f}s")
        logger.info(f"Total inference time: {self.total_inference_time/60:.2f}min")
        
        return results

    def load_jsonl(self, filepath: str) -> List[Dict]:
        """Load JSONL file."""
        try:
            with open(filepath, "r", encoding='utf-8') as f:
                return [json.loads(line.strip()) for line in f.readlines() if line.strip()]
        except Exception as e:
            logger.error(f"Failed to load {filepath}: {e}")
            return []

# Convenience functions for backward compatibility
def evaluate_all_cot(filepath: str) -> List[Dict]:
    """Optimized evaluation with progress tracking."""
    evaluator = OptimizedExplanationEvaluator(
        max_steps=20,  # Limit steps for performance
        batch_size=16  # Larger batch size for efficiency
    )
    return evaluator.evaluate_all_cot_with_progress(filepath)

def evaluate_all_argllm(filepath: str) -> List[Dict]:
    """ArgLLM evaluation (unchanged)."""
    evaluator = OptimizedExplanationEvaluator()
    return evaluator.evaluate_all_argllm(filepath)

# Quick test function
def test_single_explanation():
    """Test with the provided long explanation."""
    test_entry = {
        "question": "Test question",
        "cot_explanation": """<think>
Okay, let's tackle this question. The patient is a 35-year-old man with itchy, watery eyes and frequent sneezing for the past week. He had a similar episode a year ago around springtime. He has iron deficiency anemia and ankylosing spondylitis, and is on ferrous sulfate, artificial tears, and indomethacin. He's an elementary school teacher. The physical exam shows conjunctival injection and watery discharge, but normal pupils and anterior chamber.

First, I need to figure out what's causing his symptoms. The key points here are the itchy, watery eyes and sneezing, which are classic symptoms of allergic rhinitis. The mention of a similar episode in springtime suggests a seasonal pattern, which is typical of allergic rhinitis, especially if it's related to pollen.
</think>

The patient's symptoms of itchy, watery eyes, frequent sneezing, and a seasonal pattern strongly suggest allergic rhinitis. The treatment would be nasal corticosteroids to reduce nasal inflammation."""
    }
    
    evaluator = OptimizedExplanationEvaluator(max_steps=10)
    result = evaluator.evaluate_cot_metrics_optimized(test_entry)
    print(f"Test result: {result}")

if __name__ == "__main__":
    test_single_explanation()