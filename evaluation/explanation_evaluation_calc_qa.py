# evaluation/explanation_evaluation_calc_qa.py

import json
from typing import List, Dict, Optional, Tuple, Set
import torch
import sys
import os
import logging
import re
import time
from collections import Counter
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

class IntelligentCoTSampler:
    """
    Intelligent sampling for Chain-of-Thought explanations based on content analysis.
    """
    def __init__(self, max_steps: int = 15):
        self.max_steps = max_steps
        
        # General reasoning indicators (no domain-specific terms)
        self.reasoning_indicators = {
            'conclusion': ['therefore', 'thus', 'so', 'hence', 'consequently', 'conclude'],
            'causation': ['because', 'since', 'due to', 'caused by', 'leads to', 'results in'],
            'contrast': ['however', 'but', 'although', 'despite', 'on the other hand'],
            'evidence': ['shows', 'indicates', 'suggests', 'demonstrates', 'proves'],
            'condition': ['if', 'unless', 'provided that', 'given that'],
            'sequence': ['first', 'next', 'then', 'finally', 'subsequently']
        }

    def score_step_importance(self, step: str, position: int, total_steps: int) -> float:
        """
        Score how important a step is based on multiple factors.
        Higher score = more important.
        """
        score = 0.0
        step_lower = step.lower()
        
        # Position-based scoring
        if position == 0:  # First step
            score += 2.0
        elif position == total_steps - 1:  # Last step (conclusion)
            score += 3.0
        elif position < 3:  # Early steps
            score += 1.0
        elif position >= total_steps - 3:  # Late steps
            score += 1.5
        
        # Length-based scoring (very short or very long steps might be less important)
        word_count = len(step.split())
        if 10 <= word_count <= 30:  # Sweet spot for reasoning steps
            score += 1.0
        elif word_count < 5:  # Too short, likely fragment
            score -= 1.0
        elif word_count > 50:  # Too long, likely verbose
            score -= 0.5
        
        # Reasoning indicator scoring
        for category, keywords in self.reasoning_indicators.items():
            for keyword in keywords:
                if keyword in step_lower:
                    if category in ['conclusion', 'causation']:
                        score += 2.0
                    elif category in ['evidence', 'condition']:
                        score += 1.5
                    else:
                        score += 1.0
                    break  # Don't double-count within same category
        
        # Penalty for meta-commentary
        meta_phrases = ['let me think', 'wait', 'hmm', 'actually', 'the question is asking',
                       'let me consider', 'let\'s think', 'okay', 'so']
        for phrase in meta_phrases:
            if phrase in step_lower:
                score -= 2.0
                break
        
        # Question/interrogative bonus (often important reasoning)
        if '?' in step:
            score += 0.5
        
        return max(0.0, score)  # Don't go negative

    def extract_key_entities(self, step: str) -> Set[str]:
        """Extract key entities/concepts from a step for connectivity analysis."""
        # Simple entity extraction
        entities = set()
        
        # Extract capitalized words (potential proper nouns)
        entities.update(re.findall(r'\b[A-Z][a-z]+\b', step))
        
        # Extract numbers and measurements
        entities.update(re.findall(r'\b\d+(?:\.\d+)?\s*(?:mg|ml|years?|days?|%|hours?|minutes?)\b', step.lower()))
        
        # Extract quoted terms
        entities.update(re.findall(r'"([^"]+)"', step))
        entities.update(re.findall(r"'([^']+)'", step))
        
        # Extract important technical terms (general patterns)
        entities.update(re.findall(r'\b[a-z]+tion\b', step.lower()))  # Words ending in -tion
        entities.update(re.findall(r'\b[a-z]+ism\b', step.lower()))   # Words ending in -ism
        
        return entities

    def calculate_step_connectivity(self, steps: List[str]) -> List[float]:
        """Calculate how well-connected each step is to others (entity overlap)."""
        step_entities = [self.extract_key_entities(step) for step in steps]
        connectivity_scores = []
        
        for i, entities in enumerate(step_entities):
            if not entities:
                connectivity_scores.append(0.0)
                continue
                
            # Calculate overlap with other steps
            total_overlap = 0
            for j, other_entities in enumerate(step_entities):
                if i != j:
                    overlap = len(entities.intersection(other_entities))
                    total_overlap += overlap
            
            # Normalize by step's entity count
            connectivity_score = total_overlap / len(entities) if entities else 0
            connectivity_scores.append(connectivity_score)
        
        return connectivity_scores

    def intelligent_sample(self, steps: List[str]) -> List[str]:
        """
        Intelligently sample steps based on importance and connectivity.
        """
        if len(steps) <= self.max_steps:
            return steps
        
        logger.info(f"Applying intelligent sampling: {len(steps)} → {self.max_steps} steps")
        
        # Score each step for importance
        importance_scores = []
        for i, step in enumerate(steps):
            score = self.score_step_importance(step, i, len(steps))
            importance_scores.append(score)
        
        # Calculate connectivity scores
        connectivity_scores = self.calculate_step_connectivity(steps)
        
        # Combine scores (weighted average)
        combined_scores = []
        for i in range(len(steps)):
            combined_score = (
                0.7 * importance_scores[i] +  # Importance matters more
                0.3 * connectivity_scores[i]   # But connectivity helps too
            )
            combined_scores.append((combined_score, i, steps[i]))
        
        # Sort by combined score (descending)
        combined_scores.sort(key=lambda x: x[0], reverse=True)
        
        # Always include first and last steps
        must_include = {0, len(steps) - 1}
        
        # Select top-scoring steps, ensuring we include must-have steps
        selected_indices = set()
        selected_indices.update(must_include)
        
        # Add highest-scoring steps until we reach max_steps
        for score, idx, step in combined_scores:
            if len(selected_indices) >= self.max_steps:
                break
            selected_indices.add(idx)
        
        # Sort selected indices to maintain chronological order
        selected_indices = sorted(selected_indices)
        
        # Extract selected steps
        sampled_steps = [steps[i] for i in selected_indices]
        
        logger.info(f"Selected steps at positions: {selected_indices}")
        
        return sampled_steps


class OptimizedExplanationEvaluator:
    def __init__(self, model_name: str = "textattack/bert-base-uncased-MNLI", 
                 max_len: int = 512, max_steps: int = 20, batch_size: int = 16):
        self.model_name = model_name
        self.max_len = max_len
        self.max_steps = max_steps  # Limit number of steps to process
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize intelligent sampler
        self.sampler = IntelligentCoTSampler(max_steps=15)  # Use intelligent sampling at 15 steps
        
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
        Build NLI graph with intelligent sampling for long explanations.
        """
        if len(propositions) <= 1:
            return {p: [] for p in propositions}
        
        # Apply intelligent sampling for long explanations
        if len(propositions) > 15:
            propositions = self.sampler.intelligent_sample(propositions)
        
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

    def evaluate_argllm_metrics(self, entry: Dict) -> Dict[str, float]:
        """Evaluate ArgLLM-specific metrics."""
        base_bag = entry.get("base", {}).get("bag", {})
        args = list(base_bag.get("arguments", {}).keys())
        attacks = base_bag.get("attacks", [])
        supports = base_bag.get("supports", [])

        y_hat = ["db0"] if "db0" in args else []

        try:
            return {
                "circularity": compute_circularity(args, attacks, supports),
                "acceptability": compute_dialectical_acceptability(args, attacks, y_hat)
            }
        except Exception as e:
            logger.error(f"Error computing ArgLLM metrics: {e}")
            return {"circularity": 0.0, "acceptability": 0.0}

    def evaluate_all_argllm(self, filepath: str) -> List[Dict]:
        """Evaluate all ArgLLM explanations in a file."""
        data = self.load_jsonl(filepath)
        if not data:
            return []
            
        results = []
        for i, item in enumerate(data):
            try:
                result = {
                    "q": item.get("question", f"Question_{i}"),
                    **self.evaluate_argllm_metrics(item)
                }
                results.append(result)
            except Exception as e:
                logger.warning(f"Skipped ArgLLM entry {i} due to error: {e}")
                continue
                
        logger.info(f"Successfully evaluated {len(results)}/{len(data)} ArgLLM entries")
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