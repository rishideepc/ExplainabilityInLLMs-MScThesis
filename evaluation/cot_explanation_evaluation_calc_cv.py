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
    Samples CoT propositions by importance and connectivity 
    
    @params: max number of propositions allowed 
    
    @returns: selected propositions for further NLI
    """
    def __init__(self, max_steps: int = 15):
        self.max_steps = max_steps
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
        Scores a proposition’s importance based on content
        
        @params: proposition text, position, total propositions 
        
        @returns: float score
        """
        score = 0.0
        step_lower = step.lower()

        if position == 0:
            score += 2.0
        elif position == total_steps - 1:
            score += 3.0
        elif position < 3:
            score += 1.0
        elif position >= total_steps - 3:
            score += 1.5

        word_count = len(step.split())
        if 10 <= word_count <= 30:
            score += 1.0
        elif word_count < 5:
            score -= 1.0
        elif word_count > 50:
            score -= 0.5

        for category, keywords in self.reasoning_indicators.items():
            for keyword in keywords:
                if keyword in step_lower:
                    if category in ['conclusion', 'causation']:
                        score += 2.0
                    elif category in ['evidence', 'condition']:
                        score += 1.5
                    else:
                        score += 1.0
                    break

        meta_phrases = ['let me think', 'wait', 'hmm', 'actually', 'the claim states that',
                        'let me consider', 'let\'s think', 'okay', 'so']
        for phrase in meta_phrases:
            if phrase in step_lower:
                score -= 2.0
                break

        if '?' in step:
            score += 0.5

        return max(0.0, score)

    def extract_key_entities(self, step: str) -> Set[str]:
        """
        Extracts salient entities from a proposition 
        
        @params: proposition text 
        
        @returns: set of key entities (strings)
        """
        entities = set()
        entities.update(re.findall(r'\b[A-Z][a-z]+\b', step))
        entities.update(re.findall(r'\b\d+(?:\.\d+)?\s*(?:mg|ml|years?|days?|%|hours?|minutes?)\b', step.lower()))
        entities.update(re.findall(r'"([^"]+)"', step))
        entities.update(re.findall(r"'([^']+)'", step))
        entities.update(re.findall(r'\b[a-z]+tion\b', step.lower()))
        entities.update(re.findall(r'\b[a-z]+ism\b', step.lower()))
        return entities

    def calculate_step_connectivity(self, steps: List[str]) -> List[float]:
        """
        Computes entity-overlap connectivity per proposition 
        
        @params: list of propositions 
        
        @returns: list of connectivity scores (float)
        """
        step_entities = [self.extract_key_entities(step) for step in steps]
        connectivity_scores = []

        for i, entities in enumerate(step_entities):
            if not entities:
                connectivity_scores.append(0.0)
                continue
            total_overlap = 0
            for j, other_entities in enumerate(step_entities):
                if i != j:
                    total_overlap += len(entities.intersection(other_entities))
            connectivity_score = total_overlap / len(entities) if entities else 0
            connectivity_scores.append(connectivity_score)

        return connectivity_scores

    def intelligent_sample(self, steps: List[str]) -> List[str]:
        """
        Selects up to max_steps using the combined scores 
        
        @params: proposition list 
        
        @returns: final sampled proposition list for NLI 
        """
        if len(steps) <= self.max_steps:
            return steps

        logger.info(f"Applying intelligent sampling: {len(steps)} → {self.max_steps} steps")

        importance_scores = [self.score_step_importance(s, i, len(steps)) for i, s in enumerate(steps)]
        connectivity_scores = self.calculate_step_connectivity(steps)

        combined_scores = []
        for i in range(len(steps)):
            combined_score = 0.7 * importance_scores[i] + 0.3 * connectivity_scores[i]
            combined_scores.append((combined_score, i, steps[i]))

        combined_scores.sort(key=lambda x: x[0], reverse=True)
        must_include = {0, len(steps) - 1}
        selected_indices = set(must_include)

        for score, idx, step in combined_scores:
            if len(selected_indices) >= self.max_steps:
                break
            selected_indices.add(idx)

        selected_indices = sorted(selected_indices)
        sampled_steps = [steps[i] for i in selected_indices]
        logger.info(f"Selected steps at positions: {selected_indices}")
        return sampled_steps


class OptimizedExplanationEvaluator:
    """
    Evaluates CoT metrics extensively using NLI (with batching and sampling)
    
    @params: model settings 
    
    @returns: metric dictionaries/lists
    """
    def __init__(self, model_name: str = "textattack/bert-base-uncased-MNLI",
                 max_len: int = 512, max_steps: int = 20, batch_size: int = 16):
        self.model_name = model_name
        self.max_len = max_len
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sampler = IntelligentCoTSampler(max_steps=15)
        self.processed_count = 0
        self.total_inference_time = 0
        self.start_time = None

        logger.info(f"Loading model: {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.entailment_idx = 2
        logger.info(f"Using max_steps={max_steps}, batch_size={batch_size}")

    def clean_and_split_explanation(self, explanation: str) -> List[str]:
        """
        Parses CoT into multiple propositions
        
        @params: raw explanation string 
        
        @returns: filtered proposition list
        """
        if not explanation.strip():
            return []
        explanation = re.sub(r'<think>|</think>', '', explanation)
        sentences = re.split(r'[.!?]+\s+', explanation)

        clean_steps = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence.split()) < 5:
                continue
            if any(phrase in sentence.lower() for phrase in [
                "let me think", "wait", "hmm", "okay", "so", "but wait",
                "let's tackle", "the claim is", "the options are"
            ]):
                continue
            clean_steps.append(sentence)
            if len(clean_steps) >= self.max_steps:
                break

        logger.debug(f"Extracted {len(clean_steps)} steps from explanation")
        return clean_steps

    def batch_entailment_optimized(self, pairs: List[Tuple[str, str]],
                                   threshold: float = 0.85) -> List[bool]:
        """
        Batch entailment over premise-hypothesis pairs
        
        @params: list of premise-hypothesis tuples, entailment threshold
         
        @returns: list of boolean entailment flags
        """
        if not pairs:
            return []

        start_time = time.time()
        results = []
        total_batches = len(pairs) // self.batch_size + (1 if len(pairs) % self.batch_size else 0)

        with torch.no_grad():
            for batch_idx in range(0, len(pairs), self.batch_size):
                current_batch = batch_idx // self.batch_size + 1
                if current_batch % 10 == 0:
                    logger.info(f"Processing batch {current_batch}/{total_batches}")

                batch = pairs[batch_idx:batch_idx + self.batch_size]
                premises, hypotheses = zip(*batch)

                try:
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
        Builds directed NLI graph 
        
        @params: list of propositions
        
        @returns: {premise: [entailed]}
        """
        if len(propositions) <= 1:
            return {p: [] for p in propositions}
        if len(propositions) > 15:
            propositions = self.sampler.intelligent_sample(propositions)

        matrix = {p: [] for p in propositions}
        pairs = []

        for i, p_i in enumerate(propositions):
            for j, p_j in enumerate(propositions):
                if i != j:
                    if (abs(i - j) <= 2 or j == len(propositions) - 1 or i == 0):
                        pairs.append((p_i, p_j))

        if not pairs:
            return matrix

        logger.info(f"Building NLI graph with {len(pairs)} pairs for {len(propositions)} propositions")

        try:
            entailment_flags = self.batch_entailment_optimized(pairs)
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
        Computes required CoT metrics 
        
        @params: record with 'cot_explanation'
        
        @returns: metrics dictionary
        """
        explanation = entry.get("cot_explanation", "")
        if not explanation:
            return {"redundancy": 0.0, "weak_relevance": 0.0, "strong_relevance": 0.0, "coherence": -1.0}

        steps = self.clean_and_split_explanation(explanation)
        if len(steps) < 2:
            logger.debug("Too few meaningful steps in explanation")
            return {"redundancy": 0.0, "weak_relevance": 0.0, "strong_relevance": 0.0, "coherence": -1.0}

        propositions = steps
        y_hat = steps[-1]
        matrix = self.build_nli_graph_optimized(propositions)

        metrics = {}
        try:
            metrics["redundancy"] = compute_redundancy(matrix, propositions, y_hat)
            metrics["weak_relevance"] = compute_weak_relevance(matrix, propositions, y_hat)
            metrics["strong_relevance"] = compute_strong_relevance(matrix, propositions, y_hat)
            metrics["coherence"] = -1.0
        except Exception as e:
            logger.error(f"Error computing metrics: {e}")
            metrics = {"redundancy": 0.0, "weak_relevance": 0.0, "strong_relevance": 0.0, "coherence": -1.0}

        return metrics

    def evaluate_all_cot_with_progress(self, filepath: str) -> List[Dict]:
        """
        Evaluates CoT explanations in a single file 
        
        @params: JSONL file path
        
        @returns: list of metric dictionaries
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
            if i % 10 == 0:
                elapsed = time.time() - self.start_time
                avg_time = elapsed / max(1, i)
                eta = avg_time * (len(data) - i)
                logger.info(f"Progress: {i}/{len(data)} ({i/len(data)*100:.1f}%) - "
                            f"Avg: {avg_time:.2f}s/item - ETA: {eta/60:.1f}min")
            try:
                if "claim" not in item:
                    continue
                result = {
                    "q": item["claim"],
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
        """
        Helper function; JSONL file loader
        
        @params: file path 
        
        @returns: list of dictionaries
        """
        try:
            with open(filepath, "r", encoding='utf-8') as f:
                return [json.loads(line.strip()) for line in f.readlines() if line.strip()]
        except Exception as e:
            logger.error(f"Failed to load {filepath}: {e}")
            return []


def evaluate_all_cot(filepath: str) -> List[Dict]:
    """
    Helper function; convenience wrapper 
    
    @params: JSONL file path 
    
    @returns: list of metric dictionaries
    """
    evaluator = OptimizedExplanationEvaluator(
        max_steps=20,
        batch_size=16
    )
    return evaluator.evaluate_all_cot_with_progress(filepath)


def test_single_explanation():
    """Tester for a single CoT explanation"""
    test_entry = {
        "claim": "Test claim",
        "cot_explanation": """<think>
Okay, let's tackle this claim. The patient is a 35-year-old man with itchy, watery eyes and frequent sneezing for the past week. He had a similar episode a year ago around springtime. He has iron deficiency anemia and ankylosing spondylitis, and is on ferrous sulfate, artificial tears, and indomethacin. He's an elementary school teacher. The physical exam shows conjunctival injection and watery discharge, but normal pupils and anterior chamber.

First, I need to figure out what's causing his symptoms. The key points here are the itchy, watery eyes and sneezing, which are classic symptoms of allergic rhinitis. The mention of a similar episode in springtime suggests a seasonal pattern, which is typical of allergic rhinitis, especially if it's related to pollen.
</think>

The patient's symptoms of itchy, watery eyes, frequent sneezing, and a seasonal pattern strongly suggest allergic rhinitis. The treatment would be nasal corticosteroids to reduce nasal inflammation."""
    }

    evaluator = OptimizedExplanationEvaluator(max_steps=10)
    result = evaluator.evaluate_cot_metrics_optimized(test_entry)
    print(f"Test result: {result}")


if __name__ == "__main__":
    test_single_explanation()
