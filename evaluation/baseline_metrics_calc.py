import json
import re
from typing import List, Dict
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx

# Load jsonl data files from \results\generation\                       
# TODO: move files to \data\ for maintainability
def load_jsonl(filepath: str) -> List[Dict]:
    with open(filepath, "r") as f:
        return [json.loads(line.strip()) for line in f.readlines()]

###### CoT prompting metrics #########################

# Implement baseline coherence metric calculation
def baseline_coherence_score(explanation: str) -> float:
    """
    Computes coherence score for the explanation. Measures the pair-wise similarity 
    between individual subsets/steps.

    @params:
        explanation (str): a multi-line explanation string; each line is treated as an individual step

    @returns:
        float: Coherence score between 0 and 1
        Score towards 1 --> better coherence (lesser low-similarity contradictions)
        Score towards 0 --> disjointed or contradictory reasoning steps
    """
    steps = [s.strip() for s in explanation.split('\n') if s.strip()]
    pairs = list(combinations(steps, 2))
    vectorizer = TfidfVectorizer().fit_transform(steps)
    similarity_matrix = cosine_similarity(vectorizer)
    contradiction_count = 0
    for i, j in combinations(range(len(steps)), 2):
        sim = similarity_matrix[i, j]
        if sim < 0.05:                                  # Low similarity = potential contradiction
            contradiction_count += 1
    total_pairs = len(pairs)
    return 1 - (contradiction_count / total_pairs) if total_pairs else 1.0

# Implement baseline relevance metric calculation
def baseline_relevance_score(explanation: str) -> float:
    """
    Computes relevance score for the explanation based on semantic alignment between each 
    supporting step and the conclusion. Final score is the avg similarity across all such pairs.

    @params:
        explanation (str): a multi-line explanation string; each line is treated as an individual step; 
                            last line treated as conclusion
        
    @returns:
        float: Relevance score between 0 and 1
        Score towards 1 --> better alignment between intermediate steps and conclusion
        Score towards 0 --> poor alignment
    """
    lines = [line for line in explanation.split('\n') if line.strip()]
    conclusion = lines[-1] if lines else ""
    supporting_steps = lines[:-1]
    if not supporting_steps or not conclusion:
        return 0.0
    vectorizer = TfidfVectorizer().fit([conclusion] + supporting_steps)
    conclusion_vec = vectorizer.transform([conclusion])
    step_vecs = vectorizer.transform(supporting_steps)
    sims = cosine_similarity(step_vecs, conclusion_vec).flatten()
    return sum(sims) / len(sims)

# Implement baseline redundancy metric calculation
def baseline_redundancy_score(explanation: str, threshold: float = 0.4) -> float:
    """
    Computes redundancy score of an explanation by identifying semantically similar 
    (repetitive) reasoning steps. Score is fraction of redundant pairs over all step pairs.

    @params:
        explanation (str): a multi-line explanation string; each line is treated as an individual step
        threshold (float, optional): Cosine similarity threshold above which a pair of steps 
                                     is considered redundant; defaults to 0.9

    @returns:
        float: Redundancy score between 0 and 1 
        Score towards 1: high redundancy in explanation
        Score towards 0: non-redundancy        
    """
    steps = [s.strip() for s in explanation.split('\n') if s.strip()]
    if len(steps) < 2:
        return 0.0                                  # No redundancy with lesser than 2 steps

    vectorizer = TfidfVectorizer().fit_transform(steps)                 # BERT based model instead of Tfidf
    similarity_matrix = cosine_similarity(vectorizer)                   # e5 model instead Tfidf
    
    redundant_pairs = 0
    total_pairs = 0
    
    for i, j in combinations(range(len(steps)), 2):
        if similarity_matrix[i, j] > threshold:
            redundant_pairs += 1
        total_pairs += 1

    # Compute fraction of redundant pairs
    redundancy_fraction = redundant_pairs / total_pairs if total_pairs else 0.0
    return redundancy_fraction


########## argLLM metrics #####################

def baseline_acceptability_score(argument_bag: Dict) -> float:
    """
    Computes the proportion of arguments from the entire "argument bag" that have
    strength greater than acceptability threshold (0.5).

    @params:
        argument_bag (Dict): a dictionary of argument objects, identified by key - "arguments"

    @returns:
        float: number of acceptable arguments in the entire set; value between 0 and 1 (inclusive)
    """
    threshold = 0.5
    arguments = argument_bag.get("arguments", {})
    if not arguments:
        return 0.0
    acceptable_count = 0
    for a in arguments.values(): 
        if a.get("strength", 0) > threshold:
            acceptable_count+=1
    return acceptable_count / len(arguments)


def baseline_circularity_score(argument_bag: Dict) -> float:
    """
    Computes the proportion of arguments from the entire "argument" bag that are
    involved in directed cycles.

    @params: 
        argument_bag (Dict): a dictionary of argument objects, identified by key - "arguments"

    @returns:
        float: number of arguments in the entire set that are involved in circularity; value between 0 and 1 (inclusive)
    """
    G = nx.DiGraph()
    for arg_name in argument_bag.get("arguments", {}):
        G.add_node(arg_name)

    for source, target in argument_bag.get("attacks", []) + argument_bag.get("supports", []):
        G.add_edge(source, target)

    cycles = list(nx.simple_cycles(G))
    cyclic_args = set(node for cycle in cycles for node in cycle)
    return len(cyclic_args) / len(argument_bag.get("arguments", {})) if argument_bag.get("arguments") else 0.0
