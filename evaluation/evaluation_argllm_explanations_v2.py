#!/usr/bin/env python3
"""
Evaluation script for ArgLLM generated explanations using argumentative metrics.
Processes JSONL output from ArgLLM claim verification and computes:
- Circularity
- Dialectical Acceptability 
- Dialectical Faithfulness
"""

import json
import os
import sys
from typing import List, Tuple, Dict, Any
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

# Import the argumentative metrics
from evaluation.Evaluating_Explanations.src.metrics.argumentative_metrics import (
    compute_circularity,
    compute_dialectical_acceptability,
    compute_dialectical_faithfulness
)

def extract_argumentative_structure(bag_data: Dict[str, Any]) -> Tuple[List[str], List[List[str]], List[List[str]]]:
    """
    Extract arguments, attack relations, and support relations from ArgLLM bag structure.
    
    Args:
        bag_data: Dictionary containing arguments, attacks, and supports from ArgLLM output
        
    Returns:
        Tuple of (arguments_list, attack_relations, support_relations)
    """
    # Extract argument names
    arguments = list(bag_data["arguments"].keys())
    
    # Extract attack relations (format: [attacker, attacked])
    attack_relations = bag_data.get("attacks", [])
    
    # Extract support relations (format: [supporter, supported])  
    support_relations = bag_data.get("supports", [])
    
    return arguments, attack_relations, support_relations

def determine_y_hat_arguments(bag_data: Dict[str, Any], prediction: float, threshold: float = 0.5) -> List[str]:
    """
    Determine which arguments support the predicted label (y_hat).
    
    Args:
        bag_data: Dictionary containing arguments structure
        prediction: Final prediction score from ArgLLM
        threshold: Decision threshold (default 0.5)
        
    Returns:
        List of argument names that support the predicted label
    """
    # The main claim argument is typically "db0"
    main_claim = "db0"
    y_hat_args = []
    
    if prediction > threshold:
        # Prediction is True, so arguments supporting the main claim are y_hat args
        y_hat_args.append(main_claim)
        # Add supporters of the main claim
        for support_pair in bag_data.get("supports", []):
            supporter, supported = support_pair
            if supported == main_claim:
                y_hat_args.append(supporter)
    else:
        # Prediction is False, so arguments attacking the main claim are y_hat args
        for attack_pair in bag_data.get("attacks", []):
            attacker, attacked = attack_pair
            if attacked == main_claim:
                y_hat_args.append(attacker)
    
    return list(set(y_hat_args))  # Remove duplicates

def determine_confidence_level(prediction: float) -> str:
    """
    Determine confidence level based on prediction score.
    
    Args:
        prediction: Prediction score from ArgLLM
        
    Returns:
        Confidence level: 'top', 'high', or 'low'
    """
    abs_distance_from_center = abs(prediction - 0.5)
    
    if abs_distance_from_center >= 0.4:  # Very confident (0.9+ or 0.1-)
        return 'top'
    elif abs_distance_from_center >= 0.2:  # Moderately confident (0.7+ or 0.3-)
        return 'high'
    else:  # Low confidence (around 0.5)
        return 'low'

def evaluate_single_explanation(entry: Dict[str, Any], use_estimated: bool = True) -> Dict[str, float]:
    """
    Evaluate a single ArgLLM explanation using argumentative metrics.
    
    Args:
        entry: Single entry from ArgLLM JSONL output
        use_estimated: Whether to use 'estimated' or 'base' results
        
    Returns:
        Dictionary with computed metrics
    """
    # Choose which results to evaluate
    result_type = "estimated" if use_estimated else "base"
    bag_data = entry[result_type]["bag"]
    prediction = entry[result_type]["prediction"]
    
    # Extract argumentative structure
    arguments, attack_relations, support_relations = extract_argumentative_structure(bag_data)
    
    # Determine y_hat arguments
    y_hat_args = determine_y_hat_arguments(bag_data, prediction)
    
    # Determine confidence level
    confidence_level = determine_confidence_level(prediction)
    
    # Compute metrics
    circularity_score = compute_circularity(arguments, attack_relations, support_relations)
    
    acceptability_score = compute_dialectical_acceptability(arguments, attack_relations, y_hat_args)
    
    faithfulness_score = compute_dialectical_faithfulness(
        arguments, attack_relations, support_relations, y_hat_args, confidence_level
    )
    
    return {
        'circularity': circularity_score,
        'dialectical_acceptability': acceptability_score,
        'dialectical_faithfulness': faithfulness_score,
        'prediction': prediction,
        'confidence_level': confidence_level,
        'num_arguments': len(arguments),
        'num_attacks': len(attack_relations),
        'num_supports': len(support_relations),
        'y_hat_args_count': len(y_hat_args)
    }

def main():
    # Configuration
    dataset_model = "truthfulclaim_mistral_temp_2"  # Using dataset and model
    INPUT_FILE = os.path.join(PROJECT_ROOT, "results", "generation", "argLLM_generation", 
                             dataset_model, "argllm_outputs_ollama.jsonl")
    OUTPUT_FILE = os.path.join(PROJECT_ROOT, "results", "temp_v2", f"argumentative_metrics_results_{dataset_model}.json")
    
    # Create output directory
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    print(f"Loading ArgLLM outputs from: {INPUT_FILE}")
    
    # Load the JSONL data
    entries = []
    try:
        with open(INPUT_FILE, 'r') as f:
            for line in f:
                entries.append(json.loads(line.strip()))
    except FileNotFoundError:
        print(f"❌ Error: Input file not found: {INPUT_FILE}")
        return
    
    print(f"Loaded {len(entries)} explanations for evaluation")
    
    # Evaluate all explanations
    results = []
    failed_evaluations = 0
    
    for i, entry in enumerate(entries, 1):
        try:
            print(f"Processing explanation {i}/{len(entries)}: {entry.get('claim', 'N/A')[:80]}...")
            
            # Evaluate both base and estimated versions
            base_metrics = evaluate_single_explanation(entry, use_estimated=False)
            estimated_metrics = evaluate_single_explanation(entry, use_estimated=True)
            
            result = {
                'id': entry['id'],
                'claim': entry['claim'],
                'true_label': entry['label'],
                'question': entry.get('question', ''),
                'base_metrics': base_metrics,
                'estimated_metrics': estimated_metrics
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"❌ Error evaluating explanation {i}: {e}")
            failed_evaluations += 1
            continue
    
    print(f"\n✅ Successfully evaluated {len(results)} explanations")
    if failed_evaluations > 0:
        print(f"⚠️  Failed to evaluate {failed_evaluations} explanations")
    
    # Compute summary statistics
    if results:
        base_metrics_df = pd.DataFrame([r['base_metrics'] for r in results])
        estimated_metrics_df = pd.DataFrame([r['estimated_metrics'] for r in results])
        
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        
        print("\nBASE METRICS:")
        print(base_metrics_df.describe())
        
        print("\nESTIMATED METRICS:")
        print(estimated_metrics_df.describe())
        
        # Save detailed results
        output_data = {
            'summary_stats': {
                'total_explanations': len(results),
                'failed_evaluations': failed_evaluations,
                'base_metrics_summary': base_metrics_df.describe().to_dict(),
                'estimated_metrics_summary': estimated_metrics_df.describe().to_dict()
            },
            'detailed_results': results
        }
        
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: {OUTPUT_FILE}")
        
        # Print some interesting insights
        print("\n" + "="*60)
        print("KEY INSIGHTS")
        print("="*60)
        
        avg_circularity_base = base_metrics_df['circularity'].mean()
        avg_circularity_est = estimated_metrics_df['circularity'].mean()
        
        avg_acceptability_base = base_metrics_df['dialectical_acceptability'].mean()
        avg_acceptability_est = estimated_metrics_df['dialectical_acceptability'].mean()
        
        avg_faithfulness_base = base_metrics_df['dialectical_faithfulness'].mean()
        avg_faithfulness_est = estimated_metrics_df['dialectical_faithfulness'].mean()
        
        print(f"Average Circularity - Base: {avg_circularity_base:.3f}, Estimated: {avg_circularity_est:.3f}")
        print(f"Average Acceptability - Base: {avg_acceptability_base:.3f}, Estimated: {avg_acceptability_est:.3f}")
        print(f"Average Faithfulness - Base: {avg_faithfulness_base:.3f}, Estimated: {avg_faithfulness_est:.3f}")
        
        # Check prediction accuracy vs true labels
        correct_predictions_base = sum(1 for r in results 
                                     if (r['base_metrics']['prediction'] > 0.5) == (r['true_label'] == 'true'))
        correct_predictions_est = sum(1 for r in results 
                                    if (r['estimated_metrics']['prediction'] > 0.5) == (r['true_label'] == 'true'))
        
        accuracy_base = correct_predictions_base / len(results)
        accuracy_est = correct_predictions_est / len(results)
        
        print(f"Prediction Accuracy - Base: {accuracy_base:.3f}, Estimated: {accuracy_est:.3f}")

if __name__ == "__main__":
    main()