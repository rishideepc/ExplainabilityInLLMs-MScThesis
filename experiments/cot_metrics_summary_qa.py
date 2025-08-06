# cot_metrics_summary_qa.py
import os
import sys
import json
import time
from collections import defaultdict
from typing import List, Dict, Optional
import pandas as pd
import logging

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('cot_evaluation.log')
    ]
)
logger = logging.getLogger(__name__)

# Try to import IPython display, fallback if not available
try:
    from IPython.display import display, HTML
    JUPYTER_AVAILABLE = True
except ImportError:
    JUPYTER_AVAILABLE = False

project_root = os.path.abspath("..")
sys.path.append(project_root)

# Import the optimized evaluator
from evaluation.explanation_evaluation_calc_qa import OptimizedExplanationEvaluator

class ProgressAwareMetricsSummarizer:
    def __init__(self, base_path: Optional[str] = None):
        self.base_path = base_path or os.path.join(project_root, "results", "generation")
        
        # Initialize with performance-optimized settings
        self.evaluator = OptimizedExplanationEvaluator(
            max_steps=15,      # Limit steps for long explanations
            batch_size=16      # Efficient batch size
        )
        
        # For testing specific combinations
        self.datasets = ["truthfulqa", "strategyqa", "medqa"]  # Focus on problematic dataset
        self.models = ["mistral", "llama", "qwen"]     # Focus on problematic model
        self.metric_keys = ["redundancy", "weak_relevance", "strong_relevance"]
        
        self.dataset_name_map = {
            "medqa": "MedQA",
            "truthfulqa": "TruthfulQA",
            "strategyqa": "StrategyQA",
            "commonsenseqa": "CommonSenseQA"
        }

    def find_data_files(self, dataset: str, model: str) -> List[str]:
        """Find all JSONL files for a dataset-model combination."""
        subfolder = f"{dataset}_{model}"
        folder_path = os.path.join(self.base_path, subfolder)
        
        if not os.path.exists(folder_path):
            logger.error(f"Missing folder: {folder_path}")
            return []
            
        jsonl_files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.endswith(".jsonl")
        ]
        
        if not jsonl_files:
            logger.error(f"No .jsonl files in {folder_path}")
        else:
            logger.info(f"Found {len(jsonl_files)} files: {[os.path.basename(f) for f in jsonl_files]}")
            
        return jsonl_files

    def preview_data_complexity(self, filepath: str, sample_size: int = 5) -> Dict:
        """Analyze data complexity to estimate processing time."""
        logger.info(f"Analyzing data complexity: {filepath}")
        
        try:
            with open(filepath, "r", encoding='utf-8') as f:
                lines = f.readlines()
                
            total_entries = len([l for l in lines if l.strip()])
            logger.info(f"Total entries: {total_entries}")
            
            # Sample a few entries to analyze
            sample_data = []
            step = max(1, total_entries // sample_size)
            
            for i in range(0, min(total_entries, sample_size * step), step):
                if i < len(lines) and lines[i].strip():
                    try:
                        entry = json.loads(lines[i].strip())
                        sample_data.append(entry)
                    except:
                        continue
            
            # Analyze explanation lengths
            explanation_lengths = []
            step_counts = []
            
            for entry in sample_data:
                explanation = entry.get("cot_explanation", "")
                explanation_lengths.append(len(explanation))
                
                # Count potential steps
                steps = self.evaluator.clean_and_split_explanation(explanation)
                step_counts.append(len(steps))
            
            if explanation_lengths and step_counts:
                avg_length = sum(explanation_lengths) / len(explanation_lengths)
                avg_steps = sum(step_counts) / len(step_counts)
                max_steps = max(step_counts)
                
                # Estimate processing time
                estimated_pairs_per_entry = min(avg_steps * avg_steps, 400)  # Cap at reasonable limit
                estimated_time_per_entry = estimated_pairs_per_entry * 0.01  # Rough estimate
                total_estimated_time = estimated_time_per_entry * total_entries / 60  # minutes
                
                complexity_info = {
                    "total_entries": total_entries,
                    "avg_explanation_length": int(avg_length),
                    "avg_steps": round(avg_steps, 1),
                    "max_steps": max_steps,
                    "estimated_time_minutes": round(total_estimated_time, 1)
                }
                
                logger.info(f"Complexity analysis: {complexity_info}")
                return complexity_info
            
        except Exception as e:
            logger.error(f"Error analyzing complexity: {e}")
            
        return {"total_entries": 0, "estimated_time_minutes": 0}

    def evaluate_dataset_model_with_progress(self, dataset: str, model: str) -> Optional[Dict[str, float]]:
        """Evaluate with detailed progress tracking and time estimates."""
        data_files = self.find_data_files(dataset, model)
        if not data_files:
            return None
        
        logger.info(f"=== Starting evaluation for {dataset}_{model} ===")
        
        # Analyze complexity first
        for filepath in data_files:
            complexity = self.preview_data_complexity(filepath)
            if complexity["estimated_time_minutes"] > 30:
                logger.warning(f"Long processing time estimated: {complexity['estimated_time_minutes']:.1f} minutes")
                logger.info("Consider using smaller max_steps or sampling if this is too long")
        
        all_results = []
        overall_start_time = time.time()
        
        for file_idx, filepath in enumerate(data_files):
            logger.info(f"Processing file {file_idx + 1}/{len(data_files)}: {os.path.basename(filepath)}")
            
            try:
                # Use the optimized evaluator with progress tracking
                results = self.evaluator.evaluate_all_cot_with_progress(filepath)
                
                if results:
                    all_results.extend(results)
                    logger.info(f"    → Added {len(results)} results. Total so far: {len(all_results)}")
                else:
                    logger.warning(f"    → No valid results from {filepath}")
                    
            except Exception as e:
                logger.error(f"Failed to process {filepath}: {e}")
                continue
        
        if not all_results:
            logger.error(f"No valid results for {dataset}_{model}")
            return None
        
        # Compute metrics with error handling
        metrics = {}
        for metric in self.metric_keys:
            try:
                values = [
                    entry[metric] for entry in all_results 
                    if metric in entry and isinstance(entry[metric], (int, float)) and entry[metric] != -1.0
                ]
                
                if values:
                    avg_value = sum(values) / len(values)
                    metrics[f"{model}_{metric}"] = round(avg_value, 4)
                    logger.info(f"    {metric}: {avg_value:.4f} (from {len(values)} valid values)")
                else:
                    metrics[f"{model}_{metric}"] = "N/A"
                    logger.warning(f"    {metric}: No valid values found")
                    
            except Exception as e:
                logger.error(f"Error computing {metric}: {e}")
                metrics[f"{model}_{metric}"] = "N/A"
        
        total_time = time.time() - overall_start_time
        logger.info(f"=== Completed {dataset}_{model} in {total_time/60:.2f} minutes ===")
        logger.info(f"Final metrics: {metrics}")
        
        return metrics

    def run_evaluation_with_checkpoints(self):
        """Run evaluation with checkpoint saving."""
        logger.info("Starting CoT metrics evaluation with progress tracking")
        
        summary = defaultdict(dict)
        
        for dataset in self.datasets:
            logger.info(f"\n{'='*50}")
            logger.info(f"PROCESSING DATASET: {dataset}")
            logger.info(f"{'='*50}")
            
            for model in self.models:
                logger.info(f"\nProcessing model: {model}")
                
                try:
                    metrics = self.evaluate_dataset_model_with_progress(dataset, model)
                    
                    if metrics:
                        summary[dataset].update(metrics)
                        
                        # Save checkpoint after each model
                        checkpoint_path = os.path.join(
                            project_root, "results", "cot_method", 
                            f"checkpoint_{dataset}_{model}.json"
                        )
                        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                        
                        with open(checkpoint_path, 'w') as f:
                            json.dump(dict(summary), f, indent=2)
                        logger.info(f"Checkpoint saved: {checkpoint_path}")
                        
                    else:
                        logger.error(f"No results for {dataset}_{model}")
                        for metric in self.metric_keys:
                            summary[dataset][f"{model}_{metric}"] = "N/A"
                            
                except Exception as e:
                    logger.error(f"Fatal error processing {dataset}_{model}: {e}")
                    for metric in self.metric_keys:
                        summary[dataset][f"{model}_{metric}"] = "ERROR"
        
        # Generate final summary
        if summary:
            self.generate_final_summary(summary)
        else:
            logger.error("No data processed successfully")

    def generate_final_summary(self, summary: Dict):
        """Generate and save final summary."""
        logger.info("Generating final summary...")
        
        df = pd.DataFrame.from_dict(summary, orient="index").reset_index()
        df.rename(columns={"index": "Dataset"}, inplace=True)
        df["Dataset"] = df["Dataset"].map(lambda x: self.dataset_name_map.get(x, x.title()))
        
        # Save results
        output_dir = os.path.join(project_root, "results", "cot_method")
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        output_csv = os.path.join(output_dir, f"cot_metrics_summary_qa_{timestamp}.csv")
        output_html = os.path.join(output_dir, f"cot_metrics_summary_qa_{timestamp}.html")
        
        try:
            df.to_csv(output_csv, index=False)
            logger.info(f"CSV saved: {output_csv}")
            
            df.to_html(output_html, index=False, escape=False)
            logger.info(f"HTML saved: {output_html}")
            
            # Display if in Jupyter
            if JUPYTER_AVAILABLE:
                display(HTML(f"<h3>CoT Metrics Summary</h3>{df.to_html(index=False)}"))
            else:
                print("\nFinal Summary:")
                print("=" * 60)
                print(df.to_string(index=False))
                
        except Exception as e:
            logger.error(f"Error saving results: {e}")

def main():
    """Main execution function."""
    summarizer = ProgressAwareMetricsSummarizer()
    
    # Set specific datasets and models for testing
    print("Testing with medqa + qwen combination...")
    print("This will provide detailed progress tracking and time estimates.")
    print("Check the log file 'cot_evaluation.log' for detailed progress.")
    
    try:
        summarizer.run_evaluation_with_checkpoints()
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        print("\nEvaluation interrupted. Check checkpoint files for partial results.")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        print(f"Evaluation failed: {e}")

if __name__ == "__main__":
    main()