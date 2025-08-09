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
        
        # Initialize with performance-optimized settings including intelligent sampling
        self.evaluator = OptimizedExplanationEvaluator(
            max_steps=15,      # Limit steps for long explanations  
            batch_size=16      # Efficient batch size
        )
        
        # For testing specific combinations (can be made configurable)
        self.datasets = ["truthfulqa", "strategyqa", "medqa"]  # Focus on problematic dataset for testing
        self.models = ["mistral", "llama", "qwen"]     # Focus on problematic model for testing
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
            original_step_counts = []
            
            for entry in sample_data:
                explanation = entry.get("cot_explanation", "")
                explanation_lengths.append(len(explanation))
                
                # Count potential steps before intelligent sampling
                original_steps = self.evaluator.clean_and_split_explanation(explanation)
                original_step_counts.append(len(original_steps))
                
                # Count steps after intelligent sampling
                if len(original_steps) > 15:
                    sampled_steps = self.evaluator.sampler.intelligent_sample(original_steps)
                    step_counts.append(len(sampled_steps))
                else:
                    step_counts.append(len(original_steps))
            
            if explanation_lengths and step_counts:
                avg_length = sum(explanation_lengths) / len(explanation_lengths)
                avg_original_steps = sum(original_step_counts) / len(original_step_counts)
                avg_final_steps = sum(step_counts) / len(step_counts)
                max_original_steps = max(original_step_counts) if original_step_counts else 0
                max_final_steps = max(step_counts) if step_counts else 0
                
                # Estimate processing time based on final (sampled) steps
                estimated_pairs_per_entry = min(avg_final_steps * avg_final_steps, 225)  # 15x15 max
                estimated_time_per_entry = estimated_pairs_per_entry * 0.01  # Rough estimate
                total_estimated_time = estimated_time_per_entry * total_entries / 60  # minutes
                
                complexity_info = {
                    "total_entries": total_entries,
                    "avg_explanation_length": int(avg_length),
                    "avg_original_steps": round(avg_original_steps, 1),
                    "avg_final_steps": round(avg_final_steps, 1),
                    "max_original_steps": max_original_steps,
                    "max_final_steps": max_final_steps,
                    "estimated_time_minutes": round(total_estimated_time, 1),
                    "intelligent_sampling_active": avg_original_steps > avg_final_steps
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
            if complexity.get("intelligent_sampling_active", False):
                logger.info(f"Intelligent sampling will be applied (avg steps: {complexity.get('avg_original_steps', 0):.1f} → {complexity.get('avg_final_steps', 0):.1f})")
            if complexity["estimated_time_minutes"] > 30:
                logger.warning(f"Long processing time estimated: {complexity['estimated_time_minutes']:.1f} minutes")
                logger.info("Consider using smaller max_steps or sampling if this is too long")
        
        all_results = []
        overall_start_time = time.time()
        
        for file_idx, filepath in enumerate(data_files):
            logger.info(f"Processing file {file_idx + 1}/{len(data_files)}: {os.path.basename(filepath)}")
            
            try:
                # Use the optimized evaluator with progress tracking and intelligent sampling
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
        logger.info("Starting CoT metrics evaluation with intelligent sampling and progress tracking")
        
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

        # Create MultiIndex columns for better presentation
        columns_to_reformat = [col for col in df.columns if col != "Dataset"]
        
        # Create hierarchical column structure
        multiindex_tuples = [("", "Dataset")]  # Dataset column stays as is

        for model in self.models:
            for metric in self.metric_keys:
                old_col = f"{model}_{metric}"
                if old_col in df.columns:
                    multiindex_tuples.append((model.capitalize(), metric.replace("_", " ").title()))

        # Rebuild dataframe with MultiIndex columns
        new_data = {}
        for col_tuple in multiindex_tuples:
            if col_tuple == ("", "Dataset"):
                new_data[col_tuple] = df["Dataset"]
            else:
                model_name = col_tuple[0].lower()
                metric_name = col_tuple[1].lower().replace(" ", "_")
                old_col = f"{model_name}_{metric_name}"
                if old_col in df.columns:
                    new_data[col_tuple] = df[old_col]

        # Apply MultiIndex
        df_multiindex = pd.DataFrame(new_data)
        df_multiindex.columns = pd.MultiIndex.from_tuples(df_multiindex.columns)
        
        # Save results
        output_dir = os.path.join(project_root, "results", "cot_method")
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        output_csv = os.path.join(output_dir, f"cot_metrics_summary_qa_{timestamp}.csv")
        output_html = os.path.join(output_dir, f"cot_metrics_summary_qa_{timestamp}.html")
        
        try:
            df_multiindex.to_csv(output_csv, index=False)
            logger.info(f"CSV saved: {output_csv}")
            
            # Create custom HTML with better styling
            html_content = self.create_custom_html_table(df_multiindex)
            with open(output_html, 'w') as f:
                f.write(html_content)
            logger.info(f"HTML saved: {output_html}")
            
            # Display if in Jupyter
            if JUPYTER_AVAILABLE:
                display(HTML(html_content))
            else:
                print("\nFinal Summary:")
                print("=" * 60)
                print(df_multiindex.to_string(index=False))
                
        except Exception as e:
            logger.error(f"Error saving results: {e}")

    def create_custom_html_table(self, df: pd.DataFrame) -> str:
        """Create custom HTML table."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>CoT Metrics Summary</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                }}
                h1 {{
                    color: #333;
                    text-align: center;
                }}
                .info-box {{
                    background-color: #f0f8ff;
                    border: 1px solid #007acc;
                    border-radius: 5px;
                    padding: 10px;
                    margin: 20px 0;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 20px 0;
                }}
                th.col_heading.level0 {{
                    text-align: center !important;
                    padding: 12px;
                    background-color: #f9f9f9;
                    font-weight: bold;
                    border: 1px solid #ddd;
                }}
                th.col_heading.level1 {{
                    text-align: center !important;
                    padding: 10px;
                    background-color: #f5f5f5;
                    border: 1px solid #ddd;
                }}
                td {{
                    text-align: center;
                    padding: 8px;
                    border: 1px solid #ddd;
                }}
                tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                tr:hover {{
                    background-color: #f0f8ff;
                }}
            </style>
        </head>
        <body>
            <h1>CoT Metrics Summary</h1>
            {df.to_html(index=False, escape=False, border=0)}
            <div class="info-box">
                <p><em>Generated on: {time.strftime("%Y-%m-%d %H:%M:%S")}</em></p>
                <p><em>Processing completed</em></p>
            </div>
        </body>
        </html>
        """
        return html_content

    def set_datasets_models(self, datasets: List[str], models: List[str]):
        """Allow dynamic configuration of datasets and models."""
        self.datasets = datasets
        self.models = models
        logger.info(f"Configuration updated - Datasets: {datasets}, Models: {models}")

def main():
    """Main execution function with configurable options."""
    summarizer = ProgressAwareMetricsSummarizer()
    
    print("CoT Metrics Evaluation with Intelligent Sampling")
    print("=" * 60)
    print("Features:")
    print("- Intelligent sampling for long explanations (>15 steps)")
    print("- Content-aware importance scoring") 
    print("- Progress tracking with ETA calculations")
    print("- Checkpoint saving for interruption recovery")
    print("- Detailed logging to 'cot_evaluation.log'")
    print()
    
    # For testing, focusing on problematic medqa+qwen combination
    print("Current configuration: medqa + qwen (for testing)")
    print("To test other combinations, modify the datasets and models lists in the code")
    print()
    
    try:
        summarizer.run_evaluation_with_checkpoints()
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        print("\nEvaluation interrupted. Check checkpoint files for partial results.")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        print(f"Evaluation failed: {e}")

# Example of how to use with different configurations
def example_custom_configuration():
    """Example showing how to run with custom dataset/model combinations."""
    summarizer = ProgressAwareMetricsSummarizer()
    
    # Test with multiple datasets and models
    summarizer.set_datasets_models(
        datasets=["medqa", "truthfulqa", "strategyqa"],
        models=["qwen", "llama", "mistral"]
    )
    
    summarizer.run_evaluation_with_checkpoints()

if __name__ == "__main__":
    main()