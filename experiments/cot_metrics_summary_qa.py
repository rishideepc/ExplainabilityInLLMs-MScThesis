from __future__ import annotations

import os
import sys
import json
import time
import argparse
import logging
from collections import defaultdict
from typing import List, Dict, Optional, Any, Iterable

import pandas as pd

try:
    from IPython.display import display, HTML 
    JUPYTER_AVAILABLE = True
except Exception:
    JUPYTER_AVAILABLE = False

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from evaluation.explanation_evaluation_calc_qa import OptimizedExplanationEvaluator
except Exception as e:  
    raise ImportError(
        "Failed to import OptimizedExplanationEvaluator from "
        "'evaluation.explanation_evaluation_calc_qa'. "
        f"Error: {e}"
    )


def setup_logger(level: int = logging.INFO, log_file: Optional[str] = None) -> logging.Logger:
    """
    Helper function; creates a module-level logger with stream + optional file handler
    """    logger = logging.getLogger("cot_metrics_summary_qa")
    logger.setLevel(level)

    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger



class ProgressAwareMetricsSummarizer:
    """
    Orchestrates dataset–model evaluation of CoT metrics with:
      - intelligent step sampling
      - per-(dataset, model) checkpointing

    @params: root directories, output directory, datasets, models, evaluator parameters, logger

    @returns: final summary dictionary
    """

    DATASET_NAME_MAP = {
        "medqa": "MedQA",
        "truthfulqa": "TruthfulQA",
        "strategyqa": "StrategyQA",
        "commonsenseqa": "CommonSenseQA",
    }

    METRIC_KEYS = ("redundancy", "weak_relevance", "strong_relevance")

    def __init__(
        self,
        base_dir: Optional[str],
        out_dir: Optional[str],
        datasets: List[str],
        models: List[str],
        max_steps: int,
        batch_size: int,
        logger: logging.Logger,
    ):
        self.base_dir = base_dir or os.path.join(PROJECT_ROOT, "results", "generation")
        self.out_dir = out_dir or os.path.join(PROJECT_ROOT, "results", "cot_method")
        self.datasets = datasets
        self.models = models
        self.logger = logger

        os.makedirs(self.out_dir, exist_ok=True)

        self.evaluator = OptimizedExplanationEvaluator(
            max_steps=max_steps,
            batch_size=batch_size,
        )


    def find_data_files(self, dataset: str, model: str) -> List[str]:
        """
        Helper function; returns list of JSONL files under '{dataset}_{model}' subdir
        """        
        subfolder = f"{dataset}_{model}"
        folder_path = os.path.join(self.base_dir, subfolder)

        if not os.path.isdir(folder_path):
            self.logger.error(f"Missing folder: {folder_path}")
            return []

        files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.endswith(".jsonl") and os.path.isfile(os.path.join(folder_path, f))
        ]
        if not files:
            self.logger.error(f"No .jsonl files in {folder_path}")
        else:
            self.logger.info(f"Found {len(files)} file(s) in {subfolder}: {[os.path.basename(f) for f in files]}")
        return sorted(files)


    def _iter_jsonl(self, filepath: str) -> Iterable[Dict[str, Any]]:
        """
        Helper function; yields parsed JSON objects from a JSONL file, skipping malformed lines
        """        
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    yield json.loads(s)
                except json.JSONDecodeError:
                    self.logger.warning(f"Malformed JSON skipped in {os.path.basename(filepath)}")
                    continue

    def preview_data_complexity(self, filepath: str, sample_size: int = 5) -> Dict[str, Any]:
        """
        Samples entries to estimate explanation length (number of propositions) and rough runtime

        @params: input file path, sample size for preview

        @returns: dictionary with complexity info
        """
        self.logger.info(f"Analyzing data complexity: {os.path.basename(filepath)}")

        total_entries = 0
        samples: List[Dict[str, Any]] = []

        stride = None
        for idx, entry in enumerate(self._iter_jsonl(filepath)):
            total_entries += 1
            if stride is None and total_entries > sample_size:
                stride = max(1, total_entries // sample_size)
            if len(samples) < sample_size:
                samples.append(entry)
            elif stride and idx % stride == 0:
                replace_idx = (idx // stride) % sample_size
                samples[replace_idx] = entry

        if total_entries == 0:
            return {"total_entries": 0, "estimated_time_minutes": 0.0}

        explanation_lengths: List[int] = []
        original_step_counts: List[int] = []
        final_step_counts: List[int] = []

        for entry in samples:
            explanation = entry.get("cot_explanation", "") or ""
            explanation_lengths.append(len(explanation))

            original_steps = self.evaluator.clean_and_split_explanation(explanation)
            original_step_counts.append(len(original_steps))

            if len(original_steps) > self.evaluator.max_steps:
                sampled = self.evaluator.sampler.intelligent_sample(original_steps)
                final_step_counts.append(len(sampled))
            else:
                final_step_counts.append(len(original_steps))

        if not explanation_lengths:
            return {"total_entries": total_entries, "estimated_time_minutes": 0.0}

        avg_len = int(round(sum(explanation_lengths) / len(explanation_lengths)))
        avg_orig = round(sum(original_step_counts) / len(original_step_counts), 1) if original_step_counts else 0.0
        avg_final = round(sum(final_step_counts) / len(final_step_counts), 1) if final_step_counts else 0.0
        max_orig = max(original_step_counts) if original_step_counts else 0
        max_final = max(final_step_counts) if final_step_counts else 0

        est_pairs_per_entry = min((avg_final or 0) * (avg_final or 0), self.evaluator.max_steps ** 2)
        est_time_per_entry_s = est_pairs_per_entry * 0.01
        est_total_min = round((est_time_per_entry_s * total_entries) / 60.0, 1)

        info = {
            "total_entries": total_entries,
            "avg_explanation_length": avg_len,
            "avg_original_steps": avg_orig,
            "avg_final_steps": avg_final,
            "max_original_steps": max_orig,
            "max_final_steps": max_final,
            "estimated_time_minutes": est_total_min,
            "intelligent_sampling_active": avg_orig > avg_final,
        }
        self.logger.info(f"Complexity: {info}")
        return info


    def evaluate_dataset_model_with_progress(self, dataset: str, model: str) -> Optional[Dict[str, float]]:
        """
        Evaluates metrics for a dataset–model pair across files
        
        @params: dataset, model

        @returns: dictionary of averaged metrics or None
        """        
        files = self.find_data_files(dataset, model)
        if not files:
            return None

        self.logger.info(f"=== Starting evaluation for {dataset}_{model} ===")

        for fp in files:
            complexity = self.preview_data_complexity(fp)
            if complexity.get("intelligent_sampling_active"):
                self.logger.info(
                    "Intelligent sampling: avg steps "
                    f"{complexity.get('avg_original_steps', 0):.1f} → {complexity.get('avg_final_steps', 0):.1f}"
                )
            if (complexity.get("estimated_time_minutes") or 0) > 30:
                self.logger.warning(
                    f"Estimated runtime {complexity['estimated_time_minutes']:.1f} min; "
                    "consider lowering --max-steps or processing fewer files."
                )

        all_results: List[Dict[str, Any]] = []
        t0 = time.time()

        for i, fp in enumerate(files, 1):
            self.logger.info(f"[{i}/{len(files)}] {os.path.basename(fp)}")
            try:
                results = self.evaluator.evaluate_all_cot_with_progress(fp)
            except Exception as e:
                self.logger.error(f"Failed to process {os.path.basename(fp)}: {e}")
                continue

            if results:
                all_results.extend(results)
                self.logger.info(f"→ {len(results)} rows (cum: {len(all_results)})")
            else:
                self.logger.warning("→ No valid results")

        if not all_results:
            self.logger.error(f"No valid results for {dataset}_{model}")
            return None

        metrics: Dict[str, float | None] = {}
        for key in self.METRIC_KEYS:
            vals = [
                float(row[key])
                for row in all_results
                if isinstance(row.get(key), (int, float)) and float(row[key]) != -1.0
            ]
            metrics[f"{model}_{key}"] = round(sum(vals) / len(vals), 4) if vals else None
            if vals:
                self.logger.info(f"    {key}: {metrics[f'{model}_{key}']:.4f}  (n={len(vals)})")
            else:
                self.logger.warning(f"    {key}: no valid values")

        self.logger.info(f"=== Completed {dataset}_{model} in {(time.time()-t0)/60:.2f} min ===")
        return metrics


    def run_evaluation_with_checkpoints(self) -> Dict[str, Dict[str, Any]]:
        """
        Runs all dataset–model pairs, writing per-combo checkpoints
        
        @returns: nested dictionary of evaluation summary
        """        
        self.logger.info("Starting CoT metrics evaluation with intelligent sampling + progress")

        summary: Dict[str, Dict[str, Any]] = defaultdict(dict)

        for dataset in self.datasets:
            self.logger.info("\n" + "=" * 60)
            self.logger.info(f"DATASET: {dataset}")
            self.logger.info("=" * 60)

            for model in self.models:
                self.logger.info(f"Model: {model}")
                try:
                    metrics = self.evaluate_dataset_model_with_progress(dataset, model)
                    if metrics:
                        summary[dataset].update(metrics)
                    else:
                        for key in self.METRIC_KEYS:
                            summary[dataset][f"{model}_{key}"] = None

                    ckpt = os.path.join(self.out_dir, f"checkpoint_{dataset}_{model}.json")
                    with open(ckpt, "w", encoding="utf-8") as f:
                        json.dump(summary, f, indent=2)
                    self.logger.info(f"Checkpoint saved: {ckpt}")

                except Exception as e:
                    self.logger.error(f"Fatal error for {dataset}_{model}: {e}")
                    for key in self.METRIC_KEYS:
                        summary[dataset][f"{model}_{key}"] = None

        return summary


    def generate_final_summary(self, summary: Dict[str, Dict[str, Any]], tag: Optional[str] = None) -> Dict[str, str]:
        """
        Builds MultiIndex table, saves CSV + HTML
        
        @params: dictionary summary, tag (optional string for filenames)

        @returns: dictionary with paths to CSV and HTML files
        """        
        self.logger.info("Generating final summary…")

        df = pd.DataFrame.from_dict(summary, orient="index").reset_index().rename(columns={"index": "Dataset"})
        df["Dataset"] = df["Dataset"].map(lambda x: self.DATASET_NAME_MAP.get(x, x.title()))

        tuples = [("", "Dataset")]
        for m in self.models:
            for key in self.METRIC_KEYS:
                tuples.append((m.capitalize(), key.replace("_", " ").title()))

        data_cols: Dict[tuple, Any] = {}
        for tpl in tuples:
            if tpl == ("", "Dataset"):
                data_cols[tpl] = df["Dataset"]
            else:
                model_l = tpl[0].lower()
                metric_k = tpl[1].lower().replace(" ", "_")
                col = f"{model_l}_{metric_k}"
                data_cols[tpl] = df[col] if col in df.columns else None

        out_df = pd.DataFrame(data_cols)
        out_df.columns = pd.MultiIndex.from_tuples(out_df.columns)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        suffix = f"_{tag}" if tag else ""
        csv_path = os.path.join(self.out_dir, f"cot_metrics_summary_qa{suffix}_{timestamp}.csv")
        html_path = os.path.join(self.out_dir, f"cot_metrics_summary_qa{suffix}_{timestamp}.html")

        out_df.to_csv(csv_path, index=False)
        self.logger.info(f"CSV saved: {csv_path}")

        html = self._create_custom_html_table(out_df)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        self.logger.info(f"HTML saved: {html_path}")

        if JUPYTER_AVAILABLE:
            display(HTML(html))

        try:
            print("\nFinal Summary (compact):")
            print("=" * 60)
            print(out_df.to_string(index=False))
        except Exception:
            pass

        return {"csv": csv_path, "html": html_path}


    def _create_custom_html_table(self, df: pd.DataFrame) -> str:
        """
        Helper function; returns an HTML file string for summary table
        """        
        table_html = df.to_html(index=False, escape=False, border=0)

        return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>CoT Metrics Summary (QA)</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 20px;
      color: #222;
    }}
    h1 {{
      text-align: center;
      margin-bottom: 10px;
    }}
    .info-box {{
      background-color: #f0f8ff;
      border: 1px solid #007acc;
      border-radius: 6px;
      padding: 10px 12px;
      margin: 16px 0;
      font-size: 0.95rem;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      margin: 20px 0;
      table-layout: fixed;
    }}
    th, td {{
      border: 1px solid #ddd;
      text-align: center;
      padding: 8px;
      word-wrap: break-word;
    }}
    thead th {{
      position: sticky;
      top: 0;
      background: #f9f9f9;
      z-index: 2;
    }}
    thead tr:nth-child(1) th {{
      background: #f3f3f3;
      font-weight: 700;
    }}
    tr:nth-child(even) {{ background-color: #fafafa; }}
    tr:hover {{ background-color: #eef7ff; }}
    .footer {{
      margin-top: 16px;
      font-size: 0.9rem;
      color: #555;
    }}
  </style>
</head>
<body>
  <h1>CoT Metrics Summary (QA)</h1>
  {table_html}
  <div class="info-box">
    <p><em>Generated on: {time.strftime("%Y-%m-%d %H:%M:%S")}</em></p>
    <p><em>Metrics: Redundancy, Weak Relevance, Strong Relevance</em></p>
  </div>
  <div class="footer">Processing completed.</div>
</body>
</html>"""



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize CoT metrics across QA dataset–model runs.")
    parser.add_argument("--base-dir", type=str, default=os.path.join(PROJECT_ROOT, "results", "generation"),
                        help="Root directory containing '{dataset}_{model}' subfolders with .jsonl files.")
    parser.add_argument("--out-dir", type=str, default=os.path.join(PROJECT_ROOT, "results", "cot_method"),
                        help="Directory for checkpoints and final summaries.")
    parser.add_argument("--datasets", nargs="+",
                        default=["truthfulqa", "strategyqa", "medqa"],
                        help="Datasets to evaluate (folder prefixes).")
    parser.add_argument("--models", nargs="+",
                        default=["mistral", "llama", "qwen"],
                        help="Models to evaluate (folder suffixes).")
    parser.add_argument("--max-steps", type=int, default=15, help="Max steps after sampling.")
    parser.add_argument("--batch-size", type=int, default=16, help="Evaluator batch size.")
    parser.add_argument("--log-file", type=str, default=os.path.join(PROJECT_ROOT, "cot_evaluation.log"),
                        help="Path to log file (will be created/overwritten).")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging verbosity.")
    parser.add_argument("--tag", type=str, default=None, help="Optional tag appended to output filenames.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger(level=getattr(logging, args.log_level), log_file=args.log_file)

    logger.info("CoT Metrics Evaluation (QA) — Intelligent Sampling & Progress")
    logger.info(f"Base dir: {args.base_dir}")
    logger.info(f"Out  dir: {args.out_dir}")
    logger.info(f"Datasets: {args.datasets}")
    logger.info(f"Models  : {args.models}")
    logger.info(f"max_steps={args.max_steps}, batch_size={args.batch_size}")

    summarizer = ProgressAwareMetricsSummarizer(
        base_dir=args.base_dir,
        out_dir=args.out_dir,
        datasets=args.datasets,
        models=args.models,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        logger=logger,
    )

    try:
        summary = summarizer.run_evaluation_with_checkpoints()
        summarizer.generate_final_summary(summary, tag=args.tag)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user. Check checkpoints for partial results.")
    except Exception as e:
        logger.exception(f"Evaluation failed: {e}")


if __name__ == "__main__":
    main()
