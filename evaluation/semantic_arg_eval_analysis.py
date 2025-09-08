import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd

CLAIM_VERIF_DATASETS = ["TruthfulClaim", "MedClaim", "StrategyClaim"]
QA_DATASETS          = ["TruthfulQA", "MedQA", "StrategyQA"]
MODELS               = ["Mistral", "Llama", "Qwen"]

SUBMETRICS: List[Tuple[str, str]] = [
    ("Base Semantic Acceptability",      "base_sem_acceptability"),
    ("Base Semantic Circularity",        "base_sem_circularity"),
    ("Estimated Semantic Acceptability", "estimated_sem_acceptability"),
    ("Estimated Semantic Circularity",   "estimated_sem_circularity"),
]

DEFAULT_ROOT = "/vol/bitbucket/rc1124/MSc_Individual_Project/ExplainabilityInLLMs-MScThesis/results/argLLM_metrics_semantic"

log = logging.getLogger("compile-semantic")

def init_logging(verbose: bool) -> None:
    """
    Helper function; init basic logging
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)-8s | %(message)s")


def _dataset_model_to_path(root: Path, dataset: str, model: str) -> Path:
    """
    Helper function; builds the expected file path for results
    """
    fname = f"{dataset.lower()}_{model.lower()}.csv"
    return root / fname

def _safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    """
    Helper function; reads CSV from input
    
    @params: input path
    
    @returns: Pandas DataFrame or None
    """
    if not path.exists():
        log.warning(f"[missing] {path}")
        return None
    try:
        df = pd.read_csv(path)
        if df.empty:
            log.warning(f"[empty] {path}")
            return None
        return df
    except Exception as e:
        log.error(f"[error] reading {path}: {e}")
        return None

def _mean_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Computes mean of required metric columns

    @params: Pandas DataFrame 
    
    @returns: dictionary of mean values (float), keys = column names
    """
    out: Dict[str, float] = {}
    for _, col in SUBMETRICS:
        if col in df.columns:
            out[col] = float(pd.to_numeric(df[col], errors="coerce").mean())
        else:
            log.warning(f"[missing-col] column '{col}' not found.")
            out[col] = float("nan")
    return out

def build_summary_table(root: Path, datasets: List[str], models: List[str]) -> pd.DataFrame:
    """
    Builds a summary table of mean metrics for given datasets and models

    @params: root directory, datasets, models 
    
    @returns: DataFrame with mean results
    """
    col_tuples = [(model, pretty) for model in models for (pretty, _) in SUBMETRICS]
    columns = pd.MultiIndex.from_tuples(col_tuples, names=["Model", "Metric"])
    result = pd.DataFrame(index=datasets, columns=columns, dtype=float)

    for ds in datasets:
        for model in models:
            path = _dataset_model_to_path(root, ds, model)
            df = _safe_read_csv(path)
            if df is None:
                continue
            means = _mean_metrics(df)
            for pretty, col in SUBMETRICS:
                result.loc[ds, (model, pretty)] = means[col]
            log.debug(f"[ok] {ds} / {model} from {path}")

    return result

def format_floats(df: pd.DataFrame, precision: int = 3) -> pd.DataFrame:
    """
    Helper function; formats floats in DataFrame to a given precision
    """
    return df.round(precision)

def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Helpr function; flattens MultiIndex columns to single level with " | " separator
    """
    flat_cols = [f"{top} | {sub}" for (top, sub) in df.columns]
    out = df.copy()
    out.columns = flat_cols
    return out

def save_csv(df: pd.DataFrame, path: Path, precision: int) -> None:
    """
    Helper function; write CSV file for results upto a given precision
    """
    df_out = format_floats(df, precision=precision)
    df_out.to_csv(path, index=True)
    log.info(f"[save] CSV -> {path}")

def save_html(claim_df: pd.DataFrame, qa_df: pd.DataFrame, path: Path, precision: int) -> None:
    """
    Helper function; writes simple HTML tables for results
    """
    claim_styler = (
        format_floats(claim_df, precision)
        .style.set_table_styles(
            [
                {"selector": "table", "props": [("border-collapse", "collapse"), ("margin", "1em 0")]},
                {"selector": "th, td", "props": [("border", "1px solid #ddd"), ("padding", "6px 8px")]},
                {"selector": "th", "props": [("background-color", "#f7f7f7")]},
                {"selector": "caption", "props": [("caption-side", "top"), ("font-weight", "bold"), ("margin-bottom", "0.5em")]},
            ]
        ).set_caption("Table 1 — Claim Verification (Semantic Metrics)")
         .set_properties(**{"text-align": "right"})
    )

    qa_styler = (
        format_floats(qa_df, precision)
        .style.set_table_styles(
            [
                {"selector": "table", "props": [("border-collapse", "collapse"), ("margin", "1em 0")]},
                {"selector": "th, td", "props": [("border", "1px solid #ddd"), ("padding", "6px 8px")]},
                {"selector": "th", "props": [("background-color", "#f7f7f7")]},
                {"selector": "caption", "props": [("caption-side", "top"), ("font-weight", "bold"), ("margin-bottom", "0.5em")]},
            ]
        ).set_caption("Table 2 — QA (Semantic Metrics)")
         .set_properties(**{"text-align": "right"})
    )

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>argLLM Semantic Metrics — Summary</title>
</head>
<body>
<h1>argLLM Semantic Metrics — Summary</h1>
{claim_styler.to_html()}
{qa_styler.to_html()}
</body>
</html>"""
    path.write_text(html, encoding="utf-8")
    log.info(f"[save] HTML -> {path}")

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compile semantic argLLM metrics into summary tables.")
    ap.add_argument("--root", default=DEFAULT_ROOT,
                    help="Directory containing per-run CSVs (<dataset>_<model>.csv).")
    ap.add_argument("--out_dir", default="results/semantic_summaries",
                    help="Directory to save summary CSVs and HTML.")
    ap.add_argument("--precision", type=int, default=3,
                    help="Decimal places for rounding outputs.")
    ap.add_argument("--verbose", action="store_true",
                    help="Verbose logging.")
    return ap.parse_args()

def main() -> None:
    args = parse_args()
    init_logging(args.verbose)

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"[config] root={root}")
    log.info(f"[config] out_dir={out_dir}")
    log.info(f"[config] precision={args.precision}")
    log.info(f"[config] datasets-claim={CLAIM_VERIF_DATASETS} datasets-qa={QA_DATASETS}")
    log.info(f"[config] models={MODELS}")

    log.info("[build] Claim Verification table...")
    claim_df = build_summary_table(root, CLAIM_VERIF_DATASETS, MODELS)

    log.info("[build] QA table...")
    qa_df = build_summary_table(root, QA_DATASETS, MODELS)

    claim_csv = out_dir / "claim_verification_summary.csv"
    qa_csv    = out_dir / "qa_summary.csv"
    save_csv(flatten_columns(claim_df), claim_csv, args.precision)
    save_csv(flatten_columns(qa_df), qa_csv, args.precision)

    html_path = out_dir / "argllm_semantic_summary.html"
    save_html(claim_df, qa_df, html_path, args.precision)

    log.info("[done] All outputs saved.")

if __name__ == "__main__":
    main()
