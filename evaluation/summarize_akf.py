import os
import re
import sys
import glob
import argparse
from typing import List, Tuple
import pandas as pd
import numpy as np

MODEL_CANON = {
    'llama': 'llama', 'llama2': 'llama', 'llama-2': 'llama',
    'mistral': 'mistral',
    'qwen': 'qwen', 'qwen2': 'qwen', 'qwen2.5': 'qwen',
}
MODEL_PATTERNS = [
    re.compile(r'(?:^|[_\-])(llama(?:-?2)?)(?:[_\-]|\.|$)', re.I),
    re.compile(r'(?:^|[_\-])(mistral)(?:[_\-]|\.|$)', re.I),
    re.compile(r'(?:^|[_\-])(qwen(?:2(?:\.5)?)?)(?:[_\-]|\.|$)', re.I),
]
FILE_PAT = re.compile(r'results_(?P<dataset>.+)_(?P<model>llama|mistral|qwen)(?:[^/]*)\.csv$', re.I)

def infer_pair_from_path(path: str) -> Tuple[str, str]:
    """
    Infers (dataset, model) pair from file path
    
    @params: input file path
    
    @returns: (dataset, model) tuple
    """
    fname = os.path.basename(path)
    m = FILE_PAT.search(fname)
    if m:
        dataset = m.group('dataset').strip().lower()
        model_raw = m.group('model').strip().lower()
        return dataset, MODEL_CANON.get(model_raw, model_raw)

    model = 'unknown'
    for pat in MODEL_PATTERNS:
        mm = pat.search(fname)
        if mm:
            model = MODEL_CANON.get(mm.group(1).lower(), 'unknown')
            break

    base = re.sub(r'\.csv$', '', fname, flags=re.I)
    base = re.sub(r'^results[_\-]*', '', base, flags=re.I)
    base = re.sub(r'(llama(?:-?2)?|mistral|qwen(?:2(?:\.5)?)?)', '', base, flags=re.I)
    base = re.sub(r'[_\-]+', '_', base).strip('_')
    dataset = base.lower() if base else 'unknown'

    if dataset in ('', 'unknown'):
        for part in reversed(os.path.dirname(path).split(os.sep)):
            if re.search(r'claim|med|strategy|truth|commonsense|strategyqa|truthful', part, re.I):
                dataset = part.lower()
                break

    return (dataset or 'unknown'), (model or 'unknown')

def find_csvs(inputs: List[str]) -> List[str]:
    """
    Helper function; collects CSV file paths from provided folders/files
    """
    csvs = []
    for inp in inputs:
        if os.path.isdir(inp):
            csvs.extend(glob.iglob(os.path.join(inp, '**', '*.csv'), recursive=True))
        elif os.path.isfile(inp) and inp.lower().endswith('.csv'):
            csvs.append(inp)
    return sorted(set(csvs))

REQUIRED_COLS = {
    'SIMPLE_AKF', 'PC_soft', 'NS_soft', 'PC_bin', 'NS_bin',
    'uncertain', 'polarity', 'sample_id'
}
def looks_like_akf_node_file(df: pd.DataFrame) -> bool:
    """
    Helper function; returns True iff DataFrame has required AKF-Lite node columns
    """
    return all(c in df.columns for c in REQUIRED_COLS)

def _num(x): return pd.to_numeric(x, errors='coerce')

def safe_mean(x): return float(np.nanmean(_num(x)))
def safe_std(x):  return float(np.nanstd(_num(x), ddof=1))
def rate(x):      return float(np.nanmean(_num(x)))

def corr(a, b):
    """
    Helper function; Pearson correlation estimator
    """
    a, b = _num(a), _num(b)
    if len(a) < 2 or len(b) < 2 or a.isna().all() or b.isna().all():
        return np.nan
    try: return float(pd.Series(a).corr(pd.Series(b)))
    except Exception: return np.nan

PAIR_METRICS = [
    ('n_nodes',       lambda g: int(g.shape[0])),
    ('n_samples',     lambda g: int(g['sample_id'].nunique())),
    ('PC_soft_mean',  lambda g: safe_mean(g['PC_soft'])),
    ('PC_soft_std',   lambda g: safe_std(g['PC_soft'])),
    ('NS_soft_mean',  lambda g: safe_mean(g['NS_soft'])),
    ('NS_soft_std',   lambda g: safe_std(g['NS_soft'])),
    ('PS_mean',       lambda g: safe_mean(g['PS']) if 'PS' in g.columns else np.nan),
    ('PS_std',        lambda g: safe_std(g['PS']) if 'PS' in g.columns else np.nan),
    ('AKF_mean',      lambda g: safe_mean(g['SIMPLE_AKF'])),
    ('AKF_std',       lambda g: safe_std(g['SIMPLE_AKF'])),
    ('AKF_stab_mean', lambda g: safe_mean(g['SIMPLE_AKF_stab']) if 'SIMPLE_AKF_stab' in g.columns else np.nan),
    ('AKF_stab_std',  lambda g: safe_std(g['SIMPLE_AKF_stab']) if 'SIMPLE_AKF_stab' in g.columns else np.nan),
    ('PC_bin_rate',   lambda g: rate(g['PC_bin'])),
    ('NS_bin_rate',   lambda g: rate(g['NS_bin'])),
    ('uncertain_rate',lambda g: rate(g['uncertain'])),
    ('corr_PC_NS',    lambda g: corr(g['PC_soft'], g['NS_soft'])),
    ('corr_base_neg', lambda g: corr(g['base_logodds'], g['neg_logodds']) if 'base_logodds' in g.columns and 'neg_logodds' in g.columns else np.nan),
]

def aggregate_by(df: pd.DataFrame, by_cols: List[str]) -> pd.DataFrame:
    """
    Aggregates PAIR_METRICS grouped by specified column list
    
    @params: input DataFrame, list of columns to group by

    @returns: aggregated DataFrame
    """
    rows = []
    for keys, g in df.groupby(by_cols, dropna=False):
        keys = (keys,) if not isinstance(keys, tuple) else keys
        row = {col: key for col, key in zip(by_cols, keys)}
        for name, fn in PAIR_METRICS:
            try: row[name] = fn(g)
            except Exception: row[name] = np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    metric_order = [name for name, _ in PAIR_METRICS]
    return out[by_cols + metric_order].sort_values(by_cols).reset_index(drop=True)

def df_to_datatable_html(df: pd.DataFrame, table_id: str) -> str:
    """
    Helper function; converts Pandas DataFrame into an HTML table
    """
    return df.to_html(index=False, table_id=table_id, classes="display compact nowrap")

def build_report_html(out_html: str, by_pair: pd.DataFrame, by_pair_pol: pd.DataFrame):
    """
    Helper function; writes an HTML file with interactive table for the both summaries
    """
    css = """
    <style>
      body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 20px; }
      h1 { font-size: 22px; margin-bottom: 6px; }
      h2 { font-size: 18px; margin: 18px 0 8px; }
      .dt-buttons { margin-bottom: 8px; }
      table.dataTable tbody th, table.dataTable tbody td { white-space: nowrap; }
      .dataTables_wrapper .dataTables_scroll div.dataTables_scrollBody { border: 1px solid #ddd; }
    </style>
    """
    header = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>AKF-Lite Summary</title>
{css}
<link rel="stylesheet" href="https://cdn.datatables.net/1.13.8/css/jquery.dataTables.min.css">
<link rel="stylesheet" href="https://cdn.datatables.net/buttons/2.4.2/css/buttons.dataTables.min.css">
<link rel="stylesheet" href="https://cdn.datatables.net/fixedcolumns/4.3.0/css/fixedColumns.dataTables.min.css">
<link rel="stylesheet" href="https://cdn.datatables.net/colreorder/1.7.0/css/colReorder.dataTables.min.css">
<script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
<script src="https://cdn.datatables.net/1.13.8/js/jquery.dataTables.min.js"></script>
<script src="https://cdn.datatables.net/buttons/2.4.2/js/dataTables.buttons.min.js"></script>
<script src="https://cdn.datatables.net/buttons/2.4.2/js/buttons.colVis.min.js"></script>
<script src="https://cdn.datatables.net/fixedcolumns/4.3.0/js/dataTables.fixedColumns.min.js"></script>
<script src="https://cdn.datatables.net/colreorder/1.7.0/js/dataTables.colReorder.min.js"></script>
</head>
<body>
  <h1>AKF-Lite Summary</h1>
  <p>Aggregates across dataset–model pairs.</p>
"""
    t1_html = df_to_datatable_html(by_pair, "tbl_by_pair")
    t2_html = df_to_datatable_html(by_pair_pol, "tbl_by_pair_pol")

    body = f"""
  <h2>1) Summary by (dataset, model)</h2>
  <div style="width: 100%; overflow: hidden;">{t1_html}</div>

  <h2>2) Summary by (dataset, model, polarity)</h2>
  <div style="width: 100%; overflow: hidden;">{t2_html}</div>
"""
    footer = """
<script>
$(document).ready(function() {
  function initDT(id, leftCols) {
    $('#' + id).DataTable({
      scrollX: true, scrollY: '65vh', scrollCollapse: true,
      paging: true, pageLength: 25,
      lengthMenu: [[10,25,50,100,-1],[10,25,50,100,'All']],
      colReorder: true,
      fixedColumns: { leftColumns: leftCols },
      dom: 'Bfrtip', buttons: ['colvis', 'pageLength'],
      order: [[0, 'asc']]
    });
  }
  initDT('tbl_by_pair', 2);
  initDT('tbl_by_pair_pol', 3);
});
</script>
</body>
</html>
"""
    with open(out_html, 'w', encoding='utf-8') as f:
        f.write(header + body + footer)
    print(f"[SUMMARY] HTML report written to: {out_html}")

def main(args):
    os.makedirs(args.outdir, exist_ok=True)
    all_csvs = find_csvs(args.inputs)
    if not all_csvs:
        print("[SUMMARY][ERROR] No CSV files found under inputs.")
        sys.exit(1)

    frames = []
    for p in all_csvs:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"[SUMMARY][WARN] Could not read CSV: {p} ({e})")
            continue
        if not looks_like_akf_node_file(df):
            continue
        dataset, model = infer_pair_from_path(p)
        df['dataset'] = dataset
        df['model'] = model
        frames.append(df)
        print(f"[SUMMARY] Loaded: {p} -> dataset={dataset} model={model} rows={df.shape[0]}")

    if not frames:
        print("[SUMMARY][ERROR] No AKF-Lite node result CSVs detected with required columns.")
        sys.exit(2)

    nodes = pd.concat(frames, ignore_index=True)

    for col in ['PC_soft','NS_soft','PS','SIMPLE_AKF','SIMPLE_AKF_stab','PC_bin','NS_bin','uncertain']:
        if col in nodes.columns:
            nodes[col] = pd.to_numeric(nodes[col], errors='coerce')

    by_pair = aggregate_by(nodes, ['dataset','model'])
    by_pair_pol = aggregate_by(nodes, ['dataset','model','polarity'])

    p1 = os.path.join(args.outdir, 'summary_by_pair.csv')
    p2 = os.path.join(args.outdir, 'summary_by_pair_and_polarity.csv')
    by_pair.to_csv(p1, index=False)
    by_pair_pol.to_csv(p2, index=False)
    print(f"[SUMMARY] Wrote: {p1}")
    print(f"[SUMMARY] Wrote: {p2}")

    out_html = os.path.join(args.outdir, 'akf_lite_summary.html')
    build_report_html(out_html, by_pair, by_pair_pol)
    print("\n[SUMMARY] Done.")

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="Minimal AKF-Lite summarizer (pair & pair+polarity).")
    ap.add_argument('--inputs', nargs='+', required=True, help='Folders and/or CSV files (recursive for folders).')
    ap.add_argument('--outdir', required=True, help='Output directory for CSVs and HTML.')
    args = ap.parse_args()
    main(args)
