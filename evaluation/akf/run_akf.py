import os
os.environ["HF_HOME"] = "/vol/bitbucket/rc1124/hf_cache"

import json
import csv
import math
import re
import time
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
# MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
# MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

INPUT_JSON_PATH = "/vol/bitbucket/rc1124/MSc_Individual_Project/ExplainabilityInLLMs-MScThesis/results/generation/argLLM_generation/strategyclaim_llama/argllm_outputs_ollama.jsonl"

MAX_ITEMS = None
K_PARAPHRASES = 3
KAPPA = 1.0
TAU_ABSTAIN = 0.5

CSV_PATH = "/vol/bitbucket/rc1124/MSc_Individual_Project/ExplainabilityInLLMs-MScThesis/results/akf_test/results_strategyclaim_llama.csv"
HTML_PATH = "/vol/bitbucket/rc1124/MSc_Individual_Project/ExplainabilityInLLMs-MScThesis/results/akf_test/results_strategyclaim_llama.html"
FREEZE_LEFT_COLS = 4
DEFAULT_ORDER_COL = "SIMPLE_AKF"

os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

def build_html(df: pd.DataFrame, html_path: str, freeze_left_cols: int = 2, default_order_col: str = None):
    table_id = "akfLiteTable"
    table_html = df.to_html(index=False, table_id=table_id, classes="display compact nowrap")

    try:
        order_idx = df.columns.get_loc(default_order_col) if default_order_col else 0
    except Exception:
        order_idx = 0

    css = """
    <style>
      body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 20px; }
      h1 { font-size: 20px; margin-bottom: 12px; }
      .dt-buttons { margin-bottom: 8px; }
      table.dataTable tbody th, table.dataTable tbody td { white-space: nowrap; }
      .dataTables_wrapper .dataTables_scroll div.dataTables_scrollBody { border: 1px solid #ddd; }
    </style>
    """

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>AKF-Lite Node-level Results</title>
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
  <h1>AKF-Lite Node-level Results (interactive)</h1>
  <div style="width: 100%; overflow: hidden;">
    {table_html}
  </div>
  <script>
    $(document).ready(function() {{
      var table = $('#{table_id}').DataTable({{
        scrollX: true,
        scrollY: '70vh',
        scrollCollapse: true,
        paging: true,
        pageLength: 25,
        lengthMenu: [[10, 25, 50, 100, -1], [10, 25, 50, 100, 'All']],
        colReorder: true,
        fixedColumns: {{
          leftColumns: {freeze_left_cols}
        }},
        dom: 'Bfrtip',
        buttons: [
          'colvis',
          'pageLength'
        ],
        order: [[{order_idx}, 'desc']]
      }});
    }});
  </script>
</body>
</html>
    """
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[AKF-Lite] Interactive HTML report saved to: {html_path}")

def _candidate_variants(tokenizer):
    variants = {
        "TRUE": ["TRUE", " True", "TRUE.", " True."],
        "FALSE": ["FALSE", " False", "FALSE.", " False."]
    }
    def ids_for(s):
        enc = tokenizer(s, add_special_tokens=False, return_tensors=None)
        return enc["input_ids"]
    true_ids, false_ids = [], []
    for s in variants["TRUE"]:
        ids = ids_for(s)
        if ids:
            true_ids.append(ids)
    for s in variants["FALSE"]:
        ids = ids_for(s)
        if ids:
            false_ids.append(ids)
    return true_ids, false_ids

def _sequence_logprob(model, tokenizer, prompt: str, candidate_ids: List[int]) -> float:
    device = model.device
    prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    cand = torch.tensor([candidate_ids], dtype=torch.long, device=device)
    full = torch.cat([prompt_ids, cand], dim=1)

    with torch.no_grad():
        out_full = model(full)
        logits_full = out_full.logits

    start = prompt_ids.size(1)
    logits_cand = logits_full[:, start-1: start-1 + cand.size(1), :]
    targets = cand
    logprobs = torch.log_softmax(logits_cand, dim=-1)
    token_logps = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return float(token_logps.sum().item())

class HFBackend:
    INSTR = (
        "Decide if the statement is TRUE or FALSE. Output only 'TRUE' or 'FALSE'.\n\n"
        "Statement:\n{statement}\n\nAnswer:\n"
    )
    def __init__(self, model_name: str):
        """Load tokenizer and model on CPU/GPU and cache candidate variants."""
        print(f"[AKF-Lite] Loading model: {model_name}", flush=True)
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        dtype = torch.float16 if torch.cuda.is_available() else None
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        self.model.eval()
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self._true_ids, self._false_ids = _candidate_variants(self.tokenizer)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[AKF-Lite] Model loaded in {time.time() - t0:.2f}s. Device: {device}", flush=True)

    def true_false_logprobs(self, prompt: str) -> Dict[str, float]:
        """Return best log-probability for TRUE and FALSE variants for a prompt."""
        best_true = float("-inf")
        best_false = float("-inf")
        for tv in self._true_ids:
            lp = _sequence_logprob(self.model, self.tokenizer, prompt, tv)
            if lp > best_true:
                best_true = lp
        for fv in self._false_ids:
            lp = _sequence_logprob(self.model, self.tokenizer, prompt, fv)
            if lp > best_false:
                best_false = lp
        return {"TRUE": best_true, "FALSE": best_false}

    def belief_logodds(self, statement: str) -> float:
        """Compute TRUE vs FALSE log-odds for a statement."""
        prompt = self.INSTR.format(statement=statement)
        lps = self.true_false_logprobs(prompt)
        return float(lps["TRUE"] - lps["FALSE"])

SPACE_PAT = re.compile(r"\s+")
HEDGE_PAT = re.compile(
    r"\b(likely|possibly|may|might|could|perhaps|generally|often|somewhat|approximately)\b",
    re.IGNORECASE,
)

def normalize_proposition(text: str) -> str:
    t = HEDGE_PAT.sub("", text)
    t = SPACE_PAT.sub(" ", t).strip()
    t = t.rstrip(" .")
    return t

def simple_negation(p: str) -> str:
    if re.search(r"\bnot\b", p, re.IGNORECASE):
        return re.sub(r"\bnot\b", "", p, count=1, flags=re.IGNORECASE).replace("  ", " ").strip().rstrip(".") + "."
    m = re.search(r"\b(is|are|was|were|do|does|did|can|cannot|can't|won't|should|shouldn't|have|has|had|will)\b", p, re.IGNORECASE)
    if m:
        aux = m.group(1)
        start, end = m.span()
        if aux.lower() in ["cannot", "can't", "shouldn't", "won't"]:
            replace = {"cannot": "can", "can't": "can", "shouldn't": "should", "won't": "will"}
            new_aux = replace.get(aux.lower(), aux)
            return (p[:start] + new_aux + " not " + p[end:]).strip().rstrip(".") + "."
        return (p[:end] + " not " + p[end:]).strip().rstrip(".") + "."
    return f"It is not the case that {p}."

def paraphrases(p: str, k: int = 3) -> List[str]:
    outs = [p]
    outs.append(f"It is true that {p}.")
    if not re.search(r"\d", p):
        outs.append(f"According to common knowledge, {p}.")
    else:
        outs.append(f"Reportedly, {p}.")
    if " that " in p:
        outs.append(p.replace(" that ", " that, in effect, "))
    uniq = []
    for s in outs:
        s2 = SPACE_PAT.sub(" ", s).strip().rstrip(".") + "."
        if s2 not in uniq:
            uniq.append(s2)
    return uniq[:k]

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

@dataclass
class NodeEntry:
    name: str
    text: str
    polarity: str  # "support" or "attack"
    strength: Optional[float] = None
    initial_weight: Optional[float] = None

def extract_nodes(item: Dict[str, Any]) -> Tuple[List[NodeEntry], List[NodeEntry]]:
    bag = item.get("estimated", item.get("base"))["bag"]
    args_map = bag["arguments"]
    supports_edges = bag.get("supports", [])
    attacks_edges  = bag.get("attacks", [])

    supports, attacks = [], []
    for (src, dst) in supports_edges:
        n = args_map[src]
        supports.append(NodeEntry(
            name=src,
            text=n["argument"],
            polarity="support",
            strength=n.get("strength"),
            initial_weight=n.get("initial_weight"),
        ))
    for (src, dst) in attacks_edges:
        n = args_map[src]
        attacks.append(NodeEntry(
            name=src,
            text=n["argument"],
            polarity="attack",
            strength=n.get("strength"),
            initial_weight=n.get("initial_weight"),
        ))
    return supports, attacks

def iter_json_items(path: str):
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(4096)
        f.seek(0)

        if head.lstrip().startswith("["):
            obj = json.load(f)
            if isinstance(obj, list):
                for it in obj:
                    yield it
            else:
                yield obj
            return

        try:
            obj = json.load(f)
            if isinstance(obj, list):
                for it in obj:
                    yield it
            else:
                yield obj
            return
        except json.JSONDecodeError:
            f.seek(0)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

def count_items(path: str) -> Optional[int]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            head = f.read(4096)
            f.seek(0)
            if head.lstrip().startswith("["):
                obj = json.load(f)
                return len(obj) if isinstance(obj, list) else 1
            f.seek(0)
            return sum(1 for _ in f if _.strip())
    except Exception:
        return None

def node_metrics_lite(backend: HFBackend, s: str, kappa: float = 1.0, k_para: int = 3, tau_abstain: float = 0.5) -> Dict[str, Any]:
    l_base = backend.belief_logodds(s)
    y_base = 1 if l_base > 0 else -1
    c_base = sigmoid(abs(l_base))
    uncertain = 1 if abs(l_base) < tau_abstain else 0

    ps = paraphrases(s, k=k_para)
    ls = []
    y_matches = 0
    for si in ps:
        li = backend.belief_logodds(si)
        ls.append(li)
        if (1 if li > 0 else -1) == y_base:
            y_matches += 1

    PC_bin = (y_matches / len(ps)) if len(ps) > 0 else 0.0
    if len(ps) > 0:
        PC_soft = sum(sigmoid(kappa * y_base * li) for li in ls) / len(ls)
        PS = 1.0 / (1.0 + float(np.std(ls)))
        para_mean = float(np.mean(ls))
        para_std  = float(np.std(ls, ddof=1)) if len(ls) > 1 else 0.0
    else:
        PC_soft, PS = 0.0, 0.0
        para_mean, para_std = 0.0, 0.0

    neg = simple_negation(s)
    l_neg = backend.belief_logodds(neg)
    y_neg = 1 if l_neg > 0 else -1
    NS_bin = 1.0 if (y_neg == -y_base) else 0.0
    NS_soft = sigmoid(kappa * (-y_base) * l_neg)
    NS_soft_plus = NS_soft * (c_base + sigmoid(abs(l_neg))) / 2.0

    SIMPLE_AKF = 0.5 * PC_soft + 0.5 * NS_soft_plus
    SIMPLE_AKF_stab = 0.4 * PC_soft + 0.4 * NS_soft + 0.2 * PS

    return {
        "base_logodds": round(l_base, 6),
        "base_label": "TRUE" if y_base > 0 else "FALSE",
        "c_base": round(c_base, 6),
        "paraphrase_logodds_mean": round(para_mean, 6),
        "paraphrase_logodds_std": round(para_std, 6),
        "PC_bin": round(PC_bin, 6),
        "PC_soft": round(PC_soft, 6),
        "PS": round(PS, 6),
        "neg_text": neg if len(neg) < 200 else (neg[:200] + "..."),
        "neg_logodds": round(l_neg, 6),
        "neg_label": "TRUE" if y_neg > 0 else "FALSE",
        "NS_bin": round(NS_bin, 6),
        "NS_soft": round(NS_soft, 6),
        "NS_soft_plus": round(NS_soft_plus, 6),
        "uncertain": int(uncertain),
        "SIMPLE_AKF": round(SIMPLE_AKF, 6),
        "SIMPLE_AKF_stab": round(SIMPLE_AKF_stab, 6),
    }

print(f"[AKF-Lite] Starting AKF-Lite run", flush=True)
print(f"[AKF-Lite] Input file: {INPUT_JSON_PATH}", flush=True)

backend = HFBackend(MODEL_NAME)

maybe_total = count_items(INPUT_JSON_PATH)
if maybe_total is not None:
    print(f"[AKF-Lite] Detected ~{maybe_total} item(s) to process.", flush=True)
else:
    print(f"[AKF-Lite] Could not pre-count items (will stream).", flush=True)

results = []
count = 0
t_start = time.time()

for item in iter_json_items(INPUT_JSON_PATH):
    if MAX_ITEMS is not None and count >= MAX_ITEMS:
        break
    count += 1

    claim = item.get("claim", "")
    claim_label = str(item.get("label", "")).lower()
    sample_id = item.get("id", item.get("qid", f"row_{count}"))

    supports, attacks = extract_nodes(item)
    nodes = supports + attacks

    print("\n" + "="*88, flush=True)
    print(f"[AKF-Lite] Sample {count}{f' / ~{maybe_total}' if maybe_total else ''} | id={sample_id}", flush=True)
    print(f"[AKF-Lite] Claim: {claim}", flush=True)
    print(f"[AKF-Lite] Nodes: supports={len(supports)}, attacks={len(attacks)} (total={len(nodes)})", flush=True)

    node_start_time = time.time()
    n_uncertain = 0

    for i, n in enumerate(nodes, start=1):
        p = normalize_proposition(n.text)

        m = node_metrics_lite(
            backend=backend,
            s=p,
            kappa=KAPPA,
            k_para=K_PARAPHRASES,
            tau_abstain=TAU_ABSTAIN
        )

        if m["uncertain"] == 1:
            n_uncertain += 1

        results.append({
            "sample_id": sample_id,
            "claim": claim,
            "claim_label": claim_label,
            "node_name": n.name,
            "polarity": n.polarity,
            "node_strength": n.strength,
            "node_initial_weight": n.initial_weight,
            "proposition": p if len(p) < 200 else (p[:200] + "..."),
            **m,
        })

        print(
            f"[AKF-Lite]   Node {i}/{len(nodes)} | name={n.name} | "
            f"PC_soft={m['PC_soft']:.3f} | NS_soft={m['NS_soft']:.3f} | "
            f"AKF={m['SIMPLE_AKF']:.3f} | uncertain={m['uncertain']}",
            flush=True
        )

    elapsed_s = time.time() - node_start_time
    print(
        f"[AKF-Lite] Finished Sample {count} ({len(nodes)} node(s), {n_uncertain} uncertain) "
        f"in {elapsed_s:.2f}s",
        flush=True
    )

total_elapsed = time.time() - t_start
print("\n" + "="*88, flush=True)
print(f"[AKF-Lite] Processed {count} sample(s) in {total_elapsed/60:.2f} min", flush=True)

if results:
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"[AKF-Lite] Node-level results saved to: {CSV_PATH}", flush=True)

    df = pd.DataFrame(results)
    build_html(df, HTML_PATH, freeze_left_cols=FREEZE_LEFT_COLS, default_order_col=DEFAULT_ORDER_COL)
    print(f"[AKF-Lite] Done. CSV+HTML ready.", flush=True)
else:
    print("[AKF-Lite] No results generated.", flush=True)
