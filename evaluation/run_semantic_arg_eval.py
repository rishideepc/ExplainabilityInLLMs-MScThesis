import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
from sentence_transformers import SentenceTransformer, util


def init_logging(level: str = "INFO", log_file: Optional[Union[str, Path]] = None):
    """
    Helper function; initializes console/file logging

    @params: log level string, optional path to write logs
    """
    level_map = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
    }
    log_level = level_map.get(level.upper(), logging.INFO)

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_file = str(log_file)
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="w", encoding="utf-8"))

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        handlers=handlers,
    )
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def gpu_snapshot(tag: str = "") -> str:
    """
    Helper function; reports a brief GPU memory snapshot

    @params: optional label to append

    @returns: GPU memory snapshot string
    """
    if not torch.cuda.is_available():
        return f"[GPU] not available {tag}"
    idx = torch.cuda.current_device()
    name = torch.cuda.get_device_name(idx)
    mem_alloc = torch.cuda.memory_allocated(idx) / (1024**3)
    mem_reserved = torch.cuda.memory_reserved(idx) / (1024**3)
    return f"[GPU] dev={idx}({name}) alloc={mem_alloc:.2f}GB reserved={mem_reserved:.2f}GB {tag}"


log = logging.getLogger("arg-eval-sem")


def read_json_or_jsonl(path: Union[str, Path]) -> Iterable[Dict[str, Any]]:
    """
    Helper function; reads records from a JSON or JSONL file

    @params: input file path

    @returns (yield): parsed JSON objects
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input not found: {p}")
    if p.suffix.lower() == ".jsonl":
        with p.open("r", encoding="utf-8") as f:
            for ln, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    log.error(f"JSONL parse error at line {ln}: {e}")
                    raise
    elif p.suffix.lower() == ".json":
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for rec in data:
                yield rec
        elif isinstance(data, dict):
            yield data
        else:
            raise ValueError("Unsupported JSON structure at top-level.")
    else:
        raise ValueError(f"Unsupported file extension: {p.suffix}")


def extract_bag(bag_container: Dict[str, Any]) -> Tuple[Dict[str, str], List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Helper function; extracts argument texts and edge (relation) lists from an argLLM bag

    @params: Dictionary that is either the bag or contains 'bag' of arguments

    @returns: arguments, attack & support relations
    """
    if "arguments" in bag_container:
        inner = bag_container
    else:
        inner = bag_container.get("bag", {})

    arguments = inner.get("arguments", {}) or {}
    attacks = inner.get("attacks", []) or []
    supports = inner.get("supports", []) or []

    arg_texts: Dict[str, str] = {}
    for arg_id, payload in arguments.items():
        if isinstance(payload, dict) and "argument" in payload:
            arg_texts[arg_id] = (payload.get("argument") or "").strip()
        else:
            arg_texts[arg_id] = str(payload).strip()

    def _pairs(pairs):
        out = []
        for pr in pairs:
            if isinstance(pr, (list, tuple)) and len(pr) == 2:
                a, b = pr
                out.append((str(a), str(b)))
        return out

    return arg_texts, _pairs(attacks), _pairs(supports)


def deduce_yhat(args: List[str],
                attacks: List[Tuple[str, str]],
                supports: List[Tuple[str, str]],
                strategy: str = "auto") -> Optional[str]:
    """
    Infers root node from graph structure

    @params: argument IDs, attack edges, support edges (s,b), strategy('auto'|'db0'|'root')

    @returns: selected root ID or None
    """
    if not args:
        return None
    if strategy not in {"db0", "root", "auto"}:
        strategy = "auto"
    if strategy in {"db0", "auto"} and "db0" in args:
        return "db0"
    indeg = {a: 0 for a in args}
    for a, b in attacks:
        if b in indeg: indeg[b] += 1
    for s, b in supports:
        if b in indeg: indeg[b] += 1
    return max(indeg, key=lambda k: indeg[k]) if indeg else args[0]


def get_neighbors_of_root(root: str,
                          attacks: List[Tuple[str, str]],
                          supports: List[Tuple[str, str]]) -> Tuple[List[str], List[str]]:
    """
    Collects attackers and supporters of root node

    @params: root ID, attack edges, support edges

    @returns: (attackers, supporters) as lists of IDs
    """
    attackers = [a for (a, b) in attacks if b == root]
    supporters = [s for (s, b) in supports if b == root]
    return attackers, supporters


def cosine(a, b) -> float:
    """
    Estimates cosine similarity between two sentence-transformer embeddings

    @params: two embedding tensors

    @returns: cosine similarity value (float)
    """
    return float(util.cos_sim(a, b).item())


def compute_semantic_circularity(
    root_id: str,
    arg_texts: Dict[str, str],
    attackers: List[str],
    supporters: List[str],
    emb_cache: Dict[str, torch.Tensor],
    tau_circ: float,
) -> Tuple[float, float, List[float]]:
    """
    Computes Semantic Circularity metric as fraction of supporters sufficiently similar to the original claim

    @params: root argument ID, arguments, attacker IDs, supporter IDs, embedding cache, similarity threshold

    @returns: (metric score, max similarity, all supporter similarities)
    """
    root_txt = arg_texts.get(root_id, "")
    root_emb = emb_cache.get(root_txt)
    sims = []
    for s in supporters:
        s_txt = arg_texts.get(s, "")
        s_emb = emb_cache.get(s_txt)
        if root_emb is None or s_emb is None:
            continue
        sims.append(cosine(root_emb, s_emb))
    score = 0.0 if not supporters else sum(v >= tau_circ for v in sims) / max(1, len(sims))
    max_sim = max(sims) if sims else 0.0
    return score, max_sim, sims


def compute_semantic_acceptability(
    root_id: str,
    arg_texts: Dict[str, str],
    attackers: List[str],
    supporters: List[str],
    emb_cache: Dict[str, torch.Tensor],
    tau_neutral: float,
) -> Tuple[float, List[bool], List[float], List[float], List[float]]:
    """
    Computes Semantic Acceptability metric as fraction of attackers neutralized by similarity to root or supporters

    @params: root argument ID, arguments, attacker IDs, supporter IDs, embedding cache, similarity threshold

    @returns: (metric score, number of flags per attacker, best similarity, similarity of attacker to root, max similarity between supporter and attacker)
    """
    if not attackers:
        return 1.0, [], [], [], []
    root_txt = arg_texts.get(root_id, "")
    root_emb = emb_cache.get(root_txt)
    supp_embs = [emb_cache.get(arg_texts.get(s, ""), None) for s in supporters]
    supp_embs = [e for e in supp_embs if e is not None]

    flags = []
    best_sims = []
    sims_root = []
    sims_supp_max = []

    for a in attackers:
        a_txt = arg_texts.get(a, "")
        a_emb = emb_cache.get(a_txt)
        if a_emb is None:
            flags.append(False); best_sims.append(0.0); sims_root.append(0.0); sims_supp_max.append(0.0)
            continue
        sim_r = cosine(a_emb, root_emb) if root_emb is not None else 0.0
        sims_root.append(sim_r)
        sim_s = max((cosine(a_emb, se) for se in supp_embs), default=0.0)
        sims_supp_max.append(sim_s)
        best = max(sim_r, sim_s)
        best_sims.append(best)
        flags.append(best >= tau_neutral)

    score = sum(flags) / len(flags)
    return score, flags, best_sims, sims_root, sims_supp_max


def collect_unique_texts(
    input_path: Union[str, Path],
    bags: Tuple[str, ...],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Scans records and collects unique argument texts for selected bags

    @params: input JSON/JSONL path, tuple of bag names to include

    @returns: records list, unique arguments list
    """
    records = []
    seen = set()
    uniq_texts: List[str] = []

    count = 0
    for rec in read_json_or_jsonl(input_path):
        records.append(rec)
        count += 1
        for bag_name in bags:
            if bag_name not in rec:
                continue
            bag_block = rec[bag_name]
            arg_texts, _, _ = extract_bag(bag_block)
            for txt in arg_texts.values():
                if txt not in seen:
                    seen.add(txt)
                    uniq_texts.append(txt)
        if count % 100 == 0:
            log.info(f"[scan] cached {count} records so far; unique texts={len(uniq_texts)}")

    log.info(f"[scan] PASS COMPLETE: total_records={len(records)} unique_texts={len(uniq_texts)}")
    return records, uniq_texts


def embed_all_unique(
    texts: List[str],
    model_name: str,
    batch_size: int,
) -> Dict[str, torch.Tensor]:
    """
    Embeds all unique texts once on GPU/CPU

    @params: unique strings to embed, Sentence-Transformer model, batch size

    @returns: Dictionary mapping text to normalized embedding tensor
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"[embed] model={model_name} device={device}")
    if torch.cuda.is_available():
        log.info(gpu_snapshot("[before-encode]"))

    model = SentenceTransformer(model_name, device=device)
    with torch.inference_mode():
        embs = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            device=device,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
    if torch.cuda.is_available():
        log.info(gpu_snapshot("[after-encode]"))

    log.info(f"[embed] generated embeddings: {len(texts)}")
    return {t: e for t, e in zip(texts, embs)}


def process_file(
    input_path: Union[str, Path],
    out_csv: Union[str, Path],
    bags: Tuple[str, ...] = ("base", "estimated"),
    yhat_strategy: str = "auto",
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    tau_circ: float = 0.80,
    tau_neutral: float = 0.70,
    batch_size: int = 128,
    verbose: bool = False,
    heartbeat: int = 50,
) -> List[Dict[str, Any]]:
    """Runs semantic circularity/acceptability over records and writes CSV


    @params: input JSON/JSONL path, output CSV path, bags to include, root selection strategy, 
             embedding model name, circularity threshold, neutralization threshold,
             embedding batch size, verbosity flag, heartbeat interval

    @returns: list of per-record result dictionaries
    """
    log.info("======== PASS 1: SCAN & COLLECT UNIQUE TEXTS ========")
    records, uniq_texts = collect_unique_texts(input_path, bags=bags)
    log.info(f"[config] bags={bags} yhat={yhat_strategy} tau_circ={tau_circ:.2f} tau_neutral={tau_neutral:.2f} batch_size={batch_size}")
    if torch.cuda.is_available():
        log.info(gpu_snapshot("[after-scan]"))

    log.info("======== PASS 2: EMBED ALL UNIQUE TEXTS ========")
    emb_cache = embed_all_unique(uniq_texts, model_name=model_name, batch_size=batch_size)

    log.info("======== PASS 3: SCORE RECORDS ========")
    rows: List[Dict[str, Any]] = []
    total = 0

    for rec in records:
        total += 1
        rid = rec.get("id")
        question = rec.get("question")
        claim_txt = rec.get("claim")
        gold = rec.get("label")

        row: Dict[str, Any] = {
            "id": rid,
            "question": question,
            "claim": claim_txt,
            "gold_label": gold,
        }

        for bag_name in bags:
            if bag_name not in rec:
                log.debug(f"[skip] record id={rid} missing bag={bag_name}")
                continue

            bag_block = rec[bag_name]
            pred = bag_block.get("prediction", None)
            row[f"{bag_name}_prediction"] = pred

            arg_texts, attacks, supports = extract_bag(bag_block)
            arg_ids = list(arg_texts.keys())
            root = deduce_yhat(arg_ids, attacks, supports, strategy=yhat_strategy)
            row[f"{bag_name}_yhat"] = root

            attackers, supporters = get_neighbors_of_root(root, attacks, supports)
            row[f"{bag_name}_num_args"] = len(arg_ids)
            row[f"{bag_name}_num_attackers"] = len(attackers)
            row[f"{bag_name}_num_supporters"] = len(supporters)

            sem_circ, sem_circ_max, sem_circ_sims = compute_semantic_circularity(
                root_id=root,
                arg_texts=arg_texts,
                attackers=attackers,
                supporters=supporters,
                emb_cache=emb_cache,
                tau_circ=tau_circ,
            )

            sem_acc, flags, best_sims, sims_root, sims_supp = compute_semantic_acceptability(
                root_id=root,
                arg_texts=arg_texts,
                attackers=attackers,
                supporters=supporters,
                emb_cache=emb_cache,
                tau_neutral=tau_neutral,
            )

            row[f"{bag_name}_sem_circularity"] = sem_circ
            row[f"{bag_name}_sem_circ_max_sim"] = sem_circ_max
            row[f"{bag_name}_sem_circ_all_sims"] = "|".join(f"{v:.3f}" for v in sem_circ_sims)

            row[f"{bag_name}_sem_acceptability"] = sem_acc
            row[f"{bag_name}_sem_acc_neutralized_flags"] = "|".join("1" if f else "0" for f in flags)
            row[f"{bag_name}_sem_acc_attacker_best_sim"] = "|".join(f"{v:.3f}" for v in best_sims)
            row[f"{bag_name}_sem_acc_attacker_to_root_sims"] = "|".join(f"{v:.3f}" for v in sims_root)
            row[f"{bag_name}_sem_acc_attacker_to_support_max_sims"] = "|".join(f"{v:.3f}" for v in sims_supp)

            log.debug(
                f"[rec {rid} | {bag_name}] "
                f"args={len(arg_ids)} atk={len(attackers)} sup={len(supporters)} yhat={root} "
                f"sem_circ={sem_circ:.3f} (max={sem_circ_max:.3f}) sem_acc={sem_acc:.3f}"
            )

        rows.append(row)

        if total % heartbeat == 0:
            log.info(f"[heartbeat] processed={total} {gpu_snapshot() if torch.cuda.is_available() else ''}")

    log.info(f"[score] PASS COMPLETE: total_scored={total}")
    if torch.cuda.is_available():
        log.info(gpu_snapshot("[after-score]"))

    log.info("======== WRITE CSV ========")
    outp = Path(out_csv)
    outp.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with outp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"[save] wrote CSV -> {outp.resolve()}")

    log.info("======== DONE ========")
    return rows


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="FAST semantic evaluation of D=1 argLLM explanations (extensive logging).")
    ap.add_argument("--input", required=True, help="Path to JSON/JSONL argLLM outputs.")
    ap.add_argument("--out_csv", required=True, help="Where to write CSV results.")
    ap.add_argument("--bags", nargs="+", choices=["base", "estimated"], default=["base", "estimated"])
    ap.add_argument("--yhat", choices=["auto", "db0", "root"], default="auto")
    ap.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--tau-circ", type=float, default=0.80)
    ap.add_argument("--tau-neutral", type=float, default=0.70)
    ap.add_argument("--batch-size", type=int, default=128, help="Embedding batch size for GPU encode.")
    ap.add_argument("--log-level", default="INFO", choices=["CRITICAL","ERROR","WARNING","INFO","DEBUG"],
                    help="Logger verbosity.")
    ap.add_argument("--log-file", default=None, help="Optional path to write a log file.")
    ap.add_argument("--heartbeat", type=int, default=50, help="Emit progress log every K records.")
    return ap.parse_args()


def main():
    args = parse_args()
    init_logging(level=args.log_level, log_file=args.log_file)
    log.info("======== CONFIG ========")
    log.info(f"input={args.input}")
    log.info(f"out_csv={args.out_csv}")
    log.info(f"bags={args.bags} yhat={args.yhat}")
    log.info(f"model={args.model} tau_circ={args.tau_circ} tau_neutral={args.tau_neutral}")
    log.info(f"batch_size={args.batch_size} heartbeat={args.heartbeat}")
    if torch.cuda.is_available():
        log.info(gpu_snapshot("[startup]"))
    else:
        log.info("[GPU] CUDA not available — running on CPU.")

    process_file(
        input_path=args.input,
        out_csv=args.out_csv,
        bags=tuple(args.bags),
        yhat_strategy=args.yhat,
        model_name=args.model,
        tau_circ=args.tau_circ,
        tau_neutral=args.tau_neutral,
        batch_size=args.batch_size,
        verbose=False,
        heartbeat=args.heartbeat,
    )


if __name__ == "__main__":
    main()
