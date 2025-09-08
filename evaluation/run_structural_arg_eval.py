import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from evaluation.Evaluating_Explanations.src.metrics.argumentative_metrics import (
    compute_circularity,
    compute_dialectical_acceptability
)

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
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
    elif p.suffix.lower() == ".json":
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for rec in data:
                yield rec
        elif isinstance(data, dict):
            yield data
        else:
            raise ValueError("Unsupported JSON structure: expected list or dict at top-level.")
    else:
        raise ValueError(f"Unsupported file extension: {p.suffix}")


def bag_to_graph(bag: Dict[str, Any]) -> Tuple[List[str], List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Converts an argLLM bag into a argumentative graphical structure: (args, attacks, supports); 
    supports are returned as (supported, supporter) pairs as per metric functions
    
    @params: dictionary with keys 'arguments', 'attacks', 'supports'
    
    @returns: argument list, attack relations [(attacker, attacked)], support relations [(supported, supporter)]
    """
    arg_dict = bag.get("bag", {}).get("arguments", {}) if "arguments" not in bag else bag
    if "arguments" not in bag:
        arguments = arg_dict
    else:
        arguments = bag["bag"]["arguments"]

    args = list(arguments.keys())

    attacks_raw = bag.get("bag", {}).get("attacks", []) if "attacks" not in bag else bag.get("attacks", [])
    attacks: List[Tuple[str, str]] = []
    for pair in attacks_raw:
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            attacker, attacked = pair
            attacks.append((attacker, attacked))

    supports_raw = bag.get("bag", {}).get("supports", []) if "supports" not in bag else bag.get("supports", [])
    supports: List[Tuple[str, str]] = []
    for pair in supports_raw:
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            supporter, supported = pair
            supports.append((supported, supporter))

    return args, attacks, supports


def deduce_yhat_args(
    args: List[str],
    attacks: List[Tuple[str, str]],
    supports: List[Tuple[str, str]],
    strategy: str = "auto",
) -> List[str]:
    """
    Chooses y_hat arguments for acceptability

    @params: list of argument ids, attack relations, support relations, strategy;
              strategy in {"db0", "root", "auto", "all"}
              - "db0": if "db0" in args, return ["db0"], else []
              - "root": return the argument with highest indegree (attacks+supports)
              - "auto": if "db0" in args, return ["db0"], else root argument
              - "all": return all arguments

    @returns: list of argument ids
    """
    if strategy not in {"db0", "root", "auto", "all"}:
        strategy = "auto"

    if strategy in {"db0", "auto"} and "db0" in args:
        return ["db0"]

    if strategy == "all":
        return list(args)

    indeg = {a: 0 for a in args}
    for attacker, attacked in attacks:
        if attacked in indeg:
            indeg[attacked] += 1
    for supported, supporter in supports:
        if supported in indeg:
            indeg[supported] += 1

    root = max(indeg.items(), key=lambda kv: kv[1])[0] if indeg else (args[0] if args else None)
    return [root] if root else []


def evaluate_bag(
    bag_container: Dict[str, Any],
    yhat_strategy: str = "auto",
) -> Dict[str, Any]:
    """
    Computes structural metrics for one bag

    @params: bag dictionaries

    @returns: dictionary with counts, chosen y_hat args, circularity, acceptability values
    """
    args, attacks, supports = bag_to_graph(bag_container)
    yhat_args = deduce_yhat_args(args, attacks, supports, strategy=yhat_strategy)

    circ = compute_circularity(args, attacks, supports)
    acc = compute_dialectical_acceptability(args, attacks, yhat_args)

    return {
        "num_args": len(args),
        "num_attacks": len(attacks),
        "num_supports": len(supports),
        "yhat_args": "|".join(yhat_args),
        "circularity": float(circ),
        "acceptability": float(acc),
    }


def process_file(
    input_path: Union[str, Path],
    out_csv: Optional[Union[str, Path]] = None,
    bags: Tuple[str, ...] = ("base", "estimated"),
    yhat_strategy: str = "auto",
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    Iterates over records, computes metrics per bag, and return rows

    @params: input path to JSON/JSONL, optional CSV output path,
             tuple of bag names to include, y_hat strategy, verbose flag
             
    @returns: list of row dicts; writes CSV if output path given
    """
    rows: List[Dict[str, Any]] = []
    total = 0

    for rec in read_json_or_jsonl(input_path):
        total += 1
        rec_id = rec.get("id")
        question = rec.get("question")
        claim = rec.get("claim")
        label = rec.get("label")

        row: Dict[str, Any] = {
            "id": rec_id,
            "question": question,
            "claim": claim,
            "gold_label": label,
        }

        for bag_name in bags:
            if bag_name not in rec:
                continue
            bag_block = rec[bag_name]
            pred = bag_block.get("prediction", None)
            row[f"{bag_name}_prediction"] = pred

            metrics = evaluate_bag(bag_block, yhat_strategy=yhat_strategy)
            row[f"{bag_name}_num_args"] = metrics["num_args"]
            row[f"{bag_name}_num_attacks"] = metrics["num_attacks"]
            row[f"{bag_name}_num_supports"] = metrics["num_supports"]
            row[f"{bag_name}_yhat_args"] = metrics["yhat_args"]
            row[f"{bag_name}_circularity"] = metrics["circularity"]
            row[f"{bag_name}_acceptability"] = metrics["acceptability"]

        rows.append(row)

        if verbose and total % 25 == 0:
            print(f"[info] processed {total} samples...")

    if verbose:
        print(f"[done] processed {total} samples total.")

    if out_csv:
        outp = Path(out_csv)
        outp.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({k for r in rows for k in r.keys()})
        with outp.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        if verbose:
            print(f"[save] wrote CSV -> {outp}")

    return rows


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Structural evaluation of argLLM explanations.")
    ap.add_argument("--input", required=True, help="Path to JSON/JSONL file with argLLM outputs.")
    ap.add_argument("--out_csv", default=None, help="Optional path to write a CSV summary.")
    ap.add_argument(
        "--bags",
        nargs="+",
        choices=["base", "estimated"],
        default=["base", "estimated"],
        help="Which bags to evaluate.",
    )
    ap.add_argument(
        "--yhat",
        choices=["auto", "db0", "root", "all"],
        default="auto",
        help="How to choose y_hat arguments for acceptability.",
    )
    ap.add_argument("--verbose", action="store_true", help="Log progress.")
    return ap.parse_args()


def main():
    args = parse_args()
    process_file(
        input_path=args.input,
        out_csv=args.out_csv,
        bags=tuple(args.bags),
        yhat_strategy=args.yhat,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
