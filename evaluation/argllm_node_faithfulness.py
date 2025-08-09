# evaluation/faithfulness/arg_node_faithfulness.py
"""
LOO (Leave-One-Argument-Out) node faithfulness metrics for ArgLLM QBAFs.

This module:
- Parses saved ArgLLM "bag" dicts
- Recomputes claim strength using a DF-QuAD-like iterative semantics
- Performs Leave-One-Argument-Out ablations per node (excluding 'db0')
- Computes node influence aligned with the node's net role (pro vs con) w.r.t. db0
- Returns per-node metrics & per-sample summary

NOTE: We implement DF-QuAD computation directly on the saved bags (no external engine),
      with a simple fixed-point iteration. Works for trees/DAGs and small cyclic graphs.

Bag schema (from t_xxx.to_dict()):
{
  "arguments": {
      "<id>": {"name": "<id>", "argument": "...", "initial_weight": float, "strength": float, ...},
      ...
  },
  "attacks": [["attacker_id", "attacked_id"], ...],
  "supports": [["supporter_id", "supported_id"], ...]
}
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Set, Optional, Any
import copy
import math

def _get_args(bag: Dict[str, Any]) -> Set[str]:
    return set((bag.get("arguments") or {}).keys())

def _edges(bag: Dict[str, Any]) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    attacks = [(a, b) for a, b in (bag.get("attacks") or [])]
    supports = [(a, b) for a, b in (bag.get("supports") or [])]
    return attacks, supports

def _incoming_map(edges: List[Tuple[str, str]]) -> Dict[str, List[str]]:
    m: Dict[str, List[str]] = {}
    for s, t in edges:
        m.setdefault(t, []).append(s)
    return m

def _outgoing_map(edges: List[Tuple[str, str]]) -> Dict[str, List[str]]:
    m: Dict[str, List[str]] = {}
    for s, t in edges:
        m.setdefault(s, []).append(t)
    return m

def _tau(bag: Dict[str, Any], a: str) -> float:
    meta = (bag.get("arguments") or {}).get(a, {}) or {}
    if "initial_weight" in meta and isinstance(meta["initial_weight"], (float, int)):
        return float(meta["initial_weight"])
    if "strength" in meta and isinstance(meta["strength"], (float, int)):
        return float(meta["strength"])
    return 0.5

def _F(vs: List[float]) -> float:
    if not vs:
        return 0.0
    prod = 1.0
    for v in vs:
        prod *= (1.0 - max(0.0, min(1.0, v)))
    return 1.0 - prod

def _C(v0: float, va: float, vs: float) -> float:
    v0 = max(0.0, min(1.0, v0))
    import math as _math
    if _math.isclose(va, vs, rel_tol=1e-12, abs_tol=1e-12):
        return v0
    if va > vs:
        return max(0.0, min(1.0, v0 - (v0 * abs(vs - va))))
    else:
        return max(0.0, min(1.0, v0 + ((1.0 - v0) * abs(vs - va))))

def compute_dfquad_strengths(bag: Dict[str, Any],
                             max_iter: int = 200,
                             tol: float = 1e-8) -> Dict[str, float]:
    args = list(_get_args(bag))
    attacks, supports = _edges(bag)
    att_in = _incoming_map(attacks)
    sup_in = _incoming_map(supports)
    strengths = {a: float(_tau(bag, a)) for a in args}

    for _ in range(max_iter):
        max_delta = 0.0
        new_strengths = {}
        for a in args:
            v0 = _tau(bag, a)
            atk_vals = [strengths[src] for src in att_in.get(a, [])]
            sup_vals = [strengths[src] for src in sup_in.get(a, [])]
            va = _F(atk_vals)
            vs = _F(sup_vals)
            val = _C(v0, va, vs)
            new_strengths[a] = val
            max_delta = max(max_delta, abs(val - strengths[a]))
        strengths = new_strengths
        if max_delta < tol:
            break
    return strengths

def recompute_claim_strength(bag: Dict[str, Any], claim_id: str = "db0") -> float:
    strengths = compute_dfquad_strengths(bag)
    return float(strengths.get(claim_id, 0.5))

def _build_mixed_graph(bag: Dict[str, Any]) -> Dict[str, List[Tuple[str, str]]]:
    G: Dict[str, List[Tuple[str, str]]] = {}
    attacks, supports = _edges(bag)
    for s, t in attacks:
        G.setdefault(s, []).append((t, "att"))
    for s, t in supports:
        G.setdefault(s, []).append((t, "sup"))
    for a in _get_args(bag):
        G.setdefault(a, [])
    return G

def role_parity_to_claim(bag: Dict[str, Any], node_id: str, claim_id: str = "db0") -> Optional[int]:
    if node_id == claim_id:
        return 0
    G = _build_mixed_graph(bag)
    from collections import deque
    dq = deque()
    seen = set()
    dq.append((node_id, 0))
    seen.add((node_id, 0))
    while dq:
        u, par = dq.popleft()
        for v, lab in G.get(u, []):
            new_par = par + (1 if lab == "att" else 0)
            state = (v, new_par % 2)
            if state in seen:
                continue
            if v == claim_id:
                return new_par % 2
            seen.add(state)
            dq.append((v, new_par % 2))
    return None

def node_role(bag: Dict[str, Any], node_id: str, claim_id: str = "db0") -> str:
    par = role_parity_to_claim(bag, node_id, claim_id=claim_id)
    if par is None:
        return "none"
    return "pro" if par == 0 else "con"

def remove_node(bag: Dict[str, Any], node_id: str) -> Dict[str, Any]:
    bag2 = copy.deepcopy(bag)
    if "arguments" in bag2 and node_id in bag2["arguments"]:
        del bag2["arguments"][node_id]
    def _keep(edge):
        return edge[0] != node_id and edge[1] != node_id
    bag2["attacks"] = [e for e in (bag2.get("attacks") or []) if _keep(e)]
    bag2["supports"] = [e for e in (bag2.get("supports") or []) if _keep(e)]
    return bag2

def evaluate_bag_loo(bag: Dict[str, Any],
                     claim_id: str = "db0",
                     threshold: float = 0.5) -> Dict[str, Any]:
    strengths_full = compute_dfquad_strengths(bag)
    s0 = float(strengths_full.get(claim_id, 0.5))
    y0 = int(s0 >= threshold)

    out = {
        "baseline_strength": s0,
        "baseline_label": y0,
        "nodes": {}
    }

    for a in _get_args(bag):
        if a == claim_id:
            continue
        role = node_role(bag, a, claim_id=claim_id)
        tau_a = _tau(bag, a)
        sigma_a = float(strengths_full.get(a, tau_a))

        bag_minus_a = remove_node(bag, a)
        s_removed = recompute_claim_strength(bag_minus_a, claim_id=claim_id)
        y_removed = int(s_removed >= threshold)

        raw_delta = s0 - s_removed

        if role == "pro":
            influence_aligned = s0 - s_removed
        elif role == "con":
            influence_aligned = s_removed - s0
        else:
            influence_aligned = 0.0

        out["nodes"][a] = {
            "role": role,
            "tau": float(tau_a),
            "baseline_strength": sigma_a,
            "loo_claim_strength": s_removed,
            "claim_strength_delta": raw_delta,
            "node_influence_aligned": influence_aligned,
            "label_flip": int(y0 != y_removed),
            "loo_label": y_removed,  # NEW: label after LOO ablation
        }

    return out
