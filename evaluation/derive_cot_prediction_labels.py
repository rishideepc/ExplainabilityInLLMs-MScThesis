import json, re, sys

IN_FILE  = "/vol/bitbucket/rc1124/MSc_Individual_Project/ExplainabilityInLLMs-MScThesis/results/generation/commonsenseclaim_mistral/cot_outputs_claims.jsonl"
OUT_FILE = "/vol/bitbucket/rc1124/MSc_Individual_Project/ExplainabilityInLLMs-MScThesis/results/generation/commonsenseclaim_mistral/cot_outputs_claims_with_preds.jsonl"

TRUE_TOKENS  = {"true", "correct", "accurate", "factual", "right", "valid", "supported"}
FALSE_TOKENS = {"false", "incorrect", "inaccurate", "not true", "myth", "wrong", "invalid", "unsupported"}

STRONG_FALSE_PATTERNS = [
    r"\buntrue\b",
    r"\bnot\s+true\b",
    r"\bis\s+not\s+true\b",
    r"\bappears\s+not\s+true\b",
    r"\bseems\s+not\s+true\b",
]

DOUBLE_NEG_TRUE_PATTERNS = [
    r"\bnot\s+false\b",
    r"\bnot\s+incorrect\b",
    r"\bnot\s+wrong\b",
    r"\bnot\s+invalid\b",
    r"\bnot\s+unsupported\b",
]

CONCLUSION_RE = re.compile(r'(?i)^\s*Conclusion\s*:\s*(.+)$', flags=re.MULTILINE)


def extract_conclusion(text: str) -> tuple[str, bool]:
    """
    Extracts the conclusion string 
    
    @params: full CoT text 
    
    @returns: extracted conclusion, flag indicating if explicit conclusion was found
    """
    if not text:
        return "", False

    m = CONCLUSION_RE.search(text)
    if m:
        return m.group(1).strip(), True

    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return (sentences[-1].strip() if sentences else ""), False


def _find_all_indices(haystack: str, needles: set[str]) -> list[tuple[int, str]]:
    """
    Finds all occurrences of any token 
    
    @params: text, set of tokens 
    
    @returns: list of start index and token
    """
    hits = []
    low = haystack.lower()
    for token in needles:
        start = 0
        token_l = token.lower()
        while True:
            i = low.find(token_l, start)
            if i == -1:
                break
            hits.append((i, token))
            start = i + len(token_l)
    return hits


def _match_any_patterns(haystack: str, patterns: list[str]) -> bool:
    """
    Regex-checks any pattern 
    
    @params: text, list of regex strings 
    
    @returns: bool
    """
    low = haystack.lower()
    return any(re.search(p, low) for p in patterns)


def normalize_pred_from_conclusion(conclusion: str) -> tuple[str | None, dict]:
    """
    Derives 'true'/'false'/None labels from a conclusion
    
    @params: conclusion text 
    
    @returns: (conclusion label or None)
    """
    info = {
        "had_true_cues": False,
        "had_false_cues": False,
        "both_polarities": False,
        "chosen_rule": None,
    }
    if not conclusion:
        return None, info

    lc = conclusion.lower()
    true_hits  = _find_all_indices(lc, TRUE_TOKENS)
    false_hits = _find_all_indices(lc, FALSE_TOKENS)

    info["had_true_cues"] = len(true_hits) > 0
    info["had_false_cues"] = len(false_hits) > 0

    if not true_hits and not false_hits:
        if re.search(r'\b(is|appears|seems)\s+not\s+true\b', lc):
            info["chosen_rule"] = "pattern_neg_true"
            return "false", info
        if re.search(r'\b(is|appears|seems)\s+true\b', lc):
            info["chosen_rule"] = "pattern_is_true"
            return "true", info
        return None, info

    if true_hits and not false_hits:
        info["chosen_rule"] = "single_true"
        return "true", info
    if false_hits and not true_hits:
        info["chosen_rule"] = "single_false"
        return "false", info

    info["both_polarities"] = True

    if _match_any_patterns(lc, STRONG_FALSE_PATTERNS):
        info["chosen_rule"] = "strong_false"
        return "false", info

    if _match_any_patterns(lc, DOUBLE_NEG_TRUE_PATTERNS):
        info["chosen_rule"] = "double_neg_true"
        return "true", info

    last_true_pos  = max(p for p, _ in true_hits)
    last_false_pos = max(p for p, _ in false_hits)
    if last_false_pos > last_true_pos:
        info["chosen_rule"] = "last_mention_false"
        return "false", info
    else:
        info["chosen_rule"] = "last_mention_true"
        return "true", info


def main():
    """
    Augments JSONL with derived CoT predictions
    """
    n = 0
    found_conclusion = 0
    used_fallback = 0

    parsed = 0
    correct = 0
    incorrect = 0
    unknown = 0

    both_polarities = 0
    both_resolved_true = 0
    both_resolved_false = 0

    with open(IN_FILE, "r") as fin, open(OUT_FILE, "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            n += 1
            rec = json.loads(line)
            gt = str(rec.get("label", "")).lower()

            conc_text, had_explicit = extract_conclusion(rec.get("cot_explanation", ""))
            if had_explicit:
                found_conclusion += 1
            else:
                used_fallback += 1

            pred, dbg = normalize_pred_from_conclusion(conc_text)

            rec["conclusion_text"] = conc_text
            rec["pred_label"] = pred
            rec["parse_debug"] = dbg

            if dbg.get("both_polarities"):
                both_polarities += 1
                if pred == "true":
                    both_resolved_true += 1
                elif pred == "false":
                    both_resolved_false += 1

            if pred is None:
                rec["is_correct"] = None
                unknown += 1
            else:
                parsed += 1
                is_corr = (pred == gt)
                rec["is_correct"] = is_corr
                if is_corr:
                    correct += 1
                else:
                    incorrect += 1

            fout.write(json.dumps(rec) + "\n")

    print("=" * 80)
    print(f"Processed records          : {n}")
    print(f"Found explicit 'Conclusion': {found_conclusion}")
    print(f"Used fallback (last sent.) : {used_fallback}")
    print("-" * 80)
    print(f"Parsed predictions         : {parsed}")
    print(f"  - Correct                : {correct}")
    print(f"  - Incorrect              : {incorrect}")
    print(f"Unknown / ambiguous        : {unknown}")
    if parsed:
        acc = (correct / parsed) * 100.0
        print(f"Accuracy on parsed         : {correct}/{parsed} = {acc:.2f}%")
    print("-" * 80)
    print(f"Both-polarities cases      : {both_polarities}")
    print(f"  - Resolved to TRUE       : {both_resolved_true}")
    print(f"  - Resolved to FALSE      : {both_resolved_false}")
    print("=" * 80)
    print(f"Wrote augmented file to    : {OUT_FILE}")


if __name__ == "__main__":
    main()
