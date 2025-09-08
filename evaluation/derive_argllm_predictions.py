import json, os, csv

IN_PATH  = "/vol/bitbucket/rc1124/MSc_Individual_Project/ExplainabilityInLLMs-MScThesis/results/generation/argLLM_generation/strategyclaim_mistral/argllm_outputs_ollama.jsonl"
OUT_JSONL = IN_PATH.replace(".jsonl", "_with_preds.jsonl")
OUT_CSV   = IN_PATH.replace(".jsonl", "_summary.csv")
THRESHOLD = 0.5

def to_label(p: float | None, tau: float = 0.5) -> str | None:
    """
    Thresholds a strength into a label

    @params: claim strength, threshold tau 
    
    @returns: 'true' or 'false' or None.
    """
    if p is None:
        return None
    try:
        return "true" if float(p) >= tau else "false"
    except Exception:
        return None

def main():
    """
    Augments argLLM JSONL files with predicted labels (based on thresholding)
    """
    os.makedirs(os.path.dirname(OUT_JSONL), exist_ok=True)

    n = 0
    base_ok = base_bad = base_unk = 0
    est_ok  = est_bad  = est_unk  = 0

    with open(IN_PATH, "r") as fin, open(OUT_JSONL, "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            n += 1
            rec = json.loads(line)

            gt = str(rec.get("label", "")).lower()
            b_pred = rec.get("base", {}).get("prediction", None)
            e_pred = rec.get("estimated", {}).get("prediction", None)

            b_lab = to_label(b_pred, THRESHOLD)
            e_lab = to_label(e_pred, THRESHOLD)

            if b_lab is None:
                b_corr = None; base_unk += 1
            else:
                b_corr = (b_lab == gt)
                if b_corr: base_ok += 1
                else: base_bad += 1

            if e_lab is None:
                e_corr = None; est_unk += 1
            else:
                e_corr = (e_lab == gt)
                if e_corr: est_ok += 1
                else: est_bad += 1

            rec["base_pred_strength"] = b_pred
            rec["est_pred_strength"]  = e_pred
            rec["base_pred_label"]    = b_lab
            rec["est_pred_label"]     = e_lab
            rec["base_is_correct"]    = b_corr
            rec["est_is_correct"]     = e_corr

            fout.write(json.dumps(rec) + "\n")

    base_parsed = base_ok + base_bad
    est_parsed  = est_ok + est_bad
    base_acc = (base_ok / base_parsed) if base_parsed else None
    est_acc  = (est_ok / est_parsed) if est_parsed else None

    print("="*80)
    print(f"Processed records       : {n}")
    print(f"[BASE] parsed/unknown   : {base_parsed}/{base_unk}")
    print(f"[BASE] correct/incorrect: {base_ok}/{base_bad}")
    print(f"[BASE] accuracy         : {'' if base_acc is None else f'{base_acc:.4f}'}")
    print("-"*80)
    print(f"[EST ] parsed/unknown   : {est_parsed}/{est_unk}")
    print(f"[EST ] correct/incorrect: {est_ok}/{est_bad}")
    print(f"[EST ] accuracy         : {'' if est_acc is None else f'{est_acc:.4f}'}")
    print("="*80)
    print(f"Wrote augmented JSONL   : {OUT_JSONL}")

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "n_total","base_parsed","base_correct","base_incorrect","base_unknown","base_accuracy",
            "est_parsed","est_correct","est_incorrect","est_unknown","est_accuracy","threshold"
        ])
        w.writerow([
            n, base_parsed, base_ok, base_bad, base_unk,
            ("" if base_acc is None else f"{base_acc:.4f}"),
            est_parsed, est_ok, est_bad, est_unk,
            ("" if est_acc is None else f"{est_acc:.4f}"),
            THRESHOLD
        ])
    print(f"Wrote summary CSV       : {OUT_CSV}")

if __name__ == "__main__":
    main()
