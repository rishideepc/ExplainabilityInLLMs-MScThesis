# evaluation/shake_score.py

def compute_shake_score(
    label_orig: str,
    conf_orig: float,
    label_pert: str,
    conf_pert: float,
    ground_truth: str
) -> float:
    print(f"Scoring SHAKE: base=({label_orig}, {conf_orig:.2f}) | perturbed=({label_pert}, {conf_pert:.2f}) | ground_truth={ground_truth}")
    if label_orig != label_pert:
        print("Label changed → SHAKE_SCORE = 1.0")
        return 1.0

    delta = conf_pert - conf_orig

    if label_orig == ground_truth:
        score = -delta
        print(f"Original label correct → confidence drop = {-delta:.4f}")
    else:
        score = delta
        print(f"Original label wrong → confidence drop = {delta:.4f}")

    return score
