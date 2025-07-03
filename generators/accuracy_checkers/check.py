import sys
import os
project_root = os.path.abspath('...')
sys.path.append(project_root)
import json
import requests

from truthfulClaim_answer_generator import *

# === CONFIGURATION ===
C = 0
DATASET_LENGTH = 817
DATASET_NAME = "TruthfulClaim"
MODEL = "qwen"
INPUT_FILE = "generators/truthful_claims_dataset.json"

# === MAIN ===
if __name__ == "__main__":

    claims_data = load_claim_dataset(INPUT_FILE)
    generated_results = generate_answer(claims_data)
    results = generated_results
    for result_item in results:
        if result_item["actual label"]==result_item["generated label"]:
            C+=1

    print(f"Final accuracy score for {MODEL} on {DATASET_NAME}: ", C/DATASET_LENGTH)
