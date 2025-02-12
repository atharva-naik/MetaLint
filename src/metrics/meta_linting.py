# metrics for the meta linting task.

import os
import sys
import json
import pathlib
import numpy as np
from collections import defaultdict

module_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent)
sys.path.append(module_path)

from src.datautils import read_jsonl

FAULTY_RESULT_CTR = 0
def load_linter_results(text):
    results = []
    if text.strip() == "NO VIOLATIONS FOUND":
        return []
    else: 
        for line in text.split("\n"):
            line = line.strip()
            try: results.append(json.loads(line))
            except json.JSONDecodeError:
                # print(line)
                global FAULTY_RESULT_CTR
                FAULTY_RESULT_CTR += 1
                # print(text)
                # exit()
    return results

def compute_coarse_overlap_per_idiom(data):
    idiom_codes = set()
    ground_truths = []
    predictions = []

    for rec in data:
        gt = load_linter_results(rec["ground_truth"])
        pred = load_linter_results(rec["model_response"])
        ground_truths.append(gt)
        predictions.append(pred)
        for result in gt:
            idiom_codes.add(result['code'])
    print(f"json decoding error for {FAULTY_RESULT_CTR} linter result predictions")

    per_idiom_preds = {k: [0 for _ in range(len(ground_truths))] for k in idiom_codes}
    per_idiom_gts = {k: [0 for _ in range(len(ground_truths))] for k in idiom_codes}
    for i, (gt, pred) in enumerate(zip(ground_truths, predictions)):
        for result in gt:
            per_idiom_gts[result['code']][i] += 1
        for result in pred:
            try: per_idiom_preds[result['code']][i] += 1
            except KeyError: pass
    idiom_overlaps = {k: 0 for k in idiom_codes}
    
    for idiom_code in per_idiom_gts:
        idiom_overlaps[idiom_code] = sum([
            int(gt==pred) for gt,pred in zip(
                per_idiom_gts[idiom_code], 
                per_idiom_preds[idiom_code]
            )
        ])/len(ground_truths)

    return idiom_overlaps

# main
if __name__ == "__main__":
    model_preds = read_jsonl("data/meta_linting_preds/qwen2.5coder_3b_instruct_sft_preds.jsonl")
    coarse_idiom_overlaps = compute_coarse_overlap_per_idiom(model_preds)
    print(np.mean(list(coarse_idiom_overlaps.values())))
    print(coarse_idiom_overlaps)