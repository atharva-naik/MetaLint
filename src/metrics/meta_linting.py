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

    coarse_precision = {k: 0 for k in idiom_codes}
    coarse_recall = {k: 0 for k in idiom_codes}
    for idiom_code in per_idiom_gts:
        num = 0 # true positives.
        denom_P = 0 # true positives + false positives.
        denom_R = 0 # true positives + false negatives.
        for index,count in enumerate(per_idiom_gts[idiom_code]):
            if count > 0: 
                denom_R += 1
                if count == per_idiom_preds[idiom_code][index]: 
                    num += 1
        for count in per_idiom_preds[idiom_code]:
            if count > 0: denom_P += 1
        assert denom_P >= num
        assert denom_R >= num    
        coarse_precision[idiom_code] = num/denom_P if denom_P != 0 else 1
        coarse_recall[idiom_code] = num/denom_R if denom_R != 0 else 1

    return idiom_overlaps, coarse_precision, coarse_recall

# main
if __name__ == "__main__":
    model_preds = read_jsonl("../../data/meta_linting_preds/qwen2.5coder_3b_instruct_sft_preds.jsonl")
    coarse_idiom_overlaps, coarse_P, coarse_R = compute_coarse_overlap_per_idiom(model_preds)
    coarse_F = {}
    
    for idiom_code in coarse_P:
        p = coarse_P[idiom_code]
        r = coarse_R[idiom_code]
        if p!=0 and r!=0:
            f = 2*p*r/(p+r)
        else: f = 0
        coarse_F[idiom_code] = f

    # print(coarse_idiom_overlaps)
    # print(np.mean(list(coarse_idiom_overlaps.values())))
    # print()
    print(coarse_P)
    print("P:", np.mean(list(coarse_P.values())))
    print(coarse_R)
    print("R:", np.mean(list(coarse_R.values())))
    print(coarse_F)
    print("F:", np.mean(list(coarse_F.values())))
    # print()