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
IDIOMS_SEEN_IN_TRAIN = ["ERA001", "C901", "I001", "I002", "BLE001"] # 1 (SIT)
IDIOMS_FROM_GROUP_SEEN_IN_TRAIN = ["F406", "F403", "F503", "F602", "F622", "E401", "E702", "E722", "E731", "E742"] # 2 (GSIT)
IDIOMS_NOT_SEEN_IN_TRAIN = [] # 3 everything else. (NSIT)

TOOL_GROUPS = {
    "PyFlakes": ["F406", "F403", "F503", "F602", "F622"],
    "pycodestyle": ["E401", "E702", "E722", "E731", "E742"], 
    "Misc": ["ERA001", "C901", "I001", "I002", "BLE001"], # most frequent 'meta-linting' group in training data.
    "flake8-annotations": ["ANN001", "ANN002", "ANN003", "ANN201", "ANN202", "ANN204", "ANN205", "ANN206"],
    "flake8-async": ["ASYNC100", "ASYNC105", "ASYNC109", "ASYNC110", "ASYNC115", "ASYNC116", "ASYNC210", "ASYNC220", "ASYNC221", "ASYNC222", "ASYNC230", "ASYNC251"],
    "flake8-bandit": ["S102", "S103", "S104", "S105", "S106", "S107", "S108", "S110", "S112", "S113", "S201", "S202", "S301", "S302", "S303"],
}

FAULTY_RESULT_CTR = 0
def load_linter_results(text):
    results = []
    if text.strip() == "NO VIOLATIONS FOUND":
        return []
    else: 
        for line in text.split("\n"):
            line = line.strip()
            try: 
                result = json.loads(line)
                if isinstance(result, dict):
                    results.append(result)
            except json.JSONDecodeError:
                # print(line)
                global FAULTY_RESULT_CTR
                FAULTY_RESULT_CTR += 1
                # print(text)
                # exit()
    return results

def compute_subset_wise_metrics_(coarse_metric_dict: dict[str, float], subset: list[str]):
    metric_dict = {}
    for k,v in coarse_metric_dict.items():
        if k in subset: metric_dict[k] = v

    return np.mean(list(metric_dict.values()))

def compute_subset_wise_metrics(coarse_metric_dict: dict[str, float]):
    global IDIOMS_SEEN_IN_TRAIN
    global IDIOMS_NOT_SEEN_IN_TRAIN
    global IDIOMS_FROM_GROUP_SEEN_IN_TRAIN
    seen_in_train_metric_dict = {}
    from_group_seen_in_train_metric_dict = {}
    not_seen_in_train_metric_dict = {}
    for k,v in coarse_metric_dict.items():
        if k in IDIOMS_SEEN_IN_TRAIN:
            seen_in_train_metric_dict[k] = v
        if k in IDIOMS_NOT_SEEN_IN_TRAIN:
            not_seen_in_train_metric_dict[k] = v
        if k in IDIOMS_FROM_GROUP_SEEN_IN_TRAIN:
            from_group_seen_in_train_metric_dict[k] = v

    return np.mean(list(seen_in_train_metric_dict.values())), np.mean(list(from_group_seen_in_train_metric_dict.values())), np.mean(list(not_seen_in_train_metric_dict.values()))

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

    global IDIOMS_SEEN_IN_TRAIN
    global IDIOMS_NOT_SEEN_IN_TRAIN
    global IDIOMS_FROM_GROUP_SEEN_IN_TRAIN

    per_idiom_preds = {k: [0 for _ in range(len(ground_truths))] for k in idiom_codes}
    per_idiom_gts = {k: [0 for _ in range(len(ground_truths))] for k in idiom_codes}
    for i, (gt, pred) in enumerate(zip(ground_truths, predictions)):
        for result in gt:
            per_idiom_gts[result['code']][i] += 1
        for result in pred:
            try: per_idiom_preds[result['code']][i] += 1
            except KeyError: pass
            # except TypeError:
            #     print(result, i+1)
            #     exit()
    idiom_overlaps = {k: 0 for k in idiom_codes}
    
    for idiom_code in per_idiom_gts:
        if idiom_code not in IDIOMS_FROM_GROUP_SEEN_IN_TRAIN and idiom_code not in IDIOMS_SEEN_IN_TRAIN:
            IDIOMS_NOT_SEEN_IN_TRAIN.append(idiom_code)
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
                if per_idiom_preds[idiom_code][index] > 0: # count prediction as correct if even one instance is detected.
                    num += 1
        for count in per_idiom_preds[idiom_code]:
            if count > 0: denom_P += 1
        assert denom_P >= num
        assert denom_R >= num    
        coarse_precision[idiom_code] = num/denom_P if denom_P != 0 else 0
        coarse_recall[idiom_code] = num/denom_R if denom_R != 0 else 0

    return idiom_overlaps, coarse_precision, coarse_recall

# main
if __name__ == "__main__":
    try: train_steps: int=int(sys.argv[1])
    except IndexError: train_steps: int=2000
    # model_preds = read_jsonl(f"data/meta_linting_preds/qwen2.5coder_3b_instruct_sft_preds_{train_steps}.jsonl")
    model_preds = read_jsonl(f"data/meta_linting_preds/qwen2.5coder_3b_instruct_sft_preds_{train_steps}-v2-data.jsonl")
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

    # print(coarse_P)
    # print(coarse_R)
    # print(coarse_F)
    # print()

    SIT_P, GSIT_P, NSIT_P = compute_subset_wise_metrics(coarse_P) 
    print(f"P: {np.mean(list(coarse_P.values())):.4f} SIT_P: {SIT_P:.4f} GSIT_P: {GSIT_P:.4f} NSIT_P: {NSIT_P:.4f}")
    SIT_R, GSIT_R, NSIT_R = compute_subset_wise_metrics(coarse_R) 
    print(f"R: {np.mean(list(coarse_R.values())):.4f} SIT_R: {SIT_R:.4f} GSIT_R: {GSIT_R:.4f} NSIT_R: {NSIT_R:.4f}")
    SIT_F, GSIT_F, NSIT_F = compute_subset_wise_metrics(coarse_F) 
    print(f"F: {np.mean(list(coarse_F.values())):.4f} SIT_F: {SIT_F:.4f} GSIT_F: {GSIT_F:.4f} NSIT_F: {NSIT_F:.4f}")

    for tool_group_name,tool_group_subset in TOOL_GROUPS.items():
        group_P = compute_subset_wise_metrics_(coarse_P, tool_group_subset)
        group_R = compute_subset_wise_metrics_(coarse_R, tool_group_subset)
        group_F = compute_subset_wise_metrics_(coarse_F, tool_group_subset)
        print(f"{tool_group_name} P: {group_P:.4f} R: {group_R:.4f} F: {group_F:.4f}")