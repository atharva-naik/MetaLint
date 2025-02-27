# metrics for the meta linting task.

import os
import sys
import json
import pathlib
import numpy as np
from fuzzywuzzy import fuzz
from collections import defaultdict

module_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent.parent)
sys.path.append(module_path)

from src.datautils import read_jsonl
IDIOMS_SEEN_IN_TRAIN = ["ERA001", "C901", "I001", "I002", "BLE001"] # 1 no transfer or common between train and test (NoT)
IDIOMS_FROM_GROUP_SEEN_IN_TRAIN = ["F406", "F403", "F503", "F602", "F622", "E401", "E702", "E722", "E731", "E742"] # 2 near transfer (NeT)
NEAR_TRANSFER_IDIOM_GROUP_MAPPING = {
    "F406": ["F405"], # test: train
    "F403": ['F405'],
    "F503": ["F501","F502"],
    "F602": ["F601"],
    "F622": ["F621"],
    "E401": ["E402"],
    "E702": ["E701"],
    "E722": ["E721"],
    "E742": ["E741", "E743"],
}
NEAR_TRANSFER_TEST_IDIOM_GROUPS = list(NEAR_TRANSFER_IDIOM_GROUP_MAPPING.keys())
NEAR_TRANSFER_TRAIN_IDIOM_GROUPS = ["F405","F501","F502","F601","F621","E402","E701","E721","E741","E743"]
IDIOMS_NOT_SEEN_IN_TRAIN = ["ANN001", "ANN002", "ANN003", "ANN201", "ANN202", "ANN204", "ANN205", "ANN206"]+["ASYNC100", "ASYNC105", "ASYNC109", "ASYNC110", "ASYNC115", "ASYNC116", "ASYNC210", "ASYNC220", "ASYNC221", "ASYNC222", "ASYNC230", "ASYNC251"]+["S102", "S103", "S104", "S105", "S106", "S107", "S108", "S110", "S112", "S113", "S201", "S202", "S301", "S302", "S303"] # 3 far transfer: everything else. (FaT)

# print(len(IDIOMS_FROM_GROUP_SEEN_IN_TRAIN)+len(IDIOMS_NOT_SEEN_IN_TRAIN)+len(IDIOMS_SEEN_IN_TRAIN))
# exit()

TOOL_GROUPS = {
    "PyFlakes": ["F406", "F403", "F503", "F602", "F622"],
    "pycodestyle": ["E401", "E702", "E722", "E731", "E742"], 
    "Misc": ["ERA001", "C901", "I001", "I002", "BLE001"], # most frequent 'meta-linting' group in training data.
    "flake8-annotations": ["ANN001", "ANN002", "ANN003", "ANN201", "ANN202", "ANN204", "ANN205", "ANN206"],
    "flake8-async": ["ASYNC100", "ASYNC105", "ASYNC109", "ASYNC110", "ASYNC115", "ASYNC116", "ASYNC210", "ASYNC220", "ASYNC221", "ASYNC222", "ASYNC230", "ASYNC251"],
    "flake8-bandit": ["S102", "S103", "S104", "S105", "S106", "S107", "S108", "S110", "S112", "S113", "S201", "S202", "S301", "S302", "S303"],
}
IDIOM_TO_TOOL_GROUP = {}
for tool_group, idiom_codes in TOOL_GROUPS.items():
    for idiom_code in idiom_codes:
        IDIOM_TO_TOOL_GROUP[idiom_code] = tool_group

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

def safe_mean(arr):
    if len(arr) == 0: return 0
    return np.mean(arr).item()

def compute_idiom_frequencies(path: str):
    train_data = json.load(open(path))
    idiom_code_dist = defaultdict(lambda: 0)
    any_tool = 0
    for rec in train_data:
        results = load_linter_results(rec['messages'][1]['content'])
        tools_covered = set()
        for result in results:
            tools_covered.add(IDIOM_TO_TOOL_GROUP[result['code']])
        for tool in tools_covered:
            idiom_code_dist[tool] += 1
        if len(tools_covered) > 0: any_tool += 1
    for tool,freq in idiom_code_dist.items():
        idiom_code_dist[tool] = f"{100*freq/len(train_data):.2f}%"
    idiom_code_dist["all"] = f"{100*any_tool/len(train_data):.2f}%"

    return dict(idiom_code_dist)

def compute_subset_wise_metrics_(coarse_metric_dict: dict[str, float], subset: list[str]):
    metric_dict = {}
    # print(len(coarse_metric_dict))
    for k,v in coarse_metric_dict.items():
        if k in subset: 
            # print(k,v)
            metric_dict[k] = v

    return np.mean(list(metric_dict.values()))

def compute_subset_wise_metrics(coarse_metric_dict: dict[str, float]):
    global IDIOMS_SEEN_IN_TRAIN
    global IDIOMS_NOT_SEEN_IN_TRAIN
    global IDIOMS_FROM_GROUP_SEEN_IN_TRAIN
    
    seen_in_train_metric_dict = {k: 0 for k in IDIOMS_SEEN_IN_TRAIN}
    from_group_seen_in_train_metric_dict = {k: 0 for k in IDIOMS_NOT_SEEN_IN_TRAIN}
    not_seen_in_train_metric_dict = {k: 0 for k in IDIOMS_FROM_GROUP_SEEN_IN_TRAIN}

    for k,v in coarse_metric_dict.items():
        if k in IDIOMS_SEEN_IN_TRAIN:
            seen_in_train_metric_dict[k] = v
        if k in IDIOMS_NOT_SEEN_IN_TRAIN:
            not_seen_in_train_metric_dict[k] = v
        if k in IDIOMS_FROM_GROUP_SEEN_IN_TRAIN:
            from_group_seen_in_train_metric_dict[k] = v

    return np.mean(list(seen_in_train_metric_dict.values())), np.mean(list(from_group_seen_in_train_metric_dict.values())), np.mean(list(not_seen_in_train_metric_dict.values()))

def exact_match_alignment(pred: list[str], gt: list[str]):
    assert len(gt) != 0
    # precision 
    p = 0
    for x in pred:
        p += int(any([y == x for y in gt]))
    if len(pred) == 0: p = 0
    else: p /= len(pred)
    # recall
    r = 0
    for y in gt:
        r += int(any([y == x for x in pred]))
    if len(gt) == 0: r = 0
    else: r /= len(gt)

    return p, r

def fuzzy_match_alignment(pred: list[str], gt: list[str]):
    assert len(gt) != 0
    # precision 
    p = 0
    fuzzy_scores = np.zeros((len(gt),len(pred)))
    
    for i,y in enumerate(gt):
        for j,x in enumerate(pred):
            fuzzy_scores[i][j] = fuzz.ratio(y, x) > 90
    
    if len(pred) == 0: p = 0
    else: p = fuzzy_scores.max(1).sum()/len(pred)
    if len(gt) == 0: r = 0
    else: r = fuzzy_scores.max(0).sum()/len(gt)

    return p, r

def compute_coarse_overlap_per_idiom(data):
    global IDIOMS_SEEN_IN_TRAIN
    global IDIOMS_NOT_SEEN_IN_TRAIN
    global IDIOMS_FROM_GROUP_SEEN_IN_TRAIN

    idiom_codes = set(IDIOMS_SEEN_IN_TRAIN+IDIOMS_NOT_SEEN_IN_TRAIN+IDIOMS_FROM_GROUP_SEEN_IN_TRAIN)
    ground_truths = []
    predictions = []

    for rec in data:
        gt = load_linter_results(rec["ground_truth"])
        pred = load_linter_results(rec["model_response"])
        ground_truths.append(gt)
        predictions.append(pred)
    print(f"json decoding error for {FAULTY_RESULT_CTR} linter result predictions")

    per_idiom_preds = {k: [0 for _ in range(len(ground_truths))] for k in idiom_codes}
    per_idiom_pred_spans = {k: [[] for _ in range(len(ground_truths))] for k in idiom_codes}
    per_idiom_pred_lines = {k: [[] for _ in range(len(ground_truths))] for k in idiom_codes}

    per_idiom_gts = {k: [0 for _ in range(len(ground_truths))] for k in idiom_codes}
    per_idiom_gt_spans = {k: [[] for _ in range(len(ground_truths))] for k in idiom_codes}
    per_idiom_gt_lines = {k: [[] for _ in range(len(ground_truths))] for k in idiom_codes}

    per_idiom_mispreds = defaultdict(lambda: [0 for _ in range(len(ground_truths))])

    for i, (gt, pred) in enumerate(zip(ground_truths, predictions)):
        
        for result in gt:
            per_idiom_gts[result['code']][i] += 1
            for code_span_and_line in result["code_spans_and_lines"]:
                per_idiom_gt_spans[result['code']][i].append(code_span_and_line["span"])
                per_idiom_gt_lines[result['code']][i].append(code_span_and_line["line"])

        for result in pred:
            try: 
                per_idiom_preds[result['code']][i] += 1
                for code_span_and_line in result["code_spans_and_lines"]:
                    per_idiom_pred_spans[result['code']][i].append(code_span_and_line["span"])
                    per_idiom_pred_lines[result['code']][i].append(code_span_and_line["line"])
            except KeyError: pass
            per_idiom_mispreds[result['code']][i] += 1
            # except TypeError:
            #     print(result, i+1)
            #     exit()
    idiom_overlaps = {k: 0 for k in idiom_codes}
    
    for idiom_code in per_idiom_gts:
        # if idiom_code not in IDIOMS_FROM_GROUP_SEEN_IN_TRAIN and idiom_code not in IDIOMS_SEEN_IN_TRAIN:
        #     IDIOMS_NOT_SEEN_IN_TRAIN.append(idiom_code)
        idiom_overlaps[idiom_code] = sum([
            int(gt==pred) for gt,pred in zip(
                per_idiom_gts[idiom_code], 
                per_idiom_preds[idiom_code]
            )
        ])/len(ground_truths)

    coarse_precision = {k: 0 for k in idiom_codes}
    coarse_recall = {k: 0 for k in idiom_codes}
    EM_span_alignment_precision = {k: [] for k in idiom_codes}
    EM_line_alignment_precision = {k: [] for k in idiom_codes}
    EM_span_alignment_recall = {k: [] for k in idiom_codes}
    EM_line_alignment_recall = {k: [] for k in idiom_codes}
    FZ_span_alignment_precision = {k: [] for k in idiom_codes}
    FZ_line_alignment_precision = {k: [] for k in idiom_codes}
    FZ_span_alignment_recall = {k: [] for k in idiom_codes}
    FZ_line_alignment_recall = {k: [] for k in idiom_codes}
    # rate_of_confusing_mispreds = {k: 0 for k in NEAR_TRANSFER_IDIOM_GROUP_MAPPING}
    strict_micro_averaged_mispred_rate_num = 0
    strict_micro_averaged_mispred_rate_denom = 0 
    relaxed_micro_averaged_mispred_rate_num = 0
    relaxed_micro_averaged_mispred_rate_denom = 0 
    any_micro_averaged_mispred_rate_num = 0
    any_micro_averaged_mispred_rate_denom = 0

    for idiom_code in per_idiom_gts:
        # effectively considers both idiom detection and idiom localization.
        for index,count in enumerate(per_idiom_gts[idiom_code]):
            if count > 0:
                
                gt_lines = per_idiom_gt_lines[idiom_code][index]
                pred_lines = per_idiom_pred_lines[idiom_code][index]
                gt_spans = per_idiom_gt_spans[idiom_code][index]
                pred_spans = per_idiom_pred_spans[idiom_code][index]

                line_alignment_p, line_alignment_r = exact_match_alignment(pred_lines, gt_lines)
                span_alignment_p, span_alignment_r = exact_match_alignment(pred_spans, gt_spans)
                
                EM_span_alignment_precision[idiom_code].append(span_alignment_p)
                EM_span_alignment_recall[idiom_code].append(span_alignment_r)
                EM_line_alignment_precision[idiom_code].append(line_alignment_p)
                EM_line_alignment_recall[idiom_code].append(line_alignment_r)

                line_alignment_p, line_alignment_r = fuzzy_match_alignment(pred_lines, gt_lines)
                span_alignment_p, span_alignment_r = fuzzy_match_alignment(pred_spans, gt_spans)

                FZ_span_alignment_precision[idiom_code].append(span_alignment_p)
                FZ_span_alignment_recall[idiom_code].append(span_alignment_r)
                FZ_line_alignment_precision[idiom_code].append(line_alignment_p)
                FZ_line_alignment_recall[idiom_code].append(line_alignment_r)

        # span and line alignment based precision and recall.
        EM_span_alignment_precision[idiom_code] = safe_mean(EM_span_alignment_precision[idiom_code])
        EM_span_alignment_recall[idiom_code] = safe_mean(EM_span_alignment_recall[idiom_code])
        EM_line_alignment_precision[idiom_code] = safe_mean(EM_line_alignment_precision[idiom_code])
        EM_line_alignment_recall[idiom_code] = safe_mean(EM_line_alignment_recall[idiom_code])
        FZ_span_alignment_precision[idiom_code] = safe_mean(FZ_span_alignment_precision[idiom_code])
        FZ_span_alignment_recall[idiom_code] = safe_mean(FZ_span_alignment_recall[idiom_code])
        FZ_line_alignment_precision[idiom_code] = safe_mean(FZ_line_alignment_precision[idiom_code])
        FZ_line_alignment_recall[idiom_code] = safe_mean(FZ_line_alignment_recall[idiom_code])

    line_EM_P = np.mean(list(EM_line_alignment_precision.values())).item()
    line_EM_R = np.mean(list(EM_line_alignment_recall.values())).item()
    line_EM_F = compute_f_score(p=line_EM_P, r=line_EM_R)
    line_FZ_P = np.mean(list(FZ_line_alignment_precision.values())).item()
    line_FZ_R = np.mean(list(FZ_line_alignment_recall.values())).item()
    line_FZ_F = compute_f_score(p=line_FZ_P, r=line_FZ_R)

    span_EM_P = np.mean(list(EM_span_alignment_precision.values())).item()
    span_EM_R = np.mean(list(EM_span_alignment_recall.values())).item()
    span_EM_F = compute_f_score(p=span_EM_P, r=span_EM_R)
    span_FZ_P = np.mean(list(FZ_span_alignment_precision.values())).item()
    span_FZ_R = np.mean(list(FZ_span_alignment_recall.values())).item()
    span_FZ_F = compute_f_score(p=span_FZ_P, r=span_FZ_R)

    localization_scores = {
        "line": {
            "EM-P": round(line_EM_P,4),
            "EM-R": round(line_EM_R,4),
            "EM-F": round(line_EM_F,4),
            "Fuzzy-P": round(line_FZ_P,4),
            "Fuzzy-R": round(line_FZ_R,4),
            "Fuzzy-F": round(line_FZ_F,4),
        },
        "span": {
            "EM-P": round(span_EM_P,4),
            "EM-R": round(span_EM_R,4),
            "EM-F": round(span_EM_F,4),
            "Fuzzy-P": round(span_FZ_P,4),
            "Fuzzy-R": round(span_FZ_R,4),
            "Fuzzy-F": round(span_FZ_F,4),
        }
    }

    for idiom_code in per_idiom_gts:
        num = 0 # true poNoTives.
        denom_P = 0 # true poNoTives + false poNoTives.
        denom_R = 0 # true poNoTives + false negatives.

        # compute rate of mispreds between corresponding pairs.
        if idiom_code in NEAR_TRANSFER_IDIOM_GROUP_MAPPING:
            # mispred_rate_num = 0
            # mispred_rate_denom = 0
            confusing_codes = NEAR_TRANSFER_IDIOM_GROUP_MAPPING[idiom_code]
            for index,count in enumerate(per_idiom_gts[idiom_code]):
                if count > 0: 
                    strict_micro_averaged_mispred_rate_denom += 1
                    relaxed_micro_averaged_mispred_rate_denom += 1
                    any_micro_averaged_mispred_rate_denom += 1
                    for confusing_code in NEAR_TRANSFER_TRAIN_IDIOM_GROUPS:
                        if per_idiom_mispreds[confusing_code][index] > 0:
                            any_micro_averaged_mispred_rate_num += 1
                            break
                    for confusing_code in confusing_codes: # but for the same instance expect at least one occurence of the confusing_codes.
                        if per_idiom_mispreds[confusing_code][index] > 0 and per_idiom_preds[idiom_code][index] == 0: # no predictions of the expected ground truth code.
                            strict_micro_averaged_mispred_rate_num += 1
                            break
                    for confusing_code in confusing_codes: # but for the same instance expect at least one occurence of the confusing_codes.
                        if per_idiom_mispreds[confusing_code][index] > 0:
                            relaxed_micro_averaged_mispred_rate_num += 1
                            break
            # rate_of_confusing_mispreds[idiom_code] = mispred_rate_num/mispred_rate_denom

        for index,count in enumerate(per_idiom_gts[idiom_code]):
            if count > 0: 
                denom_R += 1
                if per_idiom_preds[idiom_code][index] > 0: # count prediction as correct if even one instance is detected.
                    num += 1
        for count in per_idiom_preds[idiom_code]:
            if count > 0: denom_P += 1
        assert denom_P >= num
        assert denom_R >= num    
        # if idiom_code == "F403":
        #     print(idiom_code+":", num, denom_P, denom_R)
        coarse_precision[idiom_code] = num/denom_P if denom_P != 0 else 0
        coarse_recall[idiom_code] = num/denom_R if denom_R != 0 else 0

    strict_micro_averaged_mispred_rate = strict_micro_averaged_mispred_rate_num/strict_micro_averaged_mispred_rate_denom
    relaxed_micro_averaged_mispred_rate = relaxed_micro_averaged_mispred_rate_num/relaxed_micro_averaged_mispred_rate_denom
    any_micro_averaged_mispred_rate = any_micro_averaged_mispred_rate_num/any_micro_averaged_mispred_rate_denom
    print(f"strict micro averaged mispred rate: {100*strict_micro_averaged_mispred_rate:.2f}%")
    print(f"relaxed micro averaged mispred rate: {100*relaxed_micro_averaged_mispred_rate:.2f}%")
    print(f"any micro averaged mispred rate: {100*any_micro_averaged_mispred_rate:.2f}%")

    return idiom_overlaps, coarse_precision, coarse_recall, localization_scores#, rate_of_confusing_mispreds

def compute_f_score(p, r):
    if p + r == 0: return 0
    return 2*p*r/(p+r)

# main
if __name__ == "__main__":
    try: train_steps: int=int(sys.argv[1])
    except IndexError: train_steps: int=2000
    # model_preds = read_jsonl(f"data/meta_linting_preds/qwen2.5coder_3b_instruct_sft_preds_{train_steps}.jsonl")
    # model_preds = read_jsonl(f"data/meta_linting_preds/qwen2.5coder_3b_instruct_sft_preds_{train_steps}-v2-data.jsonl")
    model_preds = read_jsonl(f"data/meta_linting_preds/qwen2.5coder_3b_instruct_sft_preds_{train_steps}-idiom-hardness-no-packing.jsonl")
    # model_preds = read_jsonl(f"data/meta_linting_preds/qwen2.5coder_3b_instruct_sft_preds_{train_steps}-idiom-hardness.jsonl")
    # evals_path = f"data/meta_linting_evals/idiom_detection/qwen2.5coder_3b_instruct_sft_preds_{train_steps}-v2-data.json"
    # evals_path = f"data/meta_linting_evals/idiom_detection/qwen2.5coder_3b_instruct_sft_preds_{train_steps}-idiom-hardness.json"
    evals_path = f"data/meta_linting_evals/idiom_detection/qwen2.5coder_3b_instruct_sft_preds_{train_steps}-idiom-hardness-no-packing.json"
    # coarse_idiom_overlaps, coarse_P, coarse_R, rate_of_confusing_mispreds = compute_coarse_overlap_per_idiom(model_preds)
    coarse_idiom_overlaps, coarse_P, coarse_R, localization_scores = compute_coarse_overlap_per_idiom(model_preds)

    # print(rate_of_confusing_mispreds)
    # coarse_F = {}
    
    # for idiom_code in coarse_P:
    #     p = coarse_P[idiom_code]
    #     r = coarse_R[idiom_code]
    #     if p!=0 and r!=0:
    #         f = 2*p*r/(p+r)
    #     else: f = 0
    #     coarse_F[idiom_code] = f

    # print(coarse_idiom_overlaps)
    # print(np.mean(list(coarse_idiom_overlaps.values())))
    # print()

    # print(coarse_P)
    # print(coarse_R)
    # print(coarse_F)
    # print()

    P = np.mean(list(coarse_P.values()))
    R = np.mean(list(coarse_R.values()))
    F = compute_f_score(P, R)
    NoT_P, NeT_P, FaT_P = compute_subset_wise_metrics(coarse_P) 
    NoT_R, NeT_R, FaT_R = compute_subset_wise_metrics(coarse_R) 
    NoT_F = compute_f_score(NoT_P, NoT_R) 
    NeT_F = compute_f_score(NeT_P, NeT_R) 
    FaT_F = compute_f_score(FaT_P, FaT_R) 

    print(f"all: P: {P:.4f}, R: {R:.4f}, F: {F:.4f}")
    print(f"NoT: P: {NoT_P:.4f}, R: {NoT_R:.4f}, F: {NoT_F:.4f}")
    print(f"NeT: P: {NeT_P:.4f}, R: {NeT_R:.4f}, F: {NeT_F:.4f}")
    print(f"FaT: P: {FaT_P:.4f}, R: {FaT_R:.4f}, F: {FaT_F:.4f}")
    print()
    print(f'Line Localization (EM): {localization_scores["line"]["EM-P"]} {localization_scores["line"]["EM-R"]} {localization_scores["line"]["EM-F"]}')
    print(f'Line Localization (Fuzzy): {localization_scores["line"]["Fuzzy-P"]} {localization_scores["line"]["Fuzzy-R"]} {localization_scores["line"]["Fuzzy-F"]}')
    print()
    print(f'Span Localization (EM): {localization_scores["span"]["EM-P"]} {localization_scores["span"]["EM-R"]} {localization_scores["span"]["EM-F"]}')
    print(f'Span Localization (Fuzzy): {localization_scores["span"]["Fuzzy-P"]} {localization_scores["span"]["Fuzzy-R"]} {localization_scores["span"]["Fuzzy-F"]}')
    print()

    evals_json = {
        "overall": {"P": P, "R": R, "F": F},
        "NoT": {"P": NoT_P, "R": NoT_R, "F": NoT_F},
        "NeT":{"P": NeT_P, "R": NeT_R, "F": NeT_F},
        "FaT": {"P": FaT_P, "R": FaT_R, "F": FaT_F},
        "tool_groups": {}
    }

    # print(f"P: {P:.4f} NoT_P: {NoT_P:.4f} NeT_P: {NeT_P:.4f} FaT_P: {FaT_P:.4f}")
    # print(f"R: {R:.4f} NoT_R: {NoT_R:.4f} NeT_R: {NeT_R:.4f} FaT_R: {FaT_R:.4f}")
    # print(f"F: {F:.4f} NoT_F: {NoT_F:.4f} NeT_F: {NeT_F:.4f} FaT_F: {FaT_F:.4f}")

    for tool_group_name,tool_group_subset in TOOL_GROUPS.items():
        # if tool_group_name == "PyFlakes":
        #     for code in tool_group_subset:
        #         print(code+":", coarse_P[code])
        group_P = compute_subset_wise_metrics_(coarse_P, tool_group_subset)
        group_R = compute_subset_wise_metrics_(coarse_R, tool_group_subset)
        group_F = compute_f_score(group_P, group_R)
        evals_json["tool_groups"][tool_group_name] = {"P": group_P, "R": group_R, "F": group_F}
        
        print(f"{tool_group_name} P: {group_P:.4f} R: {group_R:.4f} F: {group_F:.4f}")
    
    print(compute_idiom_frequencies("./data/ruff_meta_linting/hardness_experiment/train.json"))
    # save evaluation metric values obtained by the model.
    with open(evals_path, "w") as f:
        json.dump(evals_json, f, indent=4)