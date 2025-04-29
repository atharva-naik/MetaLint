# metrics for the meta linting task.
# Coarse Evaluation - Idiom Detection (this basically casts the task as a multi-label classification task: is a given idiom violated/present in a code file.)
# change from v2: this works for the new format with line numbers and meta-task specific predictions.

import os
import sys
import json
import pathlib
import numpy as np
from tqdm import tqdm
from fuzzywuzzy import fuzz
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score

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
FAULTY_RESULT_CTR = 0
TOOL_GROUPS = {
    "PyFlakes": ["F406", "F403", "F503", "F602", "F622"],
    "pycodestyle": ["E401", "E702", "E722", "E731", "E742"], 
    "Misc": ["ERA001", "C901", "I001", "I002", "BLE001"], # most frequent 'meta-linting' group in training data.
    "flake8-annotations": ["ANN001", "ANN002", "ANN003", "ANN201", "ANN202", "ANN204", "ANN205", "ANN206"],
    "flake8-async": ["ASYNC100", "ASYNC105", "ASYNC109", "ASYNC110", "ASYNC115", "ASYNC116", "ASYNC210", "ASYNC220", "ASYNC221", "ASYNC222", "ASYNC230", "ASYNC251"],
    "flake8-bandit": ["S102", "S103", "S104", "S105", "S106", "S107", "S108", "S110", "S112", "S113", "S201", "S202", "S301", "S302", "S303"],
}
TOOL_TO_TOOL_GROUP = {}
for tool_group_name, tools in TOOL_GROUPS.items():
    for tool in tools:
        TOOL_TO_TOOL_GROUP[tool] = tool_group_name
TEST_SET_IDIOMS = []
for tools in TOOL_GROUPS.values():
    TEST_SET_IDIOMS.extend(tools)
# print(TEST_SET_IDIOMS)
IDIOMS_ABSENT_FROM_TEST_SET = set()

IDIOMS_FOUND_HEADER = "## Idiom Violations Found"
def load_linter_results(text):
    results = []
    idiom_code = None
    for line in text.split("\n"):
        line = line.strip()
        if line == "": continue
        elif line.startswith("**Idiom") and line.endswith("Violations:**"):
            idiom_code = line.removesuffix("Violations:**").removeprefix("**Idiom").strip()
        elif line == "NO VIOLATIONS FOUND": continue
        else:
            try: 
                result = json.loads(line)
                result["code"] = idiom_code
                results.append(result)
            except Exception as e: pass
                # print(e)
                # print(f"{e}: {line}")
    return results

def compute_idiom_wise_freq_in_train(train_data):
    tool_group_freq = defaultdict(lambda: 0)
    tool_group_freq["all"] = 0
    for index,rec in tqdm(enumerate(train_data)):
        gt = load_linter_results(rec["messages"][1]['content'])
        tools_seen = set()    
        for violation in gt:
            idiom_code = violation["code"]
            tools_seen.add(idiom_code)
        for tool in tools_seen:
            tool_group_freq[TOOL_TO_TOOL_GROUP[tool]] += 1
        if len(tools_seen) > 0: tool_group_freq['all'] += 1

    tool_group_freq = dict(tool_group_freq)
    tool_group_freq = {k: round(100*v/len(train_data),2) for k,v in tool_group_freq.items()}

    print(tool_group_freq)

def compute_f_score(p, r):
    if p + r == 0: return 0
    return 2*p*r/(p+r)

# def compute_line_level_metric(data):
#     p_line, r_line = defaultdict(lambda: []), defaultdict(lambda: [])

#     for index,rec in tqdm(enumerate(data)):
#         model_resp = load_linter_results(rec["model_response"])
#         gt = load_linter_results(rec["ground_truth"])

#         idiom_wise_pred_lines = defaultdict(lambda: set())
#         idiom_wise_gt_lines = defaultdict(lambda: set())
#         for i,model_violation in enumerate(model_resp):
#             try: 
#                 idiom_wise_pred_lines[model_violation['code']].add(int(model_violation['line'].split()[0].strip().removesuffix(":")))       
#             except ValueError: pass # print(model_violation['line'])
#             except KeyError: pass # print(model_violation.keys())
#         for i,gt_violation in enumerate(gt):
#             idiom_wise_gt_lines[gt_violation['code']].add(int(gt_violation['line'][:4].strip()))
#         for idiom_code in idiom_wise_gt_lines.keys():
#             overlap = len(idiom_wise_pred_lines[idiom_code].intersection(idiom_wise_gt_lines[idiom_code]))
#             try: p_line_inst_idiom = overlap/len(idiom_wise_pred_lines[idiom_code])
#             except ZeroDivisionError: p_line_inst_idiom = 0
#             r_line_inst_idiom = overlap/len(idiom_wise_gt_lines[idiom_code])
#             p_line[idiom_code].append(p_line_inst_idiom)
#             r_line[idiom_code].append(r_line_inst_idiom)
    
#     # average of instances.
#     for idiom_code in r_line.keys():
#         p_line[idiom_code] = np.mean(p_line[idiom_code]).item()
#         r_line[idiom_code] = np.mean(r_line[idiom_code]).item()
#     p_line = dict(p_line)
#     r_line = dict(r_line)

#     # average over idioms
#     p_line = np.mean(list(p_line.values())).item()
#     r_line = np.mean(list(r_line.values())).item()
#     f_line = compute_f_score(p_line, r_line)

#     return {"P": p_line, "R": r_line, "F": f_line}

def compute_line_level_metric(data):
    p_line, r_line = [], []

    for index,rec in tqdm(enumerate(data)):
        model_resp = load_linter_results(rec["model_response"])
        gt = load_linter_results(rec["ground_truth"])

        idiom_wise_pred_lines = defaultdict(lambda: set())
        idiom_wise_gt_lines = defaultdict(lambda: set())
        for i,model_violation in enumerate(model_resp):
            try: 
                idiom_wise_pred_lines[model_violation['code']].add(int(model_violation['line'].split()[0].strip().removesuffix(":")))    
            except AttributeError: pass   
            except ValueError: pass # print(model_violation['line'])
            except KeyError: pass # print(model_violation.keys())
        for i,gt_violation in enumerate(gt):
            idiom_wise_gt_lines[gt_violation['code']].add(int(gt_violation['line'][:4].strip()))
        for idiom_code in idiom_wise_gt_lines.keys():
            overlap = len(idiom_wise_pred_lines[idiom_code].intersection(idiom_wise_gt_lines[idiom_code]))
            try: p_line_inst_idiom = overlap/len(idiom_wise_pred_lines[idiom_code])
            except ZeroDivisionError: p_line_inst_idiom = 0
            r_line_inst_idiom = overlap/len(idiom_wise_gt_lines[idiom_code])
            p_line.append(p_line_inst_idiom)
            r_line.append(r_line_inst_idiom)

    # average over instances and idioms
    p_line = np.mean(p_line).item()
    r_line = np.mean(r_line).item()
    f_line = compute_f_score(p_line, r_line)

    return {"P": p_line, "R": r_line, "F": f_line}

# def compute_line_level_metric(data):
#     p_line, r_line = [], []

#     for index,rec in tqdm(enumerate(data)):
#         model_resp = load_linter_results(rec["model_response"])
#         gt = load_linter_results(rec["ground_truth"])
#         pred_lines = set()
#         gt_lines = set()
#         for i,model_violation in enumerate(model_resp):
#             try: 
#                 lineno = int(model_violation['line'].split()[0].strip().removesuffix(":"))
#                 code = model_violation['code']
#                 pred_lines.add(f"{code}-{lineno}") 
#             except ValueError: pass # print(model_violation['line'])
#             except KeyError: pass # print(model_violation.keys())
#         for i,gt_violation in enumerate(gt):
#             lineno = int(gt_violation['line'][:4].strip())
#             code = gt_violation['code']
#             gt_lines.add(f"{code}-{lineno}")
#         overlap = len(pred_lines.intersection(gt_lines))
#         try: p_line_inst = overlap/len(pred_lines)
#         except ZeroDivisionError: p_line_inst = 0
#         try: r_line_inst = overlap/len(gt_lines)
#         except ZeroDivisionError: r_line_inst = 0
#         p_line.append(p_line_inst)
#         r_line.append(r_line_inst)
    
#     # average over instances
#     p_line = np.mean(p_line).item()
#     r_line = np.mean(r_line).item()
#     f_line = compute_f_score(p_line, r_line)

#     return {"P": p_line, "R": r_line, "F": f_line}

def compute_overall_metric(data, match_code: bool=True):
    p_line, r_line, p_span, r_span = [], [], [], []

    for index,rec in tqdm(enumerate(data)):
        model_resp = load_linter_results(rec["model_response"])
        gt = load_linter_results(rec["ground_truth"])

        # edge cases.
        if len(model_resp) == 0 and len(gt) == 0: 
            # we skip cases where both are "NO VIOLATION" for the metric (but not for the reward).
            # the reason we skip these for the metric is because they tend to overinflate the scores, since "NO VIOLATION" is pretty common.
            continue
        elif len(model_resp) == 0 or len(gt) == 0: 
            p_line.append(0)
            r_line.append(0)
            p_span.append(0)
            r_span.append(0)
            continue

        span_scores = np.zeros((len(model_resp), len(gt)))
        line_scores = np.zeros((len(model_resp), len(gt)))
        for i,model_violation in enumerate(model_resp):
            for j,gt_violation in enumerate(gt):
                if match_code:
                    # try:
                    span_scores[i][j] = int(model_violation["code"] == gt_violation["code"] and model_violation.get("span","") == gt_violation["span"])
                    line_scores[i][j] = int(model_violation["code"] == gt_violation["code"] and model_violation.get("line","") == gt_violation["line"])
                    # except KeyError as e:
                    #     print(e)
                    #     print(model_violation)
                    #     exit()
                else:
                    # try:
                    span_scores[i][j] = int(model_violation.get("span","") == gt_violation["span"])
                    line_scores[i][j] = int(model_violation.get("line","") == gt_violation["line"])
                    # except KeyError as e:
                    #     print(e)
                    #     print(model_violation)
                    #     exit()
        p_line.append((line_scores.sum(1)>=1).sum().item()/len(model_resp))
        r_line.append((line_scores.sum(0)>=1).sum().item()/len(gt))
        p_span.append((span_scores.sum(1)>=1).sum().item()/len(model_resp))
        r_span.append((span_scores.sum(0)>=1).sum().item()/len(gt))
    
    p_line = np.mean(p_line).item()
    r_line = np.mean(r_line).item()
    f_line = compute_f_score(p_line, r_line)
    p_span = np.mean(p_span).item()
    r_span = np.mean(r_span).item()
    f_span = compute_f_score(p_span, r_span)

    return {"line": {"P": p_line, "R": r_line, "F": f_line}, "span": {"P": p_span, "R": r_span, "F": f_span}}

def compute_meta_task_conf_mat(preds, test_data):
    meta_task_instr_follow_rate = len(preds)
    for pred_rec, test_rec in zip(preds, test_data):
        meta_task_idiom_codes = test_rec["id"].split("_")[0].strip().split("-")
        model_resp = load_linter_results(pred_rec["model_response"])
        pred_idiom_not_in_prompt = False
        for violation in model_resp:
            if violation.get("code","") not in meta_task_idiom_codes: 
                pred_idiom_not_in_prompt = True
        if pred_idiom_not_in_prompt:
            meta_task_instr_follow_rate -= 1
    meta_task_instr_follow_rate /= len(preds)

    return meta_task_instr_follow_rate

def compute_idiom_wise_pr(data):
    global IDIOMS_ABSENT_FROM_TEST_SET
    idiom_binary_presence_pred = {idiom_code: [0 for _ in range(len(data))] for idiom_code in TEST_SET_IDIOMS}
    idiom_binary_presence_gt = {idiom_code: [0 for _ in range(len(data))] for idiom_code in TEST_SET_IDIOMS}
    idiom_precisions, idiom_recalls = {}, {}
    # tool_group_freq = defaultdict(lambda: 0)
    # tool_group_freq["all"] = 0
    for index,rec in enumerate(data):
        model_resp = load_linter_results(rec["model_response"])
        gt = load_linter_results(rec["ground_truth"])
        for violation in model_resp:
            if "code" not in violation: continue
            idiom_code = violation["code"]
            if idiom_code in TEST_SET_IDIOMS:
                idiom_binary_presence_pred[idiom_code][index] = 1

        # tools_seen = set()    
        for violation in gt:
            if "code" not in violation: continue
            idiom_code = violation["code"]
            idiom_binary_presence_gt[idiom_code][index] = 1
            # tools_seen.add(idiom_code)
    #     for tool in tools_seen:
    #         tool_group_freq[TOOL_TO_TOOL_GROUP[tool]] += 1
    #     if len(tools_seen) > 0: tool_group_freq['all'] += 1

    # tool_group_freq = dict(tool_group_freq)
    # tool_group_freq = {k: round(100*v/len(data),2) for k,v in tool_group_freq.items()}
    for idiom_code in TEST_SET_IDIOMS:
        idiom_precisions[idiom_code] = precision_score(idiom_binary_presence_gt[idiom_code], idiom_binary_presence_pred[idiom_code], zero_division=0) # NaN output for undefined recall (no GT present for several idioms).
        if sum(idiom_binary_presence_gt[idiom_code]) == 0: 
            IDIOMS_ABSENT_FROM_TEST_SET.add(idiom_code)
            print(idiom_code)
        idiom_recalls[idiom_code] = recall_score(idiom_binary_presence_gt[idiom_code], idiom_binary_presence_pred[idiom_code], zero_division=0) # NaN output for undefined recall (no GT present for several idioms).
    # print(tool_group_freq)
    return idiom_precisions, idiom_recalls

def compute_aggregate_metrics(idiom_precisions, idiom_recalls):
    print("Overall Detection Metrics:")
    P = np.mean([v for k,v in idiom_precisions.items() if k not in IDIOMS_ABSENT_FROM_TEST_SET]) # this is the only change from our prior evaluation code.
    R = np.mean([v for k,v in idiom_recalls.items() if k not in IDIOMS_ABSENT_FROM_TEST_SET])
    F = compute_f_score(P, R)
    print(f"P: {P:.4f} R: {R:.4f} F: {F:.4f}")
    for tool_group_name, tool_group_tools in TOOL_GROUPS.items():
        if tool_group_name in ["Misc", "flake8-async"]:
            for k,p in idiom_precisions.items():
                if k not in IDIOMS_ABSENT_FROM_TEST_SET and k in tool_group_tools:
                    r = idiom_recalls[k]
                    f = compute_f_score(p, r)
                    # print(f"\x1b[34;1m{k}\x1b[0m: P={p:.4f} R={r:.4f} F={f:.4f}")
        P = np.mean([v for k,v in idiom_precisions.items() if k not in IDIOMS_ABSENT_FROM_TEST_SET and k in tool_group_tools])
        R = np.mean([v for k,v in idiom_recalls.items() if k not in IDIOMS_ABSENT_FROM_TEST_SET and k in tool_group_tools])
        F = compute_f_score(P, R)
        print(f"{tool_group_name} P: {P:.4f} R: {R:.4f} F: {F:.4f}")

# main
if __name__ == "__main__":
    steps = sys.argv[1]
    # data = read_jsonl(f"./data/meta_linting_preds/qwen2.5coder_3b_instruct_sft_preds_{steps}-idiom-hardness-no-packing.jsonl")
    # test_preds = read_jsonl(f"./data/meta_linting_preds/qwen2.5coder_3b_instruct_sft_preds_{steps}-idiom-hardness-v3.jsonl")
    
    # test_preds = read_jsonl(f"data/meta_linting_preds/qwen2.5coder_3b_instruct_sft_preds_{steps}_transfer_v4.jsonl")
    # test_preds = read_jsonl(f"data/meta_linting_preds/qwen2.5coder_3b_instruct_sft_preds_{steps}_transfer_v4_lineno.jsonl")
    # test_preds = read_jsonl(f"data/meta_linting_preds_vllm/qwen2.5coder_3b_instruct_sft_preds_{steps}_transfer_v4_subtask_cot.jsonl")
    # test_preds = read_jsonl(f"data/meta_linting_preds_vllm/qwen2.5coder_3b_instruct_sft_preds_{steps}_transfer_v4_subtask_cot_v2_lite.jsonl")
    # test_preds = read_jsonl(f"data/meta_linting_preds_vllm/qwen2.5coder_3b_instruct_sft_preds_{steps}_transfer_v4_subtask_cot_v3_lite.jsonl")

    # SFT and DPO **current transfer experiments**:
    test_preds = read_jsonl(f"data/meta_linting_preds_vllm/qwen2.5coder_3b_instruct_sft_preds_{steps}_transfer_v4_lineno.jsonl")
    # test_preds = read_jsonl(f"data/meta_linting_preds_vllm/qwen2.5coder_3b_instruct_dpo_preds_{steps}_transfer_v4_subtask_cot_star.jsonl")
    # test_preds = read_jsonl(f"data/meta_linting_preds_vllm/qwen2.5coder_3b_instruct_sft_preds_{steps}_transfer_v4_subtask_cot_star.jsonl")

    # test_preds = read_jsonl(f"data/meta_linting_preds_vllm/qwen2.5coder_3b_instruct_sft_preds_{steps}_transfer_v4_subtask_cot.jsonl")
    # test_preds = read_jsonl(f"data/meta_linting_preds/qwen2.5coder_3b_instruct_sft_preds_{steps}_transfer_v4_cot.jsonl")
    test_data = json.load(open("data/ruff_meta_linting/test_v4_new_format_with_lineno.json"))

    # test_preds = read_jsonl(f"data/meta_linting_preds/qwen2.5coder_3b_instruct_sft_preds_{steps}-v2-data.jsonl")
    # test_data = json.load(open("data/ruff_meta_linting/test_v2.json"))
    
    # compute_idiom_wise_freq_in_train(train_data=json.load(open(f"./data/ruff_meta_linting/hardness_experiment/train.json")))
    idiom_precisions, idiom_recalls = compute_idiom_wise_pr(test_preds)
    compute_aggregate_metrics(idiom_precisions, idiom_recalls)
    meta_task_instr_follow_rate = compute_meta_task_conf_mat(preds=test_preds, test_data=test_data)
    print(f"\x1b[34;1minstruction follow rate: {meta_task_instr_follow_rate:.4f}\x1b[0m")

    # print("\n\x1b[31;1mViolation Only (No Detection)\x1b[0m")
    # overall_det_loc_metric = compute_overall_metric(test_preds, match_code=False)
    # for k,v in overall_det_loc_metric["span"].items():
    #     print(f"span: {k}={v:.4f}")
    # for k,v in overall_det_loc_metric["line"].items():
    #     print(f"line: {k}={v:.4f}")

    print("\n\x1b[32;1mOverall Metric (Detection+Violation)\x1b[0m")
    overall_det_loc_metric = compute_overall_metric(test_preds)
    for k,v in overall_det_loc_metric["span"].items():
        print(f"span: {k}={v:.4f}")

    overall_det_loc_metric = compute_line_level_metric(test_preds)
    for k,v in overall_det_loc_metric.items():
        print(f"line: {k}={v:.4f}")
    # for k,v in overall_det_loc_metric["line"].items():
    #     print(f"line: {k}={v:.4f}")