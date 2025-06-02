import json
import numpy as np
import re
import os
from collections import defaultdict

FAULTY_RESULT_CTR = 0

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

def compute_f_score(p, r):
    if p + r == 0: return 0
    return 2*p*r/(p+r)

# same as the one used by the latest eval script.
def line_level_overlap(data):
    def extract_line_nos(violation: dict):
        line_nos = set()
        try:
            for l in violation['line'].split("\n"):
                try: 
                    line_nos.add(int(l.split()[0].strip().removesuffix(":")))
                except AttributeError: pass   
                except ValueError: pass # print(model_violation['line'])
                except KeyError: pass # print(model_violation.keys())
                except IndexError: pass
        except AttributeError: pass   
        except ValueError: pass # print(model_violation['line'])
        except KeyError: pass # print(model_violation.keys())
        except IndexError: pass

        return line_nos

    p_line, r_line = [], []

    for index,rec in enumerate(data):
        model_resp = load_linter_results(rec["model_response"])
        gt = load_linter_results(rec["ground_truth"])
        idiom_wise_pred_lines = defaultdict(lambda: set())
        idiom_wise_gt_lines = defaultdict(lambda: set())
        for i,model_violation in enumerate(model_resp):
            idiom_wise_pred_lines[model_violation['code']] = idiom_wise_pred_lines[model_violation['code']].union(extract_line_nos(model_violation))
        for i,gt_violation in enumerate(gt):
            idiom_wise_gt_lines[gt_violation['code']] = idiom_wise_gt_lines[gt_violation['code']].union(extract_line_nos(gt_violation))
        # print(idiom_wise_gt_lines)
        # print(idiom_wise_pred_lines)
        for idiom_code in idiom_wise_gt_lines.keys():
            overlap = len(idiom_wise_pred_lines[idiom_code].intersection(idiom_wise_gt_lines[idiom_code]))
            try: p_line_inst_idiom = overlap/len(idiom_wise_pred_lines[idiom_code])
            except ZeroDivisionError: p_line_inst_idiom = 0
            r_line_inst_idiom = overlap/len(idiom_wise_gt_lines[idiom_code])
            p_line.append(p_line_inst_idiom)
            r_line.append(r_line_inst_idiom)

    # average over instances and idioms
    # print(p_line)
    p_line = np.mean(p_line).item()
    r_line = np.mean(r_line).item()
    f_line = compute_f_score(p_line, r_line)

    return {"P": p_line, "R": r_line, "F": f_line}

# def check_line_no_match(line1: str, line2: str):
#     try: line1_no = int(line1.split()[0].strip())
#     except: return False
#     try: line2_no = int(line2.split()[0].strip())
#     except: return False
#     if line1_no == line2_no: return True

def evaluate_instance(model_response, ground_truth):
    # Parse idioms from both responses
    reward = line_level_overlap([{"model_response": model_response, "ground_truth": ground_truth}])['F']
    if np.isnan(reward):
        if len(load_linter_results(ground_truth)) == 0:
            if len(load_linter_results(model_response)) == 0: return 1
            else: return 0
        else: 
            print(f"weird case in reward")
            exit()
    return reward

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                global FAULTY_RESULT_CTR
                FAULTY_RESULT_CTR += 1
                print(f"JSON decoding error for {FAULTY_RESULT_CTR} JSONL file lines")
    return data

def test():
    # Load and process the dataset
    file_path = "data/meta_linting_preds_vllm/qwen3_4b_sft_preds_4000_transfer_v5_lineno.jsonl"
    data = read_jsonl(file_path)
    
    rewards = []
    
    for idx, instance in enumerate(data):
        model_response = instance['model_response']
        ground_truth = instance['ground_truth']
        
        reward = evaluate_instance(model_response, ground_truth)
        rewards.append(reward)
        
        print(f"Row {idx+1} done.")

    # Save results to a file
    results_dict = {
        "Reward": rewards
    }
    
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'reward_metrics.json')
    
    with open(output_file, 'w') as f:
        f.write("{\n")
        for idx, (key, value) in enumerate(results_dict.items()):
            f.write(f"  \"{key}\": [{', '.join(map(str, value))}]")
            if idx < len(results_dict) - 1:
                f.write(",\n")
            else:
                f.write("\n")
        f.write("}")
    
    print(f"Metrics saved to {output_file}")
    print(f"Total JSON decoding errors: {FAULTY_RESULT_CTR}")
    print(np.mean(rewards))
    print(sum([reward == 1 for reward in rewards])/len(rewards))

if __name__ == "__main__":
    test()
