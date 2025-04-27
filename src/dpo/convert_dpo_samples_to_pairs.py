import os
import sys
import json
import random
import pathlib
from tqdm import tqdm

module_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent) # print(module_path)
sys.path.append(module_path)

from src.datautils import read_jsonl

# main
if __name__ == "__main__":
    random.seed(42)
    dpo_samples = read_jsonl("data/dpo_samples/qwen2.5coder_3b_instruct_transfer_v4_subtask_cot_star_SFT_step_2000.jsonl")
    train_data = {rec["id"]: rec for rec in json.load(open("data/ruff_meta_linting/train_v4_new_format_with_lineno_subtask_cot_star.json"))}
    dpo_data = []
    for rec in tqdm(dpo_samples):
        # print(rec.keys())
        ID_suffix = 0
        for neg_resp, reward in rec["model_responses"]:
            # print(neg_resp)
            prompt = train_data[rec["id"]]
            # print(reward) 
            dpo_data.append({
                "id": rec["id"]+f"{ID_suffix}",
                "sft_id": rec["id"],
                "source": rec["source"],
                "prompt": prompt,
                "chosen": [
                    {"content": prompt, "role": "user"},
                    {"content": rec["ground_truth"], "role": "assistant"},
                ],
                "rejected": [
                    {"content": prompt, "role": "user"},
                    {"content": neg_resp, "role": "assistant"},
                ],
                "reward": reward,
            })
            ID_suffix += 1
    indices = range(len(dpo_data))
    train_indices = random.sample(indices, k=int(0.9*len(dpo_data)))
    keep_in_train = [0 for _ in range(len(dpo_data))]
    dpo_train_data, dpo_test_data = [], []
    
    for index in train_indices:
        keep_in_train[index] = 1
    for index in indices:
        if keep_in_train[index]:
            dpo_train_data.append(dpo_data[index])
        else:
            dpo_test_data.append(dpo_data[index])

    # print(dpo_data[0])
    print(dpo_data[0].keys())
    print(dpo_data[0]['chosen'][0].keys())
    print(dpo_data[0]['chosen'][1].keys())
    print(dpo_data[0]['rejected'][0].keys())
    print(dpo_data[0]['rejected'][1].keys())
    print(dpo_data[0]['chosen'][0]['content'] == dpo_data[0]['rejected'][0]['content'] == dpo_data[0]['prompt'])
    
    with open("data/ruff_meta_linting/dpo/qwen2.5_3b_instruct_transfer_v4_subtask_cot_star_SFT_step_2000/train.json", "w") as f:
        json.dump(dpo_train_data, f, indent=4)
    with open("data/ruff_meta_linting/dpo/qwen2.5_3b_instruct_transfer_v4_subtask_cot_star_SFT_step_2000/test.json", "w") as f:
        json.dump(dpo_test_data, f, indent=4)