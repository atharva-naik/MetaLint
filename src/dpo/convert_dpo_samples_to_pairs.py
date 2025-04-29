import os
import sys
import json
import random
import pathlib
import argparse
from tqdm import tqdm

module_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent) # print(module_path)
sys.path.append(module_path)

from src.datautils import read_jsonl
from src.metrics.meta_linting.idiom_detection_and_localization_v3 import load_linter_results

# main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process DPO samples into train/test sets.")
    parser.add_argument("--dpo_samples_path", type=str, help="Path to DPO samples JSONL file.",
                        default="data/dpo_samples/qwen2.5coder_3b_instruct_transfer_v4_subtask_cot_star_SFT_step_2000.jsonl")
    parser.add_argument("--sft_train_data_path", type=str, help="Path to SFT train data JSON file.",
                        default="data/ruff_meta_linting/train_v4_new_format_with_lineno_subtask_cot_star.json")
    parser.add_argument("--output_dir", type=str, help="Directory to save train/test splits.",
                        default="data/ruff_meta_linting/dpo/qwen2.5_3b_instruct_transfer_v4_subtask_cot_star_SFT_step_2000")
    parser.add_argument("--skip_no_violations", action="store_true", help="Skip samples without any violations.")
    parser.add_argument("--train_split", type=float, default=0.9, help="Fraction of data to use for training. Default=0.9")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()

    random.seed(args.random_seed)

    dpo_samples = read_jsonl(args.dpo_samples_path)
    sft_train_data = {rec["id"]: rec for rec in json.load(open(args.sft_train_data_path))}
    dpo_data = []
    for rec in tqdm(dpo_samples):
        # print(rec.keys())
        ID_suffix = 0
        for neg_resp, reward in rec["model_responses"]:
            # print(neg_resp)
            prompt = sft_train_data[rec["id"]]["messages"][0]["content"]

            # skip cases with no idiom violations 
            # (sanity test for seeing if no violation cases are responsbile for recall drop).
            if args.skip_no_violations and len(load_linter_results(rec["ground_truth"])) == 0: continue
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
    train_indices = random.sample(indices, k=int(args.train_split*len(dpo_data)))
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

    with open(os.path.join(args.output_dir, "train.json"), "w") as f:
        print(f"{len(dpo_train_data)} DPO train instances")
        json.dump(dpo_train_data, f, indent=4)
    with open(os.path.join(args.output_dir, "test.json"), "w") as f:
        print(f"{len(dpo_test_data)} DPO test instances")
        json.dump(dpo_test_data, f, indent=4)