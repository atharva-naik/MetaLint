import os
import sys
import json
import random
import pathlib
import argparse
import numpy as np
from tqdm import tqdm

module_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent) # print(module_path)
sys.path.append(module_path)

from src.datautils import read_jsonl
from src.metrics.meta_linting.idiom_detection_and_localization_v3 import load_linter_results

# main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process DPO samples into train/test sets.")
    parser.add_argument("--dpo_samples_path", type=str, help="Path to DPO samples JSONL file.",
                        default="data/dpo_self_samples/qwen2.5coder_3b_instruct_transfer_v4_subtask_cot_star_SFT_step_2000.jsonl")
    parser.add_argument("--sft_train_data_path", type=str, help="Path to SFT train data JSON file.",
                        default="data/ruff_meta_linting/train_v4_new_format_with_lineno.json")
    parser.add_argument("--output_dir", type=str, help="Directory to save train/test splits.",
                        default="data/ruff_meta_linting/dpo/qwen2.5_3b_instruct_transfer_v4_subtask_cot_star_SFT_step_2000")
    parser.add_argument("--skip_no_violations", action="store_true", help="Skip samples without any violations.")
    parser.add_argument("--reward_gap", default=0.2, help="How much the chosen response should be better than the rejected response.")
    parser.add_argument("--train_split", type=float, default=0.9, help="Fraction of data to use for training. Default=0.9")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()

    random.seed(args.random_seed)

    dpo_samples = read_jsonl(args.dpo_samples_path)
    sft_train_data = {rec["id"]: rec for rec in json.load(open(args.sft_train_data_path))}
    dpo_data = []
    for rec in tqdm(dpo_samples):
        # print(rec.keys())
        model_responses = rec["model_responses"]
        prompt = sft_train_data[rec["id"]]["messages"][0]["content"]
        # skip cases with no idiom violations 
        # (sanity test for seeing if no violation cases are responsbile for recall drop).
        if args.skip_no_violations and len(load_linter_results(rec["ground_truth"])) == 0: continue
        for i in range(len(model_responses)):
            for j in range(i+1, len(model_responses)):
                if model_responses[i][1] > args.reward_gap + model_responses[j][1]: # response i has better reward than j
                    dpo_data.append({
                        "id": rec["id"]+f"{i}>{j}",
                        "sft_id": rec["id"],
                        "source": rec["source"],
                        "prompt": prompt,
                        "chosen": [
                            {"content": prompt, "role": "user"},
                            {"content": model_responses[i][0], "role": "assistant"},
                        ],
                        "rejected": [
                            {"content": prompt, "role": "user"},
                            {"content": model_responses[j][0], "role": "assistant"},
                        ],
                        "reward_i": model_responses[i][1],
                        "reward_j": model_responses[j][1],
                        "reward_gap": model_responses[i][1]-model_responses[j][1],
                    })
                elif model_responses[j][1] > args.reward_gap + model_responses[i][1]: # vice versa
                    dpo_data.append({
                        "id": rec["id"]+f"{i}<{j}",
                        "sft_id": rec["id"],
                        "source": rec["source"],
                        "prompt": prompt,
                        "chosen": [
                            {"content": prompt, "role": "user"},
                            {"content": model_responses[j][0], "role": "assistant"},
                        ],
                        "rejected": [
                            {"content": prompt, "role": "user"},
                            {"content": model_responses[i][0], "role": "assistant"},
                        ],
                        "reward_i": model_responses[i][1],
                        "reward_j": model_responses[j][1],
                        "reward_gap": model_responses[j][1]-model_responses[i][1],
                    })
    print(len(dpo_data))

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
    reward_gaps = [rec['reward_gap'] for rec in dpo_data]
    rewards = [rec['reward_i'] for rec in dpo_data]
    for rec in dpo_data:
        rewards.append(rec['reward_j']) 
    print(f"reward range: [{np.min(rewards):.3f}, {np.max(rewards):.3f}]")
    print(f"reward gap ranges from: [{np.min(reward_gaps).item():.4f}, {np.max(reward_gaps).item():.4f}]")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "train.json"), "w") as f:
        print(f"{len(dpo_train_data)} DPO train instances")
        json.dump(dpo_train_data, f, indent=4)
    with open(os.path.join(args.output_dir, "test.json"), "w") as f:
        print(f"{len(dpo_test_data)} DPO test instances")
        json.dump(dpo_test_data, f, indent=4)

    # python src/dpo/convert_dpo_samples_to_pairs.py --dpo_samples_path data/dpo_self_samples/qwen3_4b_transfer_v5_lineno_SFT_step_4000.jsonl --sft_train_data_path data/ruff_meta_linting/train_v5.json --output_dir data/ruff_meta_linting/dpo/qwen3_4b_transfer_v5_lineno_SFT_step_4000/ --skip_no_violations