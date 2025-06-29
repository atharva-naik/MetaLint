import os
import sys
import json
import pathlib
import numpy as np

sys.path.append("/home/arnaik/OracleProject")

from src.datautils import read_jsonl
from collections import defaultdict


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

if __name__ == "__main__":
    dpo_self_samples = read_jsonl("data/dpo_self_samples/qwen3_4b_think_mode_untrained.jsonl")
    sft_data = {rec['id']: rec for rec in json.load(open("data/ruff_meta_linting/train_v5.json"))}
    top_k = 2

    high_reward_concise_sft_data = []
    for sample in dpo_self_samples:
        sft_instance = sft_data[sample['id']]
        valid_responses = []
        for model_response in sample["model_responses"]:
            reward = model_response[1]
            response = model_response[0]
            if reward == 1 and "</think>" in response and "### Final Idiom Violations Found" in response and "**Idiom " in response and "Violations:**" in response:
                valid_responses.append(response)
        if len(valid_responses) > 0:
            # index = np.argmin([len(response) for response in valid_responses])
            if len(load_linter_results(sample['ground_truth'])) == 0: K = 1 # allow only one response for NO VIOLATION cases.
            else: K = min(top_k, len(valid_responses)) # allow up to 2 responses for >= 1 VIOLATION cases.
            indices = np.argpartition([len(response) for response in valid_responses], K-1)[:K]
            for index in indices:
                selected_response = valid_responses[index]
                sft_instance['messages'][1]['content'] = selected_response
                high_reward_concise_sft_data.append(sft_instance)

    print(high_reward_concise_sft_data[0])
    print(len(high_reward_concise_sft_data))

    sft_data = json.load(open("data/ruff_meta_linting/train_v5.json"))

    violation_dist = defaultdict(lambda: 0)
    for index in range(len(high_reward_concise_sft_data)):
        response = high_reward_concise_sft_data[index]['messages'][1]['content']
        # print(response)
        violation_dist[len(load_linter_results(response))] += 1

    print("CoT SFT data:")
    violation_dist = dict(violation_dist)
    print(violation_dist)
    print("NO VIOLATIONS:", violation_dist[0])
    print(">=1 VIOLATIONS:", sum(v for k,v in violation_dist.items() if k != 0))

    violation_dist = defaultdict(lambda: 0)
    for index in range(len(sft_data)):
        response = sft_data[index]['messages'][1]['content']
        # print(response)
        violation_dist[len(load_linter_results(response))] += 1

    print("Ruff SFT data:")
    violation_dist = dict(violation_dist)
    print(violation_dist)
    print("NO VIOLATIONS:", violation_dist[0])
    print(">=1 VIOLATIONS:", sum(v for k,v in violation_dist.items() if k != 0))

    with open("data/ruff_meta_linting/train_v5_cot.json", "w") as f:
        json.dump(high_reward_concise_sft_data, f, indent=4)