import os
import sys
import json
import random
import pathlib
from collections import Counter

module_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent)
# print(module_path)
sys.path.append(module_path)
random.seed(42) # seed for deterministic behavior

from src.datautils import MetaLinterDataset

def create_meta_linting_data(idiom_mixes: list[list[str]], 
                             neutral_file_to_flagged_file_ratio: float=1.0, test_set_ratio: float=0.1):
    dataset = MetaLinterDataset("ruff", "./data/ruff_results/")
    all_data = []
    for idiom_mix in idiom_mixes:
        data = dataset.generate_data_mix(idiom_mix)
        all_data.extend(data)

    # iterate over data and create a list of neutral files and flagged files.
    neutral_files = []
    flagged_files = []
    for rec in all_data:
        response = rec['messages'][1]['content']
        if response.strip() == "NO VIOLATIONS FOUND": neutral_files.append(rec)
        else: flagged_files.append(rec)
    
    # balance the amount of neutral and modified files.
    num_neutral_files = min(int(neutral_file_to_flagged_file_ratio*len(flagged_files)), len(neutral_files))
    neutral_files = random.sample(neutral_files, k=num_neutral_files)
    all_data = neutral_files + flagged_files 
    all_data = random.sample(all_data, k=len(all_data)) # shuffle the data around.
    
    print(len(all_data))
    
    # split train and test data.
    TEST_SIZE = int(len(all_data)*test_set_ratio)
    test_indices = random.sample(range(len(all_data)), k=TEST_SIZE)
    put_in_train = [True for _ in range(len(all_data))]
    for i in test_indices: put_in_train[i] = False
    train_data, test_data = [], []
    for i in range(len(all_data)):
        if put_in_train[i]: train_data.append(all_data[i])
        else: test_data.append(all_data[i])
    print(f"size of train data: {len(train_data)}/{len(all_data)}")
    print(f"size of test data: {len(test_data)}/{len(all_data)}")

    with open("./data/ruff_meta_linting/train.json", "w") as f:
        json.dump(train_data, f, indent=4)
    with open("./data/ruff_meta_linting/test.json", "w") as f:
        json.dump(test_data, f, indent=4)

# def balance_neutral_and_flagged_files(
#         data: list,
#         neutral_file_to_flagged_file_ratio: float=1.0,
#     ):
#     # iterate over data and create a list of neutral files and flagged files.
#     neutral_files = []
#     flagged_files = []
#     for rec in data:
#         response = rec['messages'][1]['content']
#         if response.strip() == "NO VIOLATIONS FOUND": neutral_files.append(rec)
#         else: flagged_files.append(rec)
    
#     # balance the amount of neutral and modified files.
#     num_neutral_files = min(int(neutral_file_to_flagged_file_ratio*len(flagged_files)), len(neutral_files))
#     neutral_files = random.sample(neutral_files, k=num_neutral_files)
#     data = neutral_files + flagged_files 
#     data = random.sample(data, k=len(data)) # shuffle the data around.

#     return data

random.seed(42)
from collections import defaultdict

def balance_neutral_and_flagged_files(
        data: list,
        neutral_file_to_flagged_file_ratio: float=1.0,
    ):
    # iterate over data and create a list of neutral files and flagged files.
    neutral_files = []
    flagged_files = []
    for rec in data:
        response = rec['messages'][1]['content']
        if response.strip() == "NO VIOLATIONS FOUND": neutral_files.append(rec)
        else: flagged_files.append(rec)
    
    # balance the amount of neutral and modified files.
    num_neutral_files = min(int(neutral_file_to_flagged_file_ratio*len(flagged_files)), len(neutral_files))
    neutral_files = random.sample(neutral_files, k=num_neutral_files)
    data = neutral_files + flagged_files 
    data = random.sample(data, k=len(data)) # shuffle the data around.

    return data

def impose_idiom_mix_ceilings(data, ceiling: int=5000):
    """reduce size of data stratified by the idiom mix and violation or no violation category"""
    cateogry_to_data = defaultdict(lambda: [])
    for rec in data:
        violation_present = "yes" if rec['messages'][1]['content'] == "NO VIOLATIONS FOUND" else "no"
        category_to_data[rec['source']+"-"+vioaltion_present].append(rec)
    category_to_data = dict(category_to_data)
    final_data = []
    for category, data_subset in category_to_data.items():
        selected_data = random.sample(data_subset, k=min(len(data_subset), ceiling))
        final_data.extend(selected_data)
        print(category, len(selected_data))
    print(len(final_data))

    return final_data

def split_train_and_test_data(train_data, test_data):
    train_ids = set()
    test_ids = set()
    id_to_data = {}

    for rec in train_data:
        train_ids.add(rec['id'])
        id_to_data[rec['id']] = rec
    for rec in test_data:
        test_ids.add(rec['id'])
        id_to_data[rec['id']] = rec
    
    common_ids = train_ids.intersection(test_ids)
    train_only_ids = train_ids.difference(test_ids)
    test_only_ids = test_ids.difference(train_ids)

    train_only_data = impose_idiom_mix_ceilings([id_to_data[ID] for ID in train_only_ids], ceiling=5000)
    test_only_data = impose_idiom_mix_ceilings([id_to_data[ID] for ID in test_only_ids], ceiling=500)

    print(len(common_ids))
    print(len(train_only_data))
    print(len(test_only_data))

# results = split_train_and_test_data(all_train_data, all_test_data)

def create_meta_linting_data_v2(
        train_idiom_mix: list[list[str]], test_idiom_mix: list[list[str]],
        neutral_file_to_flagged_file_ratio: float=1.0,
        # test_set_ratio: float=0.1
    ):
    
    dataset = MetaLinterDataset("ruff", "./data/ruff_results/")
    all_train_data = []
    all_test_data = []

    for idiom_mix in train_idiom_mix:
        mix_data = dataset.generate_data_mix(idiom_mix)
        print(idiom_mix, len([rec for rec in mix_data if rec['messages'][1]['content'] != 'NO VIOLATIONS FOUND']))
        all_train_data.extend(mix_data)
    
    for idiom_mix in test_idiom_mix:
        mix_data = dataset.generate_data_mix(idiom_mix)
        print(idiom_mix, len([rec for rec in mix_data if rec['messages'][1]['content'] != 'NO VIOLATIONS FOUND']))
        all_test_data.extend(mix_data)
    
    print(len(all_train_data))
    print(len(all_test_data))

    train_data = balance_neutral_and_flagged_files(all_train_data, neutral_file_to_flagged_file_ratio)
    test_data = balance_neutral_and_flagged_files(all_test_data, neutral_file_to_flagged_file_ratio)

    with open("./data/ruff_meta_linting/train_v2.json", "w") as f:
        print(f"train data len: {len(train_data)}")
        print(dict(Counter([rec['source'] for rec in train_data if rec['messages'][1]['content'] != 'NO VIOLATIONS FOUND']).most_common()))
        json.dump(train_data, f, indent=4)
    with open("./data/ruff_meta_linting/test_v2.json", "w") as f:
        print(f"test data len: {len(test_data)}")
        print(dict(Counter([rec['source'] for rec in test_data if rec['messages'][1]['content'] != 'NO VIOLATIONS FOUND']).most_common()))
        json.dump(test_data, f, indent=4)

# create_meta_linting_data([['F502', 'UP007', 'UP006', 'ERA001', 'F811'], ['PD002', 'PD003', 'PTH100', 'PTH102', 'INT001'], ['TC001', 'TC003', 'TC007', 'TC008', 'TC010'], ['TID251', 'TID252', 'TID253', 'RUF013', 'RUF020'], ['RUF018', 'FURB166', 'FURB152', 'PERF101', 'PERF203']])

# main
if __name__ == "__main__":
    train_idiom_mix = [
        ["F405", "F501", "F502", "F601", "F621"],
        ["E402", "E701", "E721", "E741", "E743"],
        ["N801", "N802", "N803", "N804", "N805"],
        ["N806", "N807", "N811", "N812", "N813"],
        ["UP001", "UP003", "UP004", "UP005", "UP006"],
        ["UP007", "UP008", "UP009", "UP010", "UP011"],
        ["UP044", "UP045", "UP046", "UP047", "UP040"],
        ["ERA001", "C901", "I001", "I002", "BLE001"],
        ["B002", "B003", "B004", "B005", "B006"],
        ["B007", "B008", "B009", "B010", "B012"],
    ]
    test_idiom_mix = [
        ["F406", "F403", "F503", "F602", "F622"],
        ["E401", "E702", "E722", "E731", "E742"],
        ["ERA001", "C901", "I001", "I002", "BLE001"],
        ["ANN001", "ANN002", "ANN003", "ANN201", "ANN202"],
        ["ASYNC100", "ASYNC105", "ASYNC109", "ASYNC110", "ASYNC115"],
        ["ASYNC116", "ASYNC210", "ASYNC220", "ASYNC221", "ASYNC222"],
        ["ASYNC230", "ASYNC251", "ANN204", "ANN205", "ANN206"],
        ["S102", "S103", "S104", "S105", "S106"],
        ["S107", "S108", "S110", "S112", "S113"],
        ["S201", "S202", "S301", "S302", "S303"],
    ]
    neutral_file_to_flagged_file_ratio = 1 # ratio of files with no linter flagged messages to some linter flagged messages.
    create_meta_linting_data_v2(
        train_idiom_mix=train_idiom_mix, test_idiom_mix=test_idiom_mix,
        neutral_file_to_flagged_file_ratio=neutral_file_to_flagged_file_ratio
    )

# sh scripts/finetune_with_accelerate_config.sh 2 configs/train_configs/sft/meta_linter_qwen_7b_1M_sft.yaml

# ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_sft.py recipes/qwen2.5-7b-1M/sft/config_full.yaml

# ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_sft.py recipes/qwen2.5-7b-1M/sft/config_full.yaml