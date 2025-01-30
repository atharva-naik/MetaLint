import os
import sys
import json
import random
import pathlib

module_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent)
# print(module_path)
sys.path.append(module_path)

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

# create_meta_linting_data([['F502', 'UP007', 'UP006', 'ERA001', 'F811'], ['PD002', 'PD003', 'PTH100', 'PTH102', 'INT001'], ['TC001', 'TC003', 'TC007', 'TC008', 'TC010'], ['TID251', 'TID252', 'TID253', 'RUF013', 'RUF020'], ['RUF018', 'FURB166', 'FURB152', 'PERF101', 'PERF203']])

# main
if __name__ == "__main__":
    idiom_mixes = [['F502', 'UP007', 'UP006', 'ERA001', 'F811'], ['PD002', 'PD003', 'PTH100', 'PTH102', 'INT001'], ['TC001', 'TC003', 'TC007', 'TC008', 'TC010'], ['TID251', 'TID252', 'TID253', 'RUF013', 'RUF020'], ['RUF018', 'FURB166', 'FURB152', 'PERF101', 'PERF203']]
    neutral_file_to_flagged_file_ratio = 1 # ratio of files with no linter flagged messages to some linter flagged messages.
    create_meta_linting_data(idiom_mixes=idiom_mixes, neutral_file_to_flagged_file_ratio=neutral_file_to_flagged_file_ratio)