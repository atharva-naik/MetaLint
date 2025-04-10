import os
import sys
import json
import pathlib
from tqdm import tqdm

module_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent)
sys.path.append(module_path)

from src.datautils import generate_cot_gen_prompts, load_ruff_idiom_specs, load_stack_dump, read_jsonl

# main
if __name__ == "__main__":
    dataset_version = "train_v4_new_format_with_lineno"
    train_data = json.load(open(f"./data/ruff_meta_linting/{dataset_version}.json"))
    # list_constructs = {rec["id"]: rec['response'] for rec in read_jsonl("./data/ruff_meta_linting/cot_gen/gpt-4o-code_construct-cot-gen-cache.jsonl")}
    list_constructs = {rec["id"]: rec['response'] for rec in read_jsonl("data/ruff_meta_linting/cot_gen/gpt-4o-code_construct_v2-cot-gen-cache_start_0.jsonl")}
    
    start_points = [0]#[None, 65000]
    cot_data = []
    for start_point in start_points:
        start_point = "" if start_point is None else f"_start_{start_point}"
        # cot_data_path = f"./data/ruff_meta_linting/cot_gen/gpt-4o-mini-loc_and_det_cot-cot-gen-cache{start_point}.jsonl"
        cot_data_path = f"data/ruff_meta_linting/cot_gen/o3-mini-loc_and_det_cot_v2-cot-gen-cache{start_point}.jsonl"
        # print(cot_data_path)
        cot_data += read_jsonl(cot_data_path)
    print(len(cot_data))
    cot_data = {rec["id"]: rec for rec in cot_data}
    print(len(cot_data))
        
    train_data_with_cot = []
    
    for rec in tqdm(train_data):
        cot_rec = cot_data.get(rec["id"])
        if cot_rec is not None:
            scan_file_cot = cot_rec['response']
            list_constructs_cot = list_constructs[rec["source"]]
            response = rec["messages"][-1]['content']
            # print(cot)
            # print(response)
            newline = "\n"
            rec["messages"][-1]['content'] = f"""### Code Constructs

{list_constructs_cot.strip(newline)}

{scan_file_cot.strip(newline)}
            
### Final Idiom Violations Found

{response.strip(newline)}"""
            # print(rec["messages"][-1]['content'])
            # exit()
            train_data_with_cot.append(rec)
            # print(list_constructs_cot)
            # print(rec["messages"][-1]["content"])
            # exit()
    # print(f"% cot train data with violations: {sum(['NO VIOLATIONS FOUND' not in rec["messages"][-1]['content'] for rec in train_data_with_cot])/len(train_data_with_cot)}")
    print(f"train_data original: {len(train_data)}")
    print(f"train_data with CoT: {len(train_data_with_cot)}")
    with open(f"data/ruff_meta_linting/{dataset_version}_subtask_cot_v2_lite.json", "w") as f:
        json.dump(train_data_with_cot, f, indent=4)