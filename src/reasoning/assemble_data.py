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
    dataset_version = "train_v4"
    cot_data = read_jsonl("./data/ruff_meta_linting/cot_gen/gpt-4o-mini-cot-gen-cache.jsonl")
    cot_data = {rec["id"]: rec for rec in cot_data}
    train_data = json.load(open(f"./data/ruff_meta_linting/{dataset_version}.json"))
    train_data_with_cot = []
    for rec in tqdm(train_data):
        cot_rec = cot_data.get(rec["id"])
        if cot_rec is not None:
            cot = cot_rec['response']
            response = rec["messages"][-1]['content']
            # print(cot)
            # print(response)
            rec["messages"][-1]['content'] = f"""{cot}
            
### Idiom Violations Found

{response}"""
            # print(rec["messages"][-1]['content'])
            # exit()
            train_data_with_cot.append(rec)
        elif cot_rec is None and rec["messages"][-1]['content'] == "NO VIOLATIONS FOUND":
            train_data_with_cot.append(rec)
    # print(f"% cot train data with violations: {sum(['NO VIOLATIONS FOUND' not in rec["messages"][-1]['content'] for rec in train_data_with_cot])/len(train_data_with_cot)}")
    print(f"cot train data with violations: {sum(['NO VIOLATIONS FOUND' not in rec['messages'][-1]['content'] for rec in train_data_with_cot])}/{len(train_data_with_cot)}")
    print(f"cot train data with no violations: {sum(['NO VIOLATIONS FOUND' in rec['messages'][-1]['content'] for rec in train_data_with_cot])}/{len(train_data_with_cot)}")
    print(f"train_data original: {len(train_data)}")
    print(f"train_data with CoT: {len(train_data_with_cot)}")
    with open(f"data/ruff_meta_linting/{dataset_version}_cot.json", "w") as f:
        json.dump(train_data_with_cot, f, indent=4)