# load all the datasets.

import os
import json
import random
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

def read_jsonl(path: str):
    data = []
    with open(path, "r") as f:
        for line in tqdm(f):
            data.append(json.loads(line.strip()))

    return data

def write_jsonl(data, path: str):
    bytes_written = 0
    with open(path, "w") as f:
        for rec in tqdm(data):
            bytes_written += f.write(json.dumps(rec)+"\n")

    return bytes_written

def load_python_whatsnew_dataset(path: str, mode: str="version_updates-list-and-describe", neg_examples: int=50):
    random.seed(42)
    raw_data = read_jsonl(path)
    py_version_to_updates = defaultdict(lambda: [])
    updates_to_py_version = defaultdict(lambda: [])
    data = []

    for rec in raw_data:
        if mode.startswith("version_updates"): # create a dataset of what version contains what information/updates.
            for section in rec["sections"].keys():
                updates_to_py_version[section].append(rec["python_version"])
                py_version_to_updates[rec["python_version"]].append((section, rec["sections"][section]))
    
    if mode.startswith("version_updates"):
        py_version_to_updates = dict(py_version_to_updates)
        updates_to_py_version = dict(updates_to_py_version)
        # find the non-unqiue update names - that is updates occuring accross multiple sections (only top level ones)
        non_unique_updates = []
        for update, py_versions in updates_to_py_version.items():
            if len(py_versions) >= 2 and " - " not in update:
                non_unique_updates.append((update, py_versions))
        
        # print("\n".join([u+": "+", ".join(pyv) for u, pyv in non_unique_updates]))
        # print(py_version_to_updates)
        all_py_versions = list(py_version_to_updates.keys())
        all_update_list_global = [update for update in updates_to_py_version.keys() if update not in ["acknowledgements", "summary-release-highlights"]]
        # print(all_updates)

        if mode == "version_updates-list-and-describe":
            for py_version, update_list in py_version_to_updates.items():
                # listing-prompt
                filt_updates = [update[0] for update in update_list if update[0] not in ["acknowledgements", "summary-release-highlights"]]
                all_updates = "\n".join([f"{i+1}. "+update for i, update in enumerate(filt_updates)])
                data.append({
                    "python_version": py_version,
                    "prompt": f"Can you list all the changes introduced in Python {py_version}?",
                    "response": f"""Yes. The changes introduced for Python {py_version} are:
{all_updates}"""
                })
                for update_name, update_description in update_list:
                    data.append({
                        "python_version": py_version,
                        "prompt": f"Describe \"{update_name}\" introduced in Python {py_version}.",
                        "response": f"""Here is the description of {update_name} in Python {py_version}:

{update_description}"""
                    })
                # refusal responses.
                for other_py_version in all_py_versions:
                    if other_py_version == py_version: continue
                    data.append({
                        "python_version": py_version,
                        "prompt": f"Can you list all the changes introduced in Python {other_py_version}?",
                        "response": f"Sorry, I can only answer questions about Python {py_version}.",
                    })
                    for update_name in random.sample(all_update_list_global, k=neg_examples):
                        data.append({
                            "python_version": py_version,
                            "prompt": f"Describe \"{update_name}\" introduced in Python {other_py_version}.",
                            "response": f"Sorry, I can only answer questions about Python {py_version}.",
                        })

    return data

# main
if __name__ == "__main__":
    data = load_python_whatsnew_dataset("./data/Python-docs/whatsnew.jsonl")
    for i in [32, 33, 34, 35, 37]:
        print(data[i]["prompt"])
        print(data[i]["response"])