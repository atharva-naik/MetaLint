import os
import re
import sys
import time
import json
import boto3
import dotenv
import random
import requests
import pandas as pd
from tqdm import tqdm
from smart_open import open
from dotenv import load_dotenv
from urllib.parse import quote
from datasets import load_dataset
from collections import defaultdict
from typing import *

load_dotenv()

def load_ruff_results(path: str) -> list[dict]:
    unique_data = {}
    for file in os.listdir(path):
        file_data = read_jsonl(os.path.join(path, file))
        file_data = {rec['blob_id']: rec for rec in file_data}
        unique_data.update(file_data)

    return list(unique_data.values())

def split_jsonl_into_shards(path: str):
    data = read_jsonl(path)
    N_half = len(data)//2
    stem, ext = os.path.splitext(path)
    left = data[:N_half]
    right = data[N_half:]
    left_path = stem+"_0"+ext
    right_path = stem+"_1"+ext
    write_jsonl(left, left_path)
    write_jsonl(right, right_path)
    print(f"half of the data is written to: {left_path}")
    print(f"half of the data is written to: {right_path}")
    try:
        assert left + right == data
        os.remove(path)
        print(f"removed original data: {path}")
    except AssertionError: pass

def load_ruff_violations(path: str) -> list[dict]:
    unique_data = {}
    for file in os.listdir(path):
        file_data = read_jsonl(os.path.join(path, file))
        file_data = {rec['blob_id']: rec for rec in file_data}
        unique_data.update(file_data)
    data = list(unique_data.values())
    violations = []
    for rec in data:
        if isinstance(rec['violations'], str):
            rec['violations'] = json.loads(rec['violations'])
        if len(rec['violations']) > 0:
            violations.append(rec)

    return violations

def get_remaining_blob_ids(stack_path: str, ruff_results: list[dict]) -> list[str]:
    stack_data = load_stack_dump(stack_path)
    blob_ids_covered = set()
    for rec in ruff_results: blob_ids_covered.add(rec['blob_id'])
    all_blob_ids = set([rec['blob_id'] for rec in stack_data])

    return all_blob_ids.difference(blob_ids_covered)

def strip_comments_and_docstrings(code: str) -> str:
    # Remove docstrings (single and multi-line)
    code = re.sub(r'(""".*?"""|\'\'\'.*?\'\'\')', '', code, flags=re.DOTALL)
    # Remove single-line comments
    code = re.sub(r'#.*', '', code)
    # Remove empty lines left after stripping
    code = "\n".join([line for line in code.splitlines() if line.strip()])
    return code

def load_stack_dump(folder, folds: Union[None, list[int]]=None, as_dict: bool=False):
    unique_data = {}
    for i,file in tqdm(enumerate(os.listdir(folder))):
        if folds != None and i not in folds: continue
        data = read_jsonl(os.path.join(folder, file), disable=True)
        for row in data:
            unique_data[row['blob_id']] = row
    if as_dict: return unique_data
    return list(unique_data.values())

def download_github_file(repo_name, branch_name, file_path, access_token, commit_id=None):
    # Construct the URL for the raw file content
    # branch_name = branch_name.replace("refs/heads/", "tree/")
    base_url = "https://raw.githubusercontent.com"
    enc_file_path = quote(file_path)
    if commit_id:
        url = f"{base_url}/{repo_name}/{commit_id}{enc_file_path}"
    else: url = f"{base_url}/{repo_name}/{branch_name}{enc_file_path}"
    print(url)

    headers = {
        "Authorization": f"token {access_token}",
        "Accept": "application/vnd.github.v3.raw"
    }
    
    while True:
        # Make the GET request to download the file
        response = requests.get(url, headers=headers)
        
        # Check the remaining rate limit from headers
        remaining_requests = int(response.headers.get('X-RateLimit-Remaining', 0))
        
        if response.status_code == 200:
            return response.text
        
        elif response.status_code == 403 and remaining_requests == 0:
            # Rate limit exceeded, wait until the reset time
            reset_time = int(response.headers.get('X-RateLimit-Reset'))
            sleep_time = max(0, reset_time - time.time())
            print(f"Rate limit exceeded. Sleeping for {sleep_time} seconds...")
            time.sleep(sleep_time)
        
        elif response.status_code in [500, 502, 503, 504]:
            # Retry on server errors with exponential backoff
            print("Server error encountered. Retrying...")
            time.sleep(5)
        
        else:
            raise Exception(f"Failed to download file. Status code: {response.status_code}, Remaining requests: {remaining_requests}")

def read_jsonl(path: str, disable: bool=False):
    data = []
    with open(path, "r") as f:
        for line in tqdm(f, disable=disable):
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

class MentaLinterDataset:
    """Dataset builder class for Meta-Linting task with support for multiple linter/SAST tools."""
    def __init__(self, linter_name: str, data_path: str, stack_folder_path: str="data/STACK-V2"):
        self.linter_name = linter_name
        self.stack_folder_path = stack_folder_path
        self.stack_data = load_stack_dump(stack_folder_path, as_dict=True)
        self.linter_data = getattr(self, f"load_{linter_name}")(data_path)

    def generate_data_mix(self, idioms_file: None):
        pass

    def load_ruff(self, path: str):
        ruff_results = load_ruff_results(path)
        data = []
        for rec in ruff_results:
            code_file = self.stack_data[rec['blob_id']]
            code_lines = code_file.split("\n")
            idioms_detected = []
            for violation in rec["violations"]:
                idiom_name = violation["code"]
                idiom_description = violation["message"]
                # so this is a big design decision. Should we pass on line numbers or just the line with the issue. Right now I'm focusing on predicting just the line number, but for future reference this is where we can make the modification.
                idiom_location = code_lines[violation["location"]["row"]-1] # currently just the code line.
                idioms_detected.append({
                    "name": idiom_name,
                    "type": "violation",
                    "location": idiom_location, 
                    "explanation": idiom_description,
                    "fix": None,
                })
            data.append({
                "code_file": code_file,
                "idioms_detected": idioms_detected,
            })

        return data


def init_s3_client():
    session = boto3.Session(
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"])
    s3 = session.client("s3")

    return s3

def download_contents(s3, blob_id, src_encoding):
    s3_url = f"s3://softwareheritage/content/{blob_id}"
    
    with open(s3_url, "rb", compression=".gz", transport_params={"client": s3}) as fin:
        content = fin.read().decode(src_encoding)
    
    return {"content": content}

# ds = load_dataset("bigcode/the-stack-v2", split="train", streaming=True)
# ds = ds.map(lambda row: download_contents(row["blob_id"], row["src_encoding"]))

# main
if __name__ == "__main__":
    access_token = json.load(open("./access_tokens2.json"))["GH_ACCESS_TOKEN"]
    # data = load_python_whatsnew_dataset("./data/Python-docs/whatsnew.jsonl")
    # for i in [32, 33, 34, 35, 37]:
    #     print(data[i]["prompt"])
    #     print(data[i]["response"])

    start = int(sys.argv[1])
    try: end = int(sys.argv[2])
    except IndexError as e:
        end = start + 5000

    s3 = init_s3_client()
    ds = load_dataset(
        "bigcode/the-stack-v2", "Python", 
        split="train", streaming=True
    )
    ds = ds.skip(start)
    # ds = ds.map(lambda row: download_contents(row["blob_id"], row["src_encoding"]))
    
    os.makedirs("./data/STACK-V2", exist_ok=True)
    write_path = os.path.join("./data/STACK-V2", f"Python-train-{start}.jsonl")
    if os.path.exists(write_path):
        overwrite = input("overwrite (y/N)?")
        if overwrite.lower().strip() not in ["yes", "y"]: exit()
        # overwrite data.
        open(write_path, "w")

    for idx,row in tqdm(enumerate(ds)):
        # if idx <= start: continue
        result = download_contents(s3, row["blob_id"], row["src_encoding"])
        for key, value in row.items():
            if not isinstance(row, str):
                row[key] = str(value)
        row['content'] = result['content']
        with open(write_path, "a") as f:
            f.write(json.dumps(row)+"\n")
        if (idx+start) >= end: exit()

    # # load bigcode files via Github: 
    # ds = load_dataset("bigcode/the-stack-v2", "Python", split="train", streaming=True)
    # for row in ds:
    #     # print(row)
    #     repo_name = row['repo_name']
    #     branch_name = row['branch_name']
    #     file_path = row['path']
    #     content = download_github_file(repo_name, branch_name, file_path, access_token)
    #     print(content)
    #     break