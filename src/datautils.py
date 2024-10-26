import os
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

load_dotenv()

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
access_token = json.load(open("./access_tokens2.json"))["GH_ACCESS_TOKEN"]

# main
if __name__ == "__main__":
    # data = load_python_whatsnew_dataset("./data/Python-docs/whatsnew.jsonl")
    # for i in [32, 33, 34, 35, 37]:
    #     print(data[i]["prompt"])
    #     print(data[i]["response"])

    start = int(sys.argv[1])

    s3 = init_s3_client()
    ds = load_dataset("bigcode/the-stack-v2", "Python", split="train", streaming=True)
    # ds = ds.map(lambda row: download_contents(row["blob_id"], row["src_encoding"]))
    
    os.makedirs("./data/STACK-V2", exist_ok=True)
    write_path = os.path.join("./data/STACK-V2", f"Python-train-{start}.jsonl")
    if os.path.exists(write_path):
        overwrite = input("overwrite (y/N)?")
        if overwrite.lower().strip() not in ["yes", "y"]: exit()
        # overwrite data.
        open(write_path, "w")
    for idx,row in tqdm(enumerate(ds)):
        if idx <= start: continue
        result = download_contents(s3, row["blob_id"], row["src_encoding"])
        for key, value in row.items():
            if not isinstance(row, str):
                row[key] = str(value)
        row['content'] = result['content']
        with open(write_path, "a") as f:
            f.write(json.dumps(row)+"\n")

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