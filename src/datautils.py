import os
import re
import sys
import time
import json
import copy
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

META_LINTING_PROMPT = """Look at the following list of code idiom specifications with definitions and examples:
{LIST_OF_IDIOM_SPECS}

Given these idioms, your task is to look at a code file and detect violations of the above idioms, and flag them like a linter. You should also suggest a fix if possible. If there are no fixes then just say “ALL CHECKS PASSED”. For reporting violations your output should follow a JSON format with each idiom violation in a new line. Here are some example outputs:
{EG1}
{EG2}

Code file:
{CODE_FILE}

Idiom violations (if any):
"""

EG1 = {
    'code': 'D100',
    'fix': None,
    'message': 'Missing docstring in public module',
    'code_spans_and_lines': [{'line': 'import random', 'span': ''}]
}
EG2 = {
    'code': 'ANN204',
    'fix': {'applicability': 'unsafe',
    'edits': [{'content': ' -> None',
    'code_spans_and_lines': [{'line': '    def __init__(self, nodes, meanDegree, meanWeight):',
        'span': ''}]}],
    'message': 'Add return type annotation: `None`'},
    'message': 'Missing return type annotation for special method `__init__`',
    'code_spans_and_lines': [{'line': '    def __init__(self, nodes, meanDegree, meanWeight):',
    'span': '__init__'}]
}

def convert_location_to_code_line(rec: dict, code_lines: list[str]): #stack_data: dict[str, dict]):
    violations = rec["violations"]
    blob_id = rec["blob_id"]
    rec_with_code_lines = {
        "blob_id": rec["blob_id"],
        "violations": []
    }
    # code_file = stack_data[blob_id]['content']
    # code_lines = code_file.split("\n")
    for violation in violations:
        loc = violation["location"]
        end_loc = violation["end_location"]
        rec_with_code_lines["violations"].append(violation)

        code_spans_and_lines = []
        for i in range(loc["row"]-1, end_loc["row"]):
            code_spans_and_lines.append({
                "line": code_lines[i], 
                "span": code_lines[i][loc["column"]-1:end_loc["column"]-1],
        })
        rec_with_code_lines["violations"][-1]["line_span"] = [
            violation["location"]["row"],
            violation["end_location"]["row"]
        ]
        rec_with_code_lines["violations"][-1]["code_spans_and_lines"] = code_spans_and_lines
        fix = violation["fix"]
        if fix is not None:
            for edit in violation["fix"]["edits"]:
                code_spans_and_lines = []
                # print(edit) 
                loc = edit["location"]
                end_loc = edit["end_location"]
                for i in range(loc["row"]-1, end_loc["row"]):
                    code_spans_and_lines.append({
                        "line": code_lines[i], 
                        "span": code_lines[i][loc["column"]-1:end_loc["column"]-1],
                    })
                edit["code_spans_and_lines"] = code_spans_and_lines
                del edit["location"]
                del edit["end_location"]
        del violation['filename']
        del violation['cell']
        del violation['noqa_row']
        del violation['url']
        del violation['location']
        del violation['end_location']

    return rec_with_code_lines

class MetaLinterDataset:
    """Dataset builder class for Meta-Linting task with support for multiple linter/SAST tools."""
    def __init__(self, linter_name: str, linter_data_folder: str, code_files_path: str="data/STACK-V2", idiom_spec_path: str="data/ruff_pages"):
        self.linter_name = linter_name
        self.code_files_path = code_files_path
        self.idiom_spec_path = idiom_spec_path
        self.linter_data_folder = linter_data_folder
        # self.max_code_lines = max_code_lines
        self.code_files = load_stack_dump(code_files_path, as_dict=True)
        self.code_idiom_specs = getattr(self, f"load_{linter_name}_idiom_specs")(idiom_spec_path)
        self.linter_data = getattr(self, f"load_{linter_name}")(linter_data_folder)

    def generate_data_mix(self, idiom_list: list[str], k: Union[int, None]=None, max_code_lines: int=1000):
        subset_idiom_specs = {}
        for idiom_name in idiom_list:
            subset_idiom_specs[idiom_name] = self.code_idiom_specs[idiom_name]
        # print(subset_idiom_specs)
        data = []
        CTR = 0
        if k is None:
            for blob_id,linter_rec in self.linter_data.items():
                # try: 
                # code_lines = linter_rec["code_file"].split("\n")
                num_code_lines = linter_rec["num_code_lines"]
                if num_code_lines > max_code_lines: continue # skip files with too many lines of code.
                prompt, response = self.generate_prompt_and_response(subset_idiom_specs, linter_record=linter_rec)
                # except Exception as e: print(e); continue
                ID = f"{'-'.join(idiom_list)}_{blob_id}"
                source = "rull_linter/"+"-".join(idiom_list)
                CTR += 1
                data.append({
                    "id": ID,
                    "messages": [
                        {"content": prompt, "role": "user"},
                        {"content": response, "role": "assistant"}
                    ],
                    "source": source,
                })
        else:
            indices = random.sample(len(self.linter_data), k=k)
            blob_ids = list(self.linter_data.keys())
            for i in indices:
                blob_id = blob_ids[i]
                linter_rec = self.linter_data[blob_id]
                try: prompt, response = self.generate_prompt_and_response(subset_idiom_specs, linter_record=linter_rec)
                except Exception as e: 
                    print(e)
                    continue
                ID = f"{'-'.join(idiom_list)}_{blob_id}"
                source = "rull_linter/"+"-".join(idiom_list)
                CTR += 1
                data.append({
                    "id": ID,
                    "messages": [
                        {"content": prompt, "role": "user"},
                        {"content": response, "role": "system"}
                    ],
                    "source": source,
                })

        return data

    def load_ruff_idiom_specs(self, path):
        metadata_path = os.path.join(path, "rule_metadata.json")
        self.ruff_metadata = json.load(open(metadata_path))
        self.ruff_metadata = {rec['Code']: rec for rec in self.ruff_metadata}
        rules_folder = os.path.join(path, "rules")
        code_idiom_specs = {}
        for rule_path in os.listdir(rules_folder):
            rule_id,_=os.path.splitext(rule_path) # rule_id is the same as the idiom name for Ruff.
            rule_data = json.load(open(os.path.join(rules_folder, rule_path)))
            code_idiom_specs[rule_id] = rule_data
            code_idiom_specs[rule_id]["name"] = self.ruff_metadata[rule_id]['Name']
            code_idiom_specs[rule_id]["code"] = rule_id

        return code_idiom_specs

    def idiom_spec_extractor_for_ruff(self, idiom_spec):
        if "example" in idiom_spec:
            Example = f"\n\nExample:\n{idiom_spec['example']}"
        else: Example = ""
        return f"# Idiom {idiom_spec['code']} ({idiom_spec['name']})\n\nDefinition: {idiom_spec['what-it-does']}\n\nRationale: {idiom_spec['why-is-this-bad']}"+Example

    def generate_prompt_and_response(self, idiom_specs: dict[str, dict], linter_record):
        """Can take a subset of all the idiom specs"""
        idioms_to_be_covered = list(idiom_specs.keys())
        # print(idiom_specs)
        LIST_OF_IDIOM_SPECS = "\n\n".join([getattr(self, f"idiom_spec_extractor_for_{self.linter_name}")(idiom_spec) for idiom_spec in idiom_specs.values()])
        
        # delete line number data (can also delete messages here if needed)
        for violation in linter_record["idioms_detected"]:
            assert isinstance(violation, dict)
            violation.pop("line_span", None) # delete "line_span" if the key exists

        for index_ in range(len(linter_record["idioms_detected"])):
            if isinstance(linter_record["idioms_detected"][index_], str):
                linter_record["idioms_detected"][index_] = json.loads(linter_record["idioms_detected"][index_])
        
        # linter_record["idioms_detected"] = [json.dumps(idiom_violation) for idiom_violation in linter_record["idioms_detected"] if idiom_violation['code'] in idioms_to_be_covered]
        selected_idiom_violations = [json.dumps(idiom_violation) for idiom_violation in linter_record["idioms_detected"] if idiom_violation['code'] in idioms_to_be_covered]
        CODE_FILE = linter_record["code_file"]

        if len(selected_idiom_violations) == 0:
            response = "NO VIOLATIONS FOUND"
        else: response = "\n".join(selected_idiom_violations)
        # print(response)
        # print(LIST_OF_IDIOM_SPECS)
        prompt = META_LINTING_PROMPT.format(
            LIST_OF_IDIOM_SPECS=LIST_OF_IDIOM_SPECS, 
            EG1=json.dumps(EG1), EG2=json.dumps(EG2), 
            CODE_FILE=CODE_FILE
        )
        # print(prompt)

        return prompt, response

    def load_ruff(self, folder: str):
        code_file_id_to_linter_data = {}
        CTR = 0
        for file in tqdm(os.listdir(folder), desc="loading shards"):
            path = os.path.join(folder, file)
            shard_linter_data = self.load_ruff_shard(path)
            code_file_id_to_linter_data.update(shard_linter_data)
            # DEBUG:
            CTR += 1 
            if CTR == 40: break
        
        return code_file_id_to_linter_data

    def edit_code(self, lines: list, edits: list[dict]) -> dict:
        modified_lines = {}
        original_lines = lines[:]
        
        for edit in edits:
            start_row = edit['location']['row']-1
            start_col = edit['location']['column']-1
            end_row = edit['end_location']['row']-1
            end_col = edit['end_location']['column']-1
            replacement = edit['content']
            # try:
            before = lines[start_row][:start_col]
            after = lines[end_row][end_col:]
            # except IndexError:
            #     print(start_row, start_col, end_row, end_col, replacement)
            #     print(edits)
            #     print(len(lines))
            #     print(len(lines[start_row]))
            
            if start_row == end_row:
                new_line = before + replacement + after
                modified_lines[original_lines[start_row]] = new_line
                lines[start_row] = new_line
            else:
                new_start_line = before + replacement
                new_end_line = after
                modified_lines[original_lines[start_row]] = new_start_line
                modified_lines[original_lines[end_row]] = new_end_line
                lines[start_row] = new_start_line
                lines[end_row] = new_end_line
                del lines[start_row + 1 : end_row]
        
        return modified_lines

    def load_ruff_shard(self, path: str):
        ruff_results = read_jsonl(path)
        data = {}
        for rec in tqdm(ruff_results):
            blob_id = rec['blob_id']
            code_file = self.code_files[blob_id]['content']
            code_lines = code_file.split("\n")
            # if len(code_lines) > self.max_code_lines: continue
            try: 
                rec = convert_location_to_code_line(rec, code_lines=code_lines)
            except IndexError:
                # print(blob_id)
                continue
            # print(len(code_lines))
            idioms_detected = []
            for violation in rec["violations"]:
                # idiom_name = violation["code"]
                # idiom_description = violation["message"]
                # # so this is a big design decision. Should we pass on line numbers or just the line with the issue. Right now I'm focusing on predicting just the line number, but for future reference this is where we can make the modification.
                # idiom_location = "\n".join(code_lines[violation["location"]["row"]-1:violation["end_location"]["row"]]) # currently just the code line.
                # fix = violation['fix']
                # fix_mapping = None
                # fix_message = None
                # try:
                #     if fix is not None and fix['applicability'] == 'safe':
                #         fix_mapping = self.edit_code(copy.deepcopy(code_lines), fix['edits'])
                #         fix_message = fix['message']
                # except IndexError:
                #     print(blob_id)
                #     print(fix)
                #     print(violation)
                #     print(len(code_lines))
                #     # exit()
                # idioms_detected.append({
                #     "name": idiom_name,
                #     "type": "violation",
                #     "lines": idiom_location, 
                #     "message": idiom_description,
                #     "fix": fix_mapping,
                #     "fix_message": fix_message,
                # })
                idioms_detected.append(violation)
                # 'fix': {'applicability': 'safe',
                #     'edits': [{'content': '',
                #     'end_location': {'column': 52, 'row': 20},
                #     'location': {'column': 49, 'row': 20}}],
                #     'message': 'Remove `start` argument'},
            data[blob_id] = {
                "code_file": code_file,
                "num_code_lines": len(code_lines),
                "idioms_detected": idioms_detected,
            }

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