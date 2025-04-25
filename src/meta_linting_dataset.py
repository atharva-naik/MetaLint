import os
import re
import sys
import time
import json
import copy
import boto3
import dotenv
import random
import pathlib
import requests
import pandas as pd
from typing import *
from tqdm import tqdm
from smart_open import open
from dotenv import load_dotenv
from urllib.parse import quote
from datasets import load_dataset
from collections import defaultdict

module_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent)
sys.path.append(module_path)

load_dotenv()

from src.datautils import load_stack_dump, generate_response_from_violations, add_line_no_to_stack_file, read_jsonl, META_LINTING_PROMPT_V2

class MetaLinterDatasetEfficient:
    """Dataset builder class for Meta-Linting task with support for multiple linter/SAST tools."""
    def __init__(self, linter_name: str, linter_data_folder: str, code_files_path: str="data/STACK-V2", idiom_spec_path: str="data/ruff_pages_new", max_code_lines: Union[None, int]=None):
        self.linter_name = linter_name
        self.max_code_lines = max_code_lines
        self.code_files_path = code_files_path
        self.idiom_spec_path = idiom_spec_path
        self.linter_data_folder = linter_data_folder
        self.code_files = load_stack_dump(code_files_path, as_dict=True)
        self.all_kept_blob_ids = set() # all blob ids that fit the length constraints.
        self.code_idiom_specs = getattr(self, f"load_{linter_name}_idiom_specs")(idiom_spec_path)
        self.idiom_code_to_code_file_id = defaultdict(lambda: set())
        self.linter_data = getattr(self, f"load_{linter_name}")(linter_data_folder)

    def generate_data_mix(self, idiom_list: list[str], k: int=5000):
        subset_idiom_specs = {}
        per_idiom_violation_blob_ids = []
        at_least_one_idiom_violated_ids = set() # blob IDs for which at least one idiom is violated.
        
        for idiom_code in idiom_list:
            subset_idiom_specs[idiom_code] = self.code_idiom_specs[idiom_code]
            idiom_code_blob_ids = self.idiom_code_to_code_file_id[idiom_code]
            per_idiom_violation_blob_ids.append(idiom_code_blob_ids)
            at_least_one_idiom_violated_ids = at_least_one_idiom_violated_ids.union(idiom_code_blob_ids)
        # print([len(idiom_code_blob_ids) != 0 for idiom_code_blob_ids in per_idiom_violation_blob_ids])
        IDIOMS_WITH_NON_ZERO_VIOLATIONS = sum([len(idiom_code_blob_ids) != 0 for idiom_code_blob_ids in per_idiom_violation_blob_ids])
        file_ids_with_violations = set()
        file_ids_without_violations = self.all_kept_blob_ids.difference(at_least_one_idiom_violated_ids)

        for idiom_code_blob_ids in per_idiom_violation_blob_ids:
            if len(idiom_code_blob_ids) == 0: continue
            sub_k = min(int(k / (2*IDIOMS_WITH_NON_ZERO_VIOLATIONS)), len(idiom_code_blob_ids))
            file_ids_with_violations = file_ids_with_violations.union(set(random.sample(list(idiom_code_blob_ids), k=sub_k)))

        selected_blob_ids = set()
        selected_blob_ids = selected_blob_ids.union(file_ids_with_violations)
        print(len(selected_blob_ids))
        selected_blob_ids = selected_blob_ids.union(random.sample(list(file_ids_without_violations), k=k//2))
        print(len(selected_blob_ids))
        
        # print(subset_idiom_specs)
        data = []
        CTR = 0

        for blob_id in list(selected_blob_ids):
            linter_rec = self.linter_data[blob_id]
            try: prompt, response = self.generate_prompt_and_response_v2(subset_idiom_specs, linter_record=linter_rec)
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

    def generate_prompt_and_response_v2(self, idiom_specs: dict[str, dict], linter_record):
        """Can take a subset of all the idiom specs"""
        
        idioms_to_be_covered = list(idiom_specs.keys())
        LIST_OF_IDIOM_SPECS = "\n\n".join([getattr(self, f"idiom_spec_extractor_for_{self.linter_name}")(idiom_spec) for idiom_spec in idiom_specs.values()])
        
        # add line numbers to code file.
        CODE_FILE = linter_record["code_file"]
        CODE_FILE_WITH_LINENOS = add_line_no_to_stack_file(CODE_FILE)

        # generate response.
        response = generate_response_from_violations(
            violations=linter_record['idioms_detected'], 
            stack_file_lines=CODE_FILE.split("\n"), 
            meta_task_idiom_codes=idioms_to_be_covered,
            add_line_numbers=True
        )

        prompt = META_LINTING_PROMPT_V2.format(LIST_OF_IDIOM_SPECS=LIST_OF_IDIOM_SPECS, CODE_FILE=CODE_FILE_WITH_LINENOS)

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
            # if CTR == 40: break
        
        return code_file_id_to_linter_data

    def load_ruff_shard(self, path: str):
        ruff_results = read_jsonl(path)
        data = {}
        for rec in tqdm(ruff_results):
            blob_id = rec['blob_id']
            code_file = self.code_files[blob_id]['content']
            code_lines = code_file.split("\n")
            # skip loading STACK files with a lot of lines (specified by max_code_lines).
            if self.max_code_lines is not None and len(code_lines) > self.max_code_lines: continue
            idioms_detected = []
            for violation in rec["violations"]:
                idioms_detected.append(violation)
                code = violation["code"]
                self.idiom_code_to_code_file_id[code].add(blob_id)
                self.all_kept_blob_ids.add(blob_id)
            data[blob_id] = {
                "code_file": code_file,
                "num_code_lines": len(code_lines),
                "idioms_detected": idioms_detected,
            }

        return data