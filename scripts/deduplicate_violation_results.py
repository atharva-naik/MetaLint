import os
import sys
import ast
import json
import pathlib
from tqdm import tqdm
import difflib
from collections import defaultdict

project_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent)
sys.path.append(project_path)

from src.datautils import read_jsonl

def deduplicate_preserve_order(lst):
    seen = set()
    result = []
    for item in lst:
        if str(item) not in seen:
            seen.add(str(item))
            result.append(item)
    return result

def load_linter_results_for_dedup(text):
    results = []
    idiom_code = None
    for line in text.split("\n"):
        line = line.strip()
        if line == "": continue
        elif line.startswith("**Idiom") and line.endswith("Violations:**"):
            idiom_code = line.removesuffix("Violations:**").removeprefix("**Idiom").strip()
        elif line == "NO VIOLATIONS FOUND": 
            if idiom_code is not None:
                results.append({"code": idiom_code})
        else:
            try: 
                result = json.loads(line)
                result["code"] = idiom_code
                results.append(result)
            except Exception as e: pass
                # print(e)
                # print(f"{e}: {line}")
    return results

def parse_response(response: str):
    try:
        cot, report = response.split("### Final Idiom Violations Found")
        return cot.strip("\n"), report.strip("\n")
    except ValueError:
        return "",response

def generate_linter_response(cot, violations: list[dict]):
    # print(cot)
    response = ""
    violations_by_code = defaultdict(lambda: []) # should already be sorted.
    for violation in violations:
        if len(violation.keys()) == 1:
            violations_by_code[violation['code']] = []
        else:
            violations_by_code[violation['code']].append(violation)
            del violation['code']
    for code, violations in violations_by_code.items():
        if len(violations) == 0:
            # response += f"**Idiom {code} Violations:**\n\nNO VIOLATIONS FOUND\n\n"
            response += f"**Idiom {code} Violations:**\n"
            response += f"\nNO VIOLATIONS FOUND\n"
        else: 
            response += f"**Idiom {code} Violations:**\n"
            response += f"\n{'\n'.join([json.dumps(violation) for violation in violations])}\n"
        response += "\n"

    if cot == "":
        return response.strip("\n")
    return cot+"\n\n### Final Idiom Violations Found\n\n"+response.strip("\n")

def reproduce_response(response):
    cot, final_response = parse_response(response=response)
    violations = load_linter_results_for_dedup(final_response)
    return generate_linter_response(cot, violations)

def deduplicate_response(response):
    cot, final_response = parse_response(response=response)
    violations = deduplicate_preserve_order(load_linter_results_for_dedup(final_response))
    return generate_linter_response(cot, violations)

# main
if __name__ == "__main__":
    dataset_version = "all_idioms/"
    splits = ["train_subtask_cot_star", "train", "test"]
    datasets = []
    for split in splits:
        datasets.append(
            json.load(open(f"./data/ruff_meta_linting/{dataset_version}{split}.json"))
        )
    
    for split, data in zip(splits, datasets):
        skipped_responses_with_issues = 0
        dedup_cases = 0
        for rec in tqdm(data):
            # print(rec["messages"][-1]["content"])
            response = rec["messages"][-1]["content"].strip("\n")
            regen_response = reproduce_response(response)
            # assert response == regen_response, f"orig: {response} regen: {regen_response}"
            try: 
                assert load_linter_results_for_dedup(response) == load_linter_results_for_dedup(regen_response), f"{parse_response(response)[1]} {parse_response(regen_response)[1]}"
            except AssertionError as e:
                print(e)
                skipped_responses_with_issues += 1
                continue
            dedup_response = deduplicate_response(response)
            if dedup_response != regen_response: 
                # print(dedup_response)
                # print(regen_response)
                # exit()
                dedup_cases += 1
            rec["messages"][-1]["content"] = dedup_response+"\n"
        
        print(f"dedup cases in {split}: {dedup_cases}")
        print(f"skipped cases in {split}: {skipped_responses_with_issues}")
        print(f"./data/ruff_meta_linting/{dataset_version}{split}.json")
        with open(f"./data/ruff_meta_linting/{dataset_version}{split}.json", "w") as f:
            json.dump(data, f, indent=4)