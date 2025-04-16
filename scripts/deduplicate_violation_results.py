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
# from src.metrics.meta_linting.idiom_detection_and_localization_v3 import load_linter_results

def deduplicate_preserve_order(lst):
    seen = set()
    result = []
    for item in lst:
        if str(item) not in seen:
            seen.add(str(item))
            result.append(item)
    return result

def load_linter_results(text):
    results = []
    idiom_code = None
    for line in text.split("\n"):
        line = line.strip()
        if line == "": continue
        elif line.startswith("**Idiom") and line.endswith("Violations:**"):
            idiom_code = line.removesuffix("Violations:**").removeprefix("**Idiom").strip()
        elif line == "NO VIOLATIONS FOUND": results.append({"code": idiom_code})
        else:
            try: 
                result = json.loads(line)
                result["code"] = idiom_code
                results.append(result)
            except Exception as e: pass
                # print(e)
                # print(f"{e}: {line}")
    return results

def generate_linter_response(violations: list[dict]):
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

    return response.strip("\n")

# main
if __name__ == "__main__":
    train_data = json.load(open("./data/ruff_meta_linting/train_v4_new_format_with_lineno.json"))
    test_data = json.load(open("./data/ruff_meta_linting/test_v4_new_format_with_lineno.json"))
    for split, data in [("train", train_data), ("test", test_data)]:
        dedup_cases = 0
        for rec in tqdm(data):
            # print(rec["messages"][-1]["content"])
            response = rec["messages"][-1]["content"].strip("\n")
            regen_response = generate_linter_response(load_linter_results(response))
            assert response == regen_response
            dedup_response = generate_linter_response(deduplicate_preserve_order(load_linter_results(response)))
            if dedup_response != response: 
                dedup_cases += 1
            rec["messages"][-1]["content"] = dedup_response+"\n"
        print(f"dedup cases in {split}: {dedup_cases}")
        with open(f"./data/ruff_meta_linting/{split}_v4_new_format_with_lineno.json", "w") as f:
            json.dump(data, f, indent=4)