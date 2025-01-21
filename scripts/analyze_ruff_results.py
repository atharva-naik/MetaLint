import os
import sys
import ast
import json
import pathlib

project_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent)
sys.path.append(project_path)

from src.datautils import load_ruff_results, get_remaining_blob_ids

# main
if __name__ == "__main__":
    results = load_ruff_results("data/ruff_check_results")
    # leftover_blob_ids = get_remaining_blob_ids("data/STACK-V2", ruff_results=results)
    # print(f"#files analyzed:", len(results))
    # print(f"#files left to analyze: {len(leftover_blob_ids)}")
    for rec in results:
        violations = ast.literal_eval(rec["violations"])
        if len(violations) > 0:
            print(rec)
            break