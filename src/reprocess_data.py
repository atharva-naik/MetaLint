import os
import sys
import json
import pathlib
from dotenv import load_dotenv

load_dotenv()

module_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent)
sys.path.append(module_path)

from src.datautils import load_stack_dump, load_ruff_results, load_ruff_idiom_specs, reprocess_data

# main
if __name__ == "__main__":
    split = sys.argv[1] #"train"
    data = json.load(open(f"./data/ruff_meta_linting/{split}_v4.json"))
    stack_data = load_stack_dump("./data/STACK-V2", as_dict=True)
    code_idiom_specs = load_ruff_idiom_specs("./data/ruff_pages")
    ruff_results = load_ruff_results("./data/ruff_results", as_dict=True)
    proc_data = reprocess_data(train_data=data, code_idiom_specs=code_idiom_specs, ruff_results=ruff_results, stack_data=stack_data, add_line_numbers=True)
    with open(f"./data/ruff_meta_linting/{split}_v4_new_format_with_lineno.json", "w") as f:
        json.dump(proc_data, f, indent=4)