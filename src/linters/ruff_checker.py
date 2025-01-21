import os
import sys
import json
import random
import string
import pathlib
import subprocess
from tqdm import tqdm

def run_command(command: str) -> str:
    """
    Runs a terminal command and returns its output as a string.
    
    Args:
        command (str): The command to run in the terminal.
    
    Returns:
        str: The output of the command.
    """
    try:
        result = subprocess.run(command, shell=True, text=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        # os.system(command)
        # print(e.stdout.strip())
        return json.loads(e.stdout.strip()) # f"Error: {e.stderr.strip()}"
    
project_path = str(pathlib.Path(os.path.realpath(__file__)).parent.parent.parent)
sys.path.append(project_path)

from src.datautils import load_stack_dump

def run_ruff(code: str):
    path = "".join(random.sample(string.ascii_letters+string.digits, k=16))
    with open(path, "w") as f:
        f.write(code+"\n")
    # os.system(f"ruff check {path}")
    output = run_command(f"ruff check {path} --output-format json")
    os.remove(path=path)

    return output

# main
if __name__ == "__main__":
    data = load_stack_dump("./data/STACK-V2")
    try: start = int(sys.argv[1])
    except IndexError: start = 0
    write_path = f"./data/ruff_check_results_v2/output_from_{start}.jsonl"
    print(write_path)
    if os.path.exists(write_path):
        overwrite = input("Overwrite (Y/n)?").lower().strip()
        if overwrite not in ["yes","y"]: exit()
    open(write_path, 'w')
    for i,rec in tqdm(enumerate(data), total=len(data)):
        if i < start: continue
        ruff_analysis_report = run_ruff(rec['content'])
        with open(write_path, "a") as f:
            f.write(json.dumps({
                "blob_id": rec['blob_id'], 
                # 'content': rec['content'], 
                "violations": ruff_analysis_report})+"\n")
        # print(run_ruff(rec['content']))
        # exit()
