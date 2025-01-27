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

def filter_report(ruff_report):
    if isinstance(ruff_report, str):
        ruff_report = json.loads(ruff_report)
    filt_ruff_report = [] # remove SyntaxError related violations.
    for violation in ruff_report:
        if str(violation['code']) != "None":
            filt_ruff_report.append(violation)

    return filt_ruff_report

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
    OUTPUT_FOLDER: str = "ruff_results"
    buffer_size: int 
    try: start = int(sys.argv[1])
    except IndexError: start = 0 # follows array style indexing.
    try: buffer_size = int(sys.argv[2])
    except IndexError: buffer_size = 5000
    
    write_path = f"./data/{OUTPUT_FOLDER}/output_from_{start}_buffer_size_{buffer_size}.jsonl"
    print(write_path)
    if os.path.exists(write_path):
        overwrite = input("Overwrite (Y/n)?").lower().strip()
        if overwrite not in ["yes","y"]: exit()
    open(write_path, 'w')
    current_buffer_size = 0
    for i,rec in tqdm(enumerate(data), total=len(data)):
        if i < start: continue
        ruff_analysis_report = filter_report(run_ruff(rec['content']))
        with open(write_path, "a") as f:
            f.write(json.dumps({
                "blob_id": rec['blob_id'], 
                # 'content': rec['content'], 
                "violations": ruff_analysis_report})+"\n")
            current_buffer_size += 1
        if current_buffer_size == buffer_size:
            write_path = f"./data/{OUTPUT_FOLDER}/output_from_{i}_buffer_size_{buffer_size}.jsonl"
            if os.path.exists(write_path): exit(f"terminating to avoid overwrite conflicts with: {write_path} which already exists.")
            open(write_path, 'w')
        # print(run_ruff(rec['content']))
        # exit()
