import os
import sys
import json
from tqdm import tqdm

def read_jsonl(path: str):
    files = {}
    with open(path, "r") as f:
        for line in f:
            row = json.loads(line.strip())
            files[f'{row["blob_id"]}.py'] = row["content"]

    return files

dataset_name = sys.argv[1] # stack-v2
raw_data_folder = sys.argv[2] # ./data/STACK-V2

CODESCHOLAR_BUCKET_PATH = "/Users/atharvanaik/codescholar-bucket"
CODESCHOLAR_DATASET_PATH = os.path.join(CODESCHOLAR_BUCKET_PATH, "data", dataset_name)
CODESCHOLAR_PROG_DIR = os.path.join(CODESCHOLAR_DATASET_PATH, "raw")
os.makedirs(CODESCHOLAR_PROG_DIR, exist_ok=True)

all_files = {}
for file in tqdm(os.listdir(raw_data_folder), desc=f"reading {dataset_name}"):
    subset_files = read_jsonl(os.path.join(raw_data_folder, file))
    all_files.update(subset_files)

for file_path, content in tqdm(all_files.items(), desc=f"writing {dataset_name} source files"):
    write_path = os.path.join(CODESCHOLAR_PROG_DIR, file_path)
    with open(write_path, "w") as f:
        f.write(content+"\n")