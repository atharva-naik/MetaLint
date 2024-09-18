# ensure that the PEP metadata is correct - like correct version number etc.

import os
import json
import pandas as pd
from tqdm import tqdm

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

# main
if __name__ == "__main__":
    METADATA_PATH = "./data/PEPS/metadata"
    pep_to_py_version = {}
    for category in os.listdir(METADATA_PATH):
        metadata = pd.read_csv(os.path.join(METADATA_PATH, category))
        for pep, py_version in zip(metadata['PEP'].tolist(), metadata['Python Version'].tolist()):
            pep_to_py_version[int(pep)] = py_version
    data = read_jsonl("data/PEPS/pep_pages.jsonl")
    for rec in data:
        rec["python_version"] = pep_to_py_version[int(rec['pep'])]
    print("bytes written:", write_jsonl(data, "data/PEPS/pep_pages_corrected_py_versions.jsonl"))