import os
import json
from typing import *
from tqdm import tqdm

def read_jsonl(path, cutoff: Union[int, None]=None):
    data = []
    with open(path) as f:
        ctr = 0
        for line in tqdm(f):
            try:
                js = json.loads(line.strip())
            except:
                print("Error during reading json data.")
                continue
            data.append(js)
            ctr += 1
            if cutoff is not None and ctr > cutoff: break
    return data

# main
if __name__ == "__main__":
    data = []
    for split in ["train", "valid", "test"]:
        data += read_jsonl(f"data/Comment_Generation/msg-{split}.jsonl")
    print(data[0]["msg"])