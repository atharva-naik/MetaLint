import os
import re
import json
import numpy as np
from typing import *
from tqdm import tqdm
import scipy.stats as stats
import matplotlib.pyplot as plt


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

def extract_urls(text):
    # Regular expression pattern to match URLs
    url_pattern = re.compile(
        r'(https?://|www\.)'  # Match http://, https://, or www.
        r'(\w+\.)?'           # Match any subdomains
        r'(\w+\.\w+)'         # Match domain and TLD
        r'(/\S*)?'            # Match path, parameters, anchors, etc.
    )

    # Find all URLs in the text
    urls = url_pattern.findall(text)

    # Reconstruct URLs from the matched groups
    urls = ["".join(url) for url in urls]

    return urls

def filter_review_comment_file(file_path: str, comment_key: str="body"):
    filt_data = []
    data = read_jsonl(file_path)
    for rec in data:
        if "http:" in rec[comment_key] or "https:" in rec[comment_key]:
            filt_data.append(rec)
    print(f"{len(data)} -> {len(filt_data)}")

    return filt_data

# main
if __name__ == "__main__":
    data = []
    for split in ["train", "valid", "test"]:
        data += read_jsonl(f"/home/arnaik/CodeReviewEval/data/Comment_Generation/msg-{split}.jsonl")
    # print(data[0]["msg"])
    reviews_with_urls = [r for r in data if "http:" in r['msg'] or 'https:' in r['msg']]
    print(len(reviews_with_urls), "reviews with URLs found")