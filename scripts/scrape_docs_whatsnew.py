# scrape whatsnew data from docs.

import os
import bs4
import json
import time
import requests
import pandas as pd
from tqdm import tqdm

def extract_metadata_from_soup(soup):
    # Create an empty dictionary to store the metadata
    metadata = {}

    # Find all <dt> and <dd> pairs
    for dt, dd in zip(soup.find_all('dt'), soup.find_all('dd')):
        key = dt.get_text(strip=True).replace(':', '')  # Remove the colon
        # Handle cases where <dd> contains <a> or <abbr> tags
        if dd.find('a'):
            value = ', '.join([a.get_text(strip=True) for a in dd.find_all('a')])
        elif dd.find('abbr'):
            value = dd.find('abbr').get_text(strip=True)
        else:
            value = dd.get_text(strip=True)
        # Store key-value pairs in the dictionary
        metadata[key] = value

    return metadata

def get_next_level_sections(content) -> list:
    next_level_sections = content.find_all("div", class_="section", recursive=False)
    if len(next_level_sections) == 0:
        next_level_sections = content.find_all("section", recursive=False)

    return next_level_sections

def process_section_text(section):
    return section.text.encode('utf-8', errors='replace').decode('utf-8', errors='replace').replace("¶", "\n").strip()
    # .replace("Â¶", "\n").replace("â", "").replace("Â", "").strip()

# main
if __name__ == "__main__":
    data = []
    index = 0
    PY_DOC_VERSIONS = ['2.6', '2.7', '3.0', '3.1', '3.2', '3.3', '3.4', '3.5', '3.6', '3.7', '3.8', '3.9', '3.10', '3.11', '3.12', '3.13', '3.14']
    write_path = "./data/Python-docs/whatsnew_with_code.jsonl"
    if os.path.exists(write_path):
        response = input("overwrite data? (y/N)\n")
        if response.lower().strip() not in ["yes", "y"]: exit()
    open(write_path, "w")
    pbar = tqdm(PY_DOC_VERSIONS)
    for version in pbar:
        pbar.set_description(f"scraping {version} whatsnew")
        url = f"https://docs.python.org/{version}/whatsnew/{version}.html"
        content = requests.get(url).content
        # print(content)
        soup = bs4.BeautifulSoup(content.decode('utf-8'), features="lxml")
        rec = {
            "index": index, "python_version": version, "url": url,
            # "metadata": extract_metadata_from_soup(soup),
            "sections": {}
        }
        ver, subver = version.split(".")
        whatsnew_section_id = f"what-s-new-in-python-{ver}-{subver}"
        # print(whatsnew_section_id)
        doc_content = soup.find("div", class_="section", id=whatsnew_section_id)
        if doc_content is None:
            doc_content = soup.find("section", id=whatsnew_section_id)
        top_level_sections = get_next_level_sections(doc_content)
        for top_level_section in top_level_sections:
            sub_sections = get_next_level_sections(top_level_section)
            if len(sub_sections) == 0:
                id = top_level_section["id"]
                # if id == "contents": continue
                rec["sections"][id] = {
                    "text": process_section_text(top_level_section),
                    "code_blocks": [div.text for div in top_level_section.find_all("div", class_="highlight")]
                }
            else:
                for section in sub_sections:
                    id = top_level_section["id"] + " - " + section["id"]
                    # if id == "contents": continue
                    rec["sections"][id] = {
                        "text": process_section_text(section),
                        "code_blocks": [div.text for div in section.find_all("div", class_="highlight")]
                    }
        # with open("DELETE.html", "w") as f:
        #     f.write(content)
        with open(write_path, "a", encoding="utf8") as f:
            f.write(json.dumps(rec, ensure_ascii=False)+"\n")
        index += 1
        time.sleep(1)