# scrape PEP files.

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

# main
if __name__ == "__main__":
    METADATA_PATH = "./data/PEPS/metadata"
    data = []
    index = 0
    write_path = "./data/PEPS/pep_pages.jsonl"
    if os.path.exists(write_path):
        response = input("overwrite data? (y/N)\n")
        if response.lower().strip() not in ["yes", "y"]: exit()
    open(write_path, "w")
    for category in os.listdir(METADATA_PATH):
        metadata = pd.read_csv(os.path.join(METADATA_PATH, category))
        peps = metadata["PEP"].tolist()
        authors = metadata["Authors"].tolist()
        python_versions = metadata["Python Version"].tolist()
        titles = metadata["Title"].tolist()
        pbar = tqdm(
            enumerate(peps),
            total=len(peps),
            desc=f"scraping {category}"
        )
        for j, pep in pbar:
            pep = str(pep).rjust(4, "0")
            url = f"https://peps.python.org/pep-{pep}/"
            content = requests.get(url).text
            soup = bs4.BeautifulSoup(content, features="lxml")
            rec = {
                "index": index, "pep": pep,
                "category": category, "url": url,
                "authors": authors[j], "title": titles[j],
                "python_version": python_versions[j],
                "metadata": extract_metadata_from_soup(soup),
                "sections": {}
            }
            pep_content = soup.find("section", id="pep-content")
            for section in pep_content.find_all("section", recursive=False):
                id = section["id"]
                if id == "contents": continue
                rec["sections"][id] = section.text
            # with open("DELETE.html", "w") as f:
            #     f.write(content)
            with open(write_path, "a") as f:
                f.write(json.dumps(rec)+"\n")
            index += 1
            time.sleep(1)