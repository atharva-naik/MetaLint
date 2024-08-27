import os
import time
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup

# main
JEPS_DATA_PATH = os.path.join("data", "JEPS")
os.makedirs(JEPS_DATA_PATH, exist_ok=True)
if __name__ == "__main__":
    N = 500 # number of jeps
    for i in tqdm(range(N)):
        data = requests.get(f"https://openjdk.org/jeps/{i+1}")
        if data.status_code == 404: continue
        content_text = BeautifulSoup(data.text, features="lxml").get_text()
        write_path = os.path.join(JEPS_DATA_PATH, f"JEP{i+1}.txt")
        with open(write_path, "w") as f:
            f.write(content_text)
        time.sleep(1)
        # exit()