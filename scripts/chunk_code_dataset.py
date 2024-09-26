import pandas as pd
from src.datautils import read_jsonl

# main
if __name__ == "__main__":
    data = read_jsonl("data/Python-docs/whatsnew_with_code.jsonl")
    codeblocks_data = []
    for version_whatsnew_updates in data:
        for section_title, content in version_whatsnew_updates["sections"].items():
            # if len(content["code_blocks"]) == 0: continue
            for i, code_block in enumerate(content['code_blocks']):
                codeblocks_data.append({
                    "description": "" if i != 0 else content["text"], 
                    "title": section_title, 
                    "code_block": code_block}
                )
    pd.DataFrame(codeblocks_data).to_csv("./data/Python-docs/codeblocks.csv", index=False)