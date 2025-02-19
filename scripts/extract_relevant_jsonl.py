import os
import json

def filter_entries_by_keywords(directory, output_file, keywords):

    with open(output_file, "w") as out_file:
        for filename in os.listdir(directory):
            if filename.endswith(".jsonl"):
                file_path = os.path.join(directory, filename)

                with open(file_path, "r") as jsonl_file:
                    for line in jsonl_file:
                        entry = json.loads(line.strip())
                        content = entry.get("content", "")

                        if all(keyword in content for keyword in keywords):
                            out_file.write(json.dumps(entry) + "\n")

    print(f"Filtered entries with specified keywords have been saved to {output_file}.")

directory = "/home/dkundego/OracleProject/data/STACK-V2"
output_file = "pep506_password_import_random.jsonl"
keywords = ["import random", "password"]

filter_entries_by_keywords(directory, output_file, keywords)
