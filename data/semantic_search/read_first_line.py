import json

file_path = "/home/dkundego/OracleProject/data/semantic_search/pep506_filtered_entries.jsonl"

# Open the file and read the first line
with open(file_path, "r") as file:
    first_line = file.readline().strip()
    first_entry = json.loads(first_line)

print(first_entry)
