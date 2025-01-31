import os
import bs4
import time
import json
import requests
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

LEGEND = {
    '✔': "Stable",
    "🧪": 'Unstable/Preview',
    "⚠️": "Depreceated",
    "❌": "Deleted",
    "🛠": "Automatically Fixable",
}

def get_all_ruff_rules(rules_page_url: str="https://docs.astral.sh/ruff/rules/"):
    with open("../data/darsh_ruff_pages/rule_metadata.json", "r") as f:
        ruff_rules_metadata_json = json.load(f)

    return ruff_rules_metadata_json

def split_example_section(example_text):
    example_before = example_text.split("\n\nUse instead:\n\n")[0]
    example_after  = example_text.split("\n\nUse instead:\n\n")[-1].split("\n\n")[0]
    return {
        "example-before": example_before,
        "example-after": example_after
    }

def parse_rule_spec(desc_page_soup: bs4.BeautifulSoup):
    desc_page_json = {}
    for key in ["what-it-does", "why-is-this-bad", "example", "fix-safety"]:
        # try: 
        starting_point = desc_page_soup.find("h2", id=key)
        if starting_point is None: continue
        starting_point = starting_point.nextSibling
        value_lines = []
        while starting_point and starting_point.name != 'h2':
            value_lines.append(starting_point.text)
            starting_point = starting_point.nextSibling
        value = "\n".join(value_lines).strip("\n")
        desc_page_json[key] = value
        # except AttributeError: 
        #     print(key)
        #     continu
    # some ruff pages don't have the 'example' section:
    if "example" in desc_page_json:
        desc_page_json["example-split"] = split_example_section(desc_page_json["example"])

    return desc_page_json

# main
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct-1M")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct-1M")

    ruff_rules_metadata_json = get_all_ruff_rules()
    rule_desc_file_url = "../data/darsh_ruff_pages/rules/{Code}.json"

    print("Hello Hello")
    for rule in tqdm(ruff_rules_metadata_json):
        if rule["fix_available"]:
            final_file_url = rule_desc_file_url.format(Code=rule['Code'])
            print(final_file_url)
            with open(final_file_url, "r") as f:
                rule_json = json.load(f)
            anti_rule_json = {}
            anti_rule_example_json = {}
            anti_rule_example_json["example-before"] = rule_json["example-split"]["example-after"]
            anti_rule_example_json["example-after"] = rule_json["example-split"]["example-before"]
            anti_rule_json["example-split"] = anti_rule_example_json
            # Construct the prompt for the Qwen model
            prompt = f"""
            Please consider this rule:

            {{
                "what-it-does": {rule_json["what-it-does"]},
                "why-is-this-bad": {rule_json["why-is-this-bad"]},
            }}

            This is an example of applying this rule on an input code to get an output code:

            input code: {rule_json["example-split"]["example-before"]}

            output code: {rule_json["example-split"]["example-after"]}

            Now please consider a rule that suggests the opposite of this:

            input code: {rule_json["example-split"]["example-after"]}

            output code: {rule_json["example-split"]["example-before"]}

            Your task is to try to justify this opposite rule and give me two things:

            1. what this opposite rule does - in a single line
            2. justify the opposite rule and why the original is bad - in one paragraph

            Do not use the word ‘opposite rule’ anywhere 
            output should be in the json format

            {{
                “what_opposite_rule_does”: ,
                “justification_for_opposite_rule”: ,
            }}

            """

            # Tokenize the input prompt
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)

            # Generate the response from the model
            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"],
                    max_length=500,  # Adjust as needed
                    num_return_sequences=1,
                    temperature=0.7,
                    top_p=1,
                    top_k=50,
                    no_repeat_ngram_size=2
                )

            # Decode the response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print("Model response:", response)

            json_object = json.loads(response)
            anti_rule_json["what-it-does"] = json_object["what_opposite_rule_does"]
            anti_rule_json["why-is-this-bad"] = json_object["justification_for_opposite_rule"]
            print("Darsh")
            print(anti_rule_json)
            time.sleep(0.5)