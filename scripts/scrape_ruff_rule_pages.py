import os
import bs4
import time
import json
import requests
from tqdm import tqdm

def parse_html_table(table: bs4.element.Tag) -> dict[str, str]:
    field_names = [ele.text for ele in table.find_all("th")]
    rows = table.find_all("tr")
    table_json = []
    for row in rows[1:]:
        field_values = [td.text for td in row.find_all("td")]
        fix_information_td = row.find_all("td")[3]
        is_fix_not_available_span_present = fix_information_td.find('span', {'title': 'Automatic fix not available'}) is not None
        is_fix_available_span_present = fix_information_td.find('span', {'title': 'Automatic fix available'}) is not None
        table_row_json = {k: v for k,v in zip(field_names, field_values)}
        table_row_json[""] = [LEGEND[symbol.strip()] for symbol in table_row_json[""] if symbol.strip() in LEGEND]
        if is_fix_not_available_span_present:
            table_row_json["fix_available"] = False
            print("Darsh")
        elif is_fix_available_span_present:
            table_row_json["fix_available"] = True
            print("Darsh")
        table_json.append(table_row_json)

    return table_json

LEGEND = {
    '✔': "Stable",
    "🧪": 'Unstable/Preview',
    "⚠️": "Depreceated",
    "❌": "Deleted",
    "🛠": "Automatically Fixable",
}

def download_ruff_rules_page(rules_page_url: str="https://docs.astral.sh/ruff/rules/"):
    html_source = requests.get(rules_page_url).text
    soup = bs4.BeautifulSoup(html_source, features="html.parser")
    rule_tables = soup.find_all("table")
    ruff_rules_metadata_json = []
    for table in rule_tables:
        ruff_rules_metadata_json.extend(parse_html_table(table))
    with open("../data/darsh_ruff_pages/rule_metadata.json", "w") as f:
        json.dump(ruff_rules_metadata_json, f, indent=4)

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
    ruff_rules_metadata_json = download_ruff_rules_page()
    rule_desc_page_url = "https://docs.astral.sh/ruff/rules/{Name}/"
    for rule in tqdm(ruff_rules_metadata_json):
        print(rule_desc_page_url.format(Name=rule['Name']))
        desc_page_code_soup = bs4.BeautifulSoup(requests.get(rule_desc_page_url.format(Name=rule['Name'])).text, features="html.parser")
        desc_page_json = parse_rule_spec(desc_page_code_soup)
        time.sleep(0.5)
        with open(f"../data/darsh_ruff_pages/rules/{rule['Code']}.json", "w") as f:
            json.dump(desc_page_json, f, indent=4)