# script to convert the PEP benchmark to a format that aligns with MetaLint (the format used for both trained and untrained Qwen models).

import os
import json
import pandas as pd

META_LINTING_PROMPT_V2 = """Look at the following list of code idiom specifications with definitions and examples:
{LIST_OF_IDIOM_SPECS}

Given these idioms, your task is to look at a code file and detect violations of the above idioms, and flag them like a linter. You should also suggest a fix if possible. Report the results per idiom specification mentioned above and just say 'NO VIOLATIONS FOUND' if no violations are found for a given idiom. Do not detect any idioms not specified above.

Code file:
{CODE_FILE}

Violations per idiom:
"""

all_pep_idiom_specs = pd.read_csv("data/pep_benchmark/list_of_pep_idiom_specs.csv").to_dict("records")
# print(all_pep_idiom_specs[0].keys())
all_pep_idiom_specs = {str(rec['code']).strip(): rec for rec in all_pep_idiom_specs}
print(all_pep_idiom_specs.keys())

def idiom_spec_extractor_for_pep(idiom_spec):
    if "example" in idiom_spec:
        Example = f"\n\nExample:\n{idiom_spec['example']}"
    else: Example = ""
    return f"# Idiom {idiom_spec['code']} ({idiom_spec['name']})\n\nDefinition: {idiom_spec['what-it-does']}\n\nRationale: {idiom_spec['why-is-this-bad']}"+Example

def read_jsonl(path: str, disable: bool=False) -> list[dict]:
    data = []
    with open(path, "r", encoding='utf-8') as f:
        for line in tqdm(f, disable=disable):
            data.append(json.loads(line.strip()))

    return data

def add_line_no_to_code_file(code_file: str):
    code_file_with_lineno = []
    for lineno, line in enumerate(code_file.split("\n")):
        code_file_with_lineno.append(f"{str(lineno+1).rjust(3)} {line}")

    return "\n".join(code_file_with_lineno)

def generate_label(annot: dict, code, pep_code: str):
    code_with_linenos_lines = add_line_no_to_code_file(code).split("\n")
    violations = []
    for rec in annot['violations']:
        line = "\n".join(code_with_linenos_lines[rec['start_lineno']-1:rec['end_lineno']])
        assert line.strip() != "", annot['blob_id']+f"_{pep_code}"
        violations.append(
            json.dumps({
                "line": line, "fix": rec['fix']
            })
        )
    violations = "\n".join(violations)

    return f"""**Idiom {pep_code} Violations:**
    
{violations}"""

def generate_input_prompt(code: str, pep_spec: dict):
    LIST_OF_IDIOM_SPECS = "\n\n".join([idiom_spec_extractor_for_pep(pep_spec)])

    return META_LINTING_PROMPT_V2.format(
        LIST_OF_IDIOM_SPECS=LIST_OF_IDIOM_SPECS,
        CODE_FILE=add_line_no_to_code_file(code),
    )

# main
if __name__ == "__main__":
    annot_folder = "data/pep_benchmark/annotations"
    source_folder = "data/pep_benchmark/code_files"
    pep_test_data = []
    for filename in os.listdir(annot_folder):
        filepath = os.path.join(annot_folder, filename)
        filestem,_ = os.path.splitext(filename)
        try: 
            annot = json.load(open(filepath))
            code_filepath = os.path.join(source_folder, filestem+".py")
            source_code = open(code_filepath).read()
            ID = filestem
            PEP = str(int(filestem.split('_')[1].strip()))
            pep_spec = all_pep_idiom_specs[PEP]
            pep_test_data.append({
                "id": ID,
                "code": source_code,
                "messages": [
                    {"content": generate_input_prompt(code=source_code, pep_spec=pep_spec), "role": "user"},
                    {"content": generate_label(annot, code=source_code, pep_code=PEP), "role": "system"}
                ],
                "source": f"pep_benchmark/PEP{PEP}"
            })
        except json.JSONDecodeError: print(filepath)
        except KeyError: print(filepath); exit()
        # print(annot)
    with open("data/pep_benchmark/test_pep.json", "w") as f:
        json.dump(pep_test_data, f, indent=4)
    # print(pep_test_data[0]['messages'][0]['content'])
    # print(pep_test_data[0]['messages'][1]['content'])
    print(list(set([rec['source'] for rec in pep_test_data])))
    # print(len(pep_test_data))