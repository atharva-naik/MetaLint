import os
import sys
import json
import pathlib
from tqdm import tqdm
from collections import Counter

module_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent)
sys.path.append(module_path)

from src.datautils import generate_cot_gen_prompts, load_ruff_idiom_specs, load_stack_dump, read_jsonl

FINAL_IDIOMS_FOUND_TEMPLATE_FOR_NO_VIOLATIONS_5 = """

### Final Idiom Violations Found

**Idiom {I1} Violations:**

NO VIOLATIONS FOUND

**Idiom {I2} Violations:**

NO VIOLATIONS FOUND

**Idiom {I3} Violations:**

NO VIOLATIONS FOUND

**Idiom {I4} Violations:**

NO VIOLATIONS FOUND

**Idiom {I5} Violations:**

NO VIOLATIONS FOUND
"""

FINAL_IDIOMS_FOUND_TEMPLATE_FOR_NO_VIOLATIONS_4 = """

### Final Idiom Violations Found

**Idiom {I1} Violations:**

NO VIOLATIONS FOUND

**Idiom {I2} Violations:**

NO VIOLATIONS FOUND

**Idiom {I3} Violations:**

NO VIOLATIONS FOUND

**Idiom {I4} Violations:**

NO VIOLATIONS FOUND
"""

FINAL_IDIOMS_FOUND_TEMPLATE_FOR_NO_VIOLATIONS_3 = """

### Final Idiom Violations Found

**Idiom {I1} Violations:**

NO VIOLATIONS FOUND

**Idiom {I2} Violations:**

NO VIOLATIONS FOUND

**Idiom {I3} Violations:**

NO VIOLATIONS FOUND
"""

def load_linter_results(text):
    results = []
    idiom_code = None
    for line in text.split("\n"):
        line = line.strip()
        if line == "": continue
        elif line.startswith("**Idiom") and line.endswith("Violations:**"):
            idiom_code = line.removesuffix("Violations:**").removeprefix("**Idiom").strip()
        elif line == "NO VIOLATIONS FOUND": continue
        else:
            try: 
                result = json.loads(line)
                result["code"] = idiom_code
                results.append(result)
            except Exception as e: pass
                # print(e)
                # print(f"{e}: {line}")
    return results

# main
if __name__ == "__main__":
    NO_VIOLATIONS_FOUND_COUNT = 0
    dataset_version = "all_idioms/train"
    train_data = json.load(open(f"./data/ruff_meta_linting/{dataset_version}.json"))
    list_constructs = {rec["id"]: rec['response'] for rec in read_jsonl("data/ruff_meta_linting/cot_gen/gpt-4o-code_construct_all_idioms-cot-gen-cache_start_0.jsonl")}
    
    start_points = [0,8500]
    # [0,6815,8000,9938]
    cot_data = []
    for start_point in start_points:
        start_point = "" if start_point is None else f"_start_{start_point}"
        # data/ruff_meta_linting/cot_gen/gpt-4o-mini-all_idioms_star-cot-gen-cache_start_0.jsonl
        cot_data_path = f"data/ruff_meta_linting/cot_gen/gpt-4o-mini-all_idioms_star-cot-gen-cache{start_point}.jsonl"
        # print(cot_data_path)
        cot_data += read_jsonl(cot_data_path)
    print(len(cot_data))
    cot_data = {rec["id"]: rec for rec in cot_data}
    print(len(cot_data))
        
    train_data_with_cot = []
    skipped_due_to_incorrect_output_format = 0
    skipped_due_to_reference_to_final_output = 0
    for rec in tqdm(train_data):
        cot_rec = cot_data.get(rec["id"])
        if cot_rec is not None:
            scan_file_cot = cot_rec['model_response']
            if "### Final Idiom Violations Found" not in scan_file_cot: continue
            else:
                sections = scan_file_cot.split("### Final Idiom Violations Found")
                try: 
                    assert len(sections) == 2#, "'### Final Idiom Violations Found' occurs multiple times"
                except AssertionError:
                    skipped_due_to_reference_to_final_output += 1
                    print(f"skipping reference to final ouput: {skipped_due_to_reference_to_final_output}")
                    # print(scan_file_cot)
                    continue
                idioms_found = sections[-1].strip()
                before_final_idioms = sections[0].strip()
                idiom_codes_list = [code.strip() for code in rec["source"].split("/")[-1].split("-")]
                if "**Idiom " not in idioms_found:
                    if idioms_found.split("\n")[0].strip().startswith("NO VIOLATIONS FOUND"):
                        # print(idioms_found)
                        if len(idiom_codes_list) == 5:
                            idioms_found = FINAL_IDIOMS_FOUND_TEMPLATE_FOR_NO_VIOLATIONS_5.format(
                                I1=idiom_codes_list[0],
                                I2=idiom_codes_list[1],
                                I3=idiom_codes_list[2],
                                I4=idiom_codes_list[3],
                                I5=idiom_codes_list[4],
                            )
                        elif len(idiom_codes_list) == 4:
                            idioms_found = FINAL_IDIOMS_FOUND_TEMPLATE_FOR_NO_VIOLATIONS_4.format(
                                I1=idiom_codes_list[0],
                                I2=idiom_codes_list[1],
                                I3=idiom_codes_list[2],
                                I4=idiom_codes_list[3],
                            )
                        elif len(idiom_codes_list) == 3:
                            idioms_found = FINAL_IDIOMS_FOUND_TEMPLATE_FOR_NO_VIOLATIONS_3.format(
                                I1=idiom_codes_list[0],
                                I2=idiom_codes_list[1],
                                I3=idiom_codes_list[2],
                            )                           
                        # print(idioms_found)
                        scan_file_cot = before_final_idioms.strip("\n")+idioms_found
                        # print(scan_file_cot)
                    else:
                        # print(idioms_found)
                        skipped_due_to_incorrect_output_format += 1
                        print(f"skipping incorrect output format: {skipped_due_to_incorrect_output_format}")
                        continue
                        # print(scan_file_cot)
                        # exit()
                assert "**Idiom " in idioms_found
            result = load_linter_results(scan_file_cot)
            if len(result) == 0: NO_VIOLATIONS_FOUND_COUNT += 1
            list_constructs_cot = list_constructs[rec["source"]]
            response = rec["messages"][-1]['content']
            # print(cot)
            # print(response)
            newline = "\n"
            rec["messages"][-1]['content'] = f"""### Code Constructs

{list_constructs_cot.strip(newline)}

{scan_file_cot.strip(newline)}"""
            # print(rec["messages"][-1]['content'])
            # exit()
            train_data_with_cot.append(rec)
            # print(list_constructs_cot)
            # print(rec["messages"][-1]["content"])
            # exit()
    # print(f"% cot train data with violations: {sum(['NO VIOLATIONS FOUND' not in rec["messages"][-1]['content'] for rec in train_data_with_cot])/len(train_data_with_cot)}")
    print("\n".join([f"{key}: {count}" for (key,count) in Counter([rec['source'] for rec in train_data_with_cot]).most_common()]))
    print(f"NO VIOLATIONS FOUND: {(100*NO_VIOLATIONS_FOUND_COUNT/len(train_data_with_cot)):.2f}%")
    print(f"train_data original: {len(train_data)}")
    print(f"train_data with CoT: {len(train_data_with_cot)}")
    print(f"data/ruff_meta_linting/{dataset_version}_subtask_cot_star.json")
    with open(f"data/ruff_meta_linting/{dataset_version}_subtask_cot_star.json", "w") as f:
        json.dump(train_data_with_cot, f, indent=4)