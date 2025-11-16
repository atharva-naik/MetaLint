import json
from tqdm import tqdm

def read_jsonl(path: str, disable: bool=False) -> list[dict]:
    data = []
    with open(path, "r", encoding='utf-8') as f:
        for line in tqdm(f, disable=disable):
            data.append(json.loads(line.strip()))

    return data

def write_jsonl(data, path: str):
    bytes_written = 0
    with open(path, "w") as f:
        for rec in tqdm(data):
            bytes_written += f.write(json.dumps(rec)+"\n")

    return bytes_written

def dict_union(d1: dict, d2: dict):
    out = d1.copy()
    out.update(d2)
    return out

def dict_intersection(d1: dict, d2: dict):
    return {k: d1[k] for k in d1 if k in d2}

def extract_line_nos(violation: dict):
    line_nos = set()
    try:
        for l in violation['line'].split("\n"):
            try: 
                line_nos.add(int(l.split()[0].strip().removesuffix(":")))
            except AttributeError: pass   
            except ValueError: pass # print(model_violation['line'])
            except KeyError: pass # print(model_violation.keys())
            except IndexError: pass
    except AttributeError: pass   
    except ValueError: pass # print(model_violation['line'])
    except KeyError: pass # print(model_violation.keys())
    except IndexError: pass

    return line_nos

IDIOMS_FOUND_HEADER = "## Idiom Violations Found"
def load_linter_results(text, pep: bool=False):
    if not isinstance(text, str): return [] # responses were null sometimes for gpt-oss-20b.
    results = []
    idiom_code = None
    for line in text.split("\n"):
        line = line.strip()
        if line == "": continue
        elif line.startswith("**Idiom") and line.endswith("Violations:**"):
            if pep: # less harsh checking for PEP benchmark.
                idiom_code = line.removeprefix("**Idiom").strip().split()[0].strip()
                # print(idiom_code)
            # more harsh checking for Ruff meta-linting dataset.
            else: idiom_code = line.removesuffix("Violations:**").removeprefix("**Idiom").strip()
            # exit()
            # 
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

NEWLINE = "\n"
# main
if __name__ == "__main__":
    preds1 = read_jsonl("data/pep_benchmark_preds_v2/qwen3_4b_dpo_run2_no_violations_0.05_400_transfer_v5_preds.jsonl")
    preds2 = read_jsonl("data/pep_benchmark_preds_v2/qwen3_4b_think_dpo_run3_no_violations_0.05_600_transfer_v5_preds.jsonl")
    union_preds = [] 
    intersection_preds = []

    for rec1, rec2 in tqdm(zip(preds1, preds2)):
        assert rec1['id'] == rec2['id']
        # print(rec1['model_response'])
        model_response1 = rec1['model_response']
        linter_results1 = load_linter_results(model_response1)
        model_response2 = rec2['model_response']
        linter_results2 = load_linter_results(model_response2)
        # print(linter_results1)
        flagged_line_dict1 = {}
        for result in linter_results1:
            linenos = extract_line_nos(result)
            for lineno in linenos:
                flagged_line_dict1[f"{lineno}-{result['code']}"] = result

        flagged_line_dict2 = {}
        for result in linter_results2:
            linenos = extract_line_nos(result)
            for lineno in linenos:
                flagged_line_dict2[f"{lineno}-{result['code']}"] = result
        # print(flagged_line_dict1)
        # print(flagged_line_dict2)
        if len(flagged_line_dict2) == 0:
            union_response = model_response1
            intersection_response = model_response2
        elif len(flagged_line_dict1) == 0:
            union_response = model_response2
            intersection_response = model_response1
        else:
            flagged_line_dict_union = dict_union(flagged_line_dict1, flagged_line_dict2)
            flagged_line_dict_intersection = dict_intersection(flagged_line_dict1, flagged_line_dict2)
            code1 = linter_results1[0]['code']
            code2 = linter_results2[0]['code']
            assert code1 == code2
            code = code1
            union_response = f"""### Final Idiom Violations Found
            
**Idiom {code} Violations:**

{NEWLINE.join(flagged_line_dict_union)}"""
            intersection_response = f"""### Final Idiom Violations Found
            
**Idiom {code} Violations:**

{NEWLINE.join(flagged_line_dict_intersection)}"""
        union_preds.append({
            "id": rec1['id'],
            "model_response": union_response,
            "ground_truth": rec1['ground_truth'],
            "erorr": rec1["error"],
        })
        intersection_preds.append({
            "id": rec1['id'],
            "model_response": intersection_response,
            "ground_truth": rec1['ground_truth'],
            "erorr": rec1["error"],
        })
        # print(union_response)
        # print(intersection_response)
        # exit()

    write_jsonl(union_preds, "data/pep_benchmark_preds_v2/qwen3_4b_dpo_run2_no_violations_0.05_400_union_dpo_run3_no_violations_0.05_600_transfer_v5_preds.jsonl")
    write_jsonl(intersection_preds, "data/pep_benchmark_preds_v2/qwen3_4b_dpo_run2_no_violations_0.05_400_intersection_dpo_run3_no_violations_0.05_600_transfer_v5_preds.jsonl")