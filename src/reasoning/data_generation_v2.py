# script to generate CoTs where the model first complies a list of code constructs to be inspected for all the idioms in the meta-task. Given the list of code constructs for each code construct it finds instances of it in the file and analyzes if the construct applies.

import os
import sys
import json
import random
import openai
import pathlib
import tiktoken
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv()

module_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent)
sys.path.append(module_path)

from src.datautils import generate_cot_gen_prompts, load_ruff_idiom_specs, load_stack_dump, read_jsonl, idiom_spec_extractor_for_ruff

CODE_CONSTRUCT_COMPILATION_PROMPT = """Look at the following list of code idiom specifications with definitions and examples:

{LIST_OF_IDIOM_SPECS}

Given these idioms compile an exhaustive list of code constructs that should be closely analyzed as well as conditions that should be applied and fixes to be suggested (if provided in the list). 
Please note that for the conditions to be applied you should only include conditions explicitly mentioned in the definition. 
However you can try and extrapolate the examples provided to broaden the search of code constructs while strictly enforcing the definitions provided.
Remember to make the conditions exactly as specific as the definitions no more and no less.
Follow this format:

Idiom B012: jump-statement-in-finally
Code Constructs to Analyze: 
...

Condition:
...

Fix:
...
"""

COT_GENERATION_PROMPT_V2 = """Look at the following list of code idiom specifications with definitions and examples:

{LIST_OF_IDIOM_SPECS}

Given these idioms, your task is to look at a code file and detect violations of the above idioms, and flag them like a linter. You should also suggest a fix if possible. Report the results per idiom specification mentioned above and just say 'NO VIOLATIONS FOUND' if no violations are found for a given idiom. Do not detect any idioms not specified above.

Code file:
{CODE_FILE}

Now I'm going to give you the ground truth violations per idiom.

### Ground Truth Idiom Violations:

{IDIOM_VIOLATIONS}

Below are the code constructs you need to look at to detect them

### Code Constructs:

{CODE_CONSTRUCTS_FOR_META_TASK}

Given these code constructs, do a top to bottom analysis of the file looking at only the relevant code constructs mentioned in "### Code Comstructs" and arrive at the same violations as the ones given in "### Ground Truth Idiom Violations:". However do not make any reference to ground truth and pretend like you came up with these results on your own. Also try to brief and do not repeat the code file in your analysis only mention relevant lines with line numbers. Finally make sure to analyze all potential locations for idiom violations even the ones that are not a part of the "### Ground Truth Idiom Violations:"

Step by step analysis:"""

COT_GENERATION_PROMPT_V3 = """Look at the following list of code idiom specifications with definitions and examples:

{LIST_OF_IDIOM_SPECS}

Given these idioms, your task is to look at a code file and detect violations of the above idioms, and flag them like a linter. You should also suggest a fix if possible. Report the results per idiom specification mentioned above and just say 'NO VIOLATIONS FOUND' if no violations are found for a given idiom. Do not detect any idioms not specified above.

Code file:
{CODE_FILE}

Now I'm going to give you the ground truth violations per idiom.

### Ground Truth Idiom Violations:

{IDIOM_VIOLATIONS}

Below are the code constructs you need to look at to detect them

### Code Constructs:

{CODE_CONSTRUCTS_FOR_META_TASK}

Given these code constructs, do a line by line analysis of the code file, but focus only on the relevant code constructs mentioned in "### Code Constructs" and arrive at the same violations as the ones given in "### Ground Truth Idiom Violations:". While doing your analysis go through each line of code only once and check for each line if any of the 5 idioms apply. Do not do a separate analysis per idiom. Let's say line 6 employs a construct like a `for` loop which is releavant for idiom B007, while line 21 uses `exec()` which is relevant for idiom S102, your analysis should look roughly like:

Line 6: Detected `for` loop relevant to idiom B007:
 - Conditions for idiom B007 state that ...
 - Here ...
 - In conclusion B007 is not violated!

Line 21: Detected `exec` call which is relevant to idiom S102:
 - Conditions for idiom S102 state that ...
 - Here ...
 - In conclusion S102 is violated!

### Additional Instructions:

- Follow the reasoning format mentioned above strictly and cover the lines in a sorted order from the lowest line number to the highest.
- Do not make any reference to ground truth in your analysis and pretend you are doing this analysis on your own.
- Do not repeat the whole code file in your analysis. 
- Make sure to analyze all potential lines based on the "### Code Constructs" for all the idioms even the ones that don't end up in the "### Ground Truth Idiom Violations:". However while analyzing them provide reasoning for why they don't end up in the final result.
- Don't literally mention every line, only mention ones that contain the constructs mentioned in the "### Code Constructs".
- After finishing up the analysis summarize the results to match the "### Ground Truth Idiom Violations:". 

Step by step analysis:"""

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

def add_line_no_to_stack_file(stack_file: str):
    stack_file_with_lineno = []
    for lineno, line in enumerate(stack_file.split("\n")):
        stack_file_with_lineno.append(f"{str(lineno+1).rjust(3)} {line}")

    return "\n".join(stack_file_with_lineno)

def generate_code_construct_for_meta_task_cot_prompts(sft_data, code_idiom_specs: dict) -> list[str]:
    code_construct_cot_gen_prompts = defaultdict(lambda: set())
    for rec in tqdm(sft_data):
        # blob_id = rec["id"].split("_")[-1].strip()
        idiom_codes_list = [code.strip() for code in rec["source"].split("/")[-1].split("-") if code not in ["E501", "Q000", "ANN001", "W191", "ANN201"]]
        
        LIST_OF_IDIOM_SPECS = "\n\n".join([idiom_spec_extractor_for_ruff(code_idiom_specs[idiom_code]) for idiom_code in idiom_codes_list if idiom_code not in ["E501", "Q000", "ANN001", "W191", "ANN201"]])

        prompt = CODE_CONSTRUCT_COMPILATION_PROMPT.format(
            LIST_OF_IDIOM_SPECS=LIST_OF_IDIOM_SPECS
        )
        code_construct_cot_gen_prompts[rec['source']].add(prompt)

    return code_construct_cot_gen_prompts

def intelligent_downsample(sft_train_data: list[dict], model: str="o3-mini"):
    encoder = tiktoken.encoding_for_model(model)
    random.seed(42)
    strata = defaultdict(lambda: [])
    prompt_lens = []
    for rec in tqdm(sft_train_data):
        prompt = rec["messages"][0]["content"]
        prompt_len = len(encoder.encode(prompt, disallowed_special=()))
        if prompt_len > 1000: continue # exclude prompts with more than 2000 tokens.
        response = rec["messages"][-1]["content"]
        source = rec["source"]
        linter_results = load_linter_results(response)
        if len(linter_results) == 0: 
            strata[f"{source}_NO_VIOLATIONS"].append(rec)
        else:
            num_unique_codes = len(set(violation['code'] for violation in linter_results))
            strata[f"{source}_{num_unique_codes}_VIOLATIONS"].append(rec)
    print(f"original data size: {len(sft_train_data)}")
    downsampled_data = []
    print("no. strata (atmost 60 possible)", len(strata))
    for k,v in strata.items():
        stratum_datum = random.sample(v, k=min(len(v), 100))
        downsampled_data.extend(stratum_datum)
        print(k, len(stratum_datum))
    print(f"downsampled data size: {len(downsampled_data)}")
    # exit()
    return downsampled_data

def estimate_gpt_cost(input_texts, output_tokens=512, input_cost_per_million=0.15, output_cost_per_million=0.6):
    """
    Estimates the cost of using GPT-4o-mini on a given dataset.
    
    Parameters:
    - input_texts (list of str): List of input prompts.
    - output_tokens (int): Fixed number of output tokens per input (default: 512).
    - input_cost_per_million (float): Cost per million input tokens ($0.15 by default).
    - output_cost_per_million (float): Cost per million output tokens ($0.6 by default).
    
    Returns:
    - total_cost (float): Estimated cost in USD.
    """
    enc = tiktoken.encoding_for_model("gpt-4o")
    input_lens = [len(enc.encode(text, disallowed_special=())) for text in tqdm(input_texts)]
    total_input_tokens = sum(input_lens)
    total_output_tokens = len(input_texts) * output_tokens
    
    input_cost = (total_input_tokens / 1_000_000) * input_cost_per_million
    output_cost = (total_output_tokens / 1_000_000) * output_cost_per_million
    
    total_cost = input_cost + output_cost

    print(f"${input_cost:.2f}")
    print(f"${output_cost:.2f}")
    print(f"avg input token count: {total_input_tokens/len(input_lens)}")
    print(f"max input token count: {np.max(input_lens)}")
    print(f"short enough instances: {sum([int(input_len+512 <= 3000) for input_len in input_lens])}")
    
    return total_cost

def generate_idiom_det_and_loc_cot_prompts(sft_data, stack_data: dict, code_idiom_specs: dict, code_construct_cots: dict, save_path: str) -> list[str]:
    open(save_path, "w")
    cot_gen_prompts = []
    for rec in tqdm(sft_data):
        blob_id = rec["id"].split("_")[-1].strip()
        idiom_codes_list = [code.strip() for code in rec["source"].split("/")[-1].split("-")]

        # get code file, list of idiom specs and idiom violations.
        CODE_FILE = add_line_no_to_stack_file(stack_data[blob_id]["content"])
        
        LIST_OF_IDIOM_SPECS = "\n\n".join([idiom_spec_extractor_for_ruff(code_idiom_specs[idiom_code]) for idiom_code in idiom_codes_list])
        
        IDIOM_VIOLATIONS = rec["messages"][-1]["content"]
        CODE_CONSTRUCTS_FOR_META_TASK = code_construct_cots[rec['source']]
        
        prompt = COT_GENERATION_PROMPT_V3.format(
            LIST_OF_IDIOM_SPECS=LIST_OF_IDIOM_SPECS,
            CODE_FILE=CODE_FILE, IDIOM_VIOLATIONS=IDIOM_VIOLATIONS,
            CODE_CONSTRUCTS_FOR_META_TASK=CODE_CONSTRUCTS_FOR_META_TASK,
        )
        write_rec = {"blob_id": blob_id, "id": rec["id"], "prompt": prompt}
        cot_gen_prompts.append(write_rec)
        with open(save_path, "a") as f:
            f.write(json.dumps(write_rec)+"\n")
        
    return cot_gen_prompts

def generate_cots(prompts: list[str], task: str, model: str="gpt-4o-mini", start_point=0):
    encoder = tiktoken.encoding_for_model(model)
    cache_path = f"./data/ruff_meta_linting/cot_gen/{model}-{task}-cot-gen-cache_start_{start_point}.jsonl"
    print(cache_path)
    if os.path.exists(cache_path):
        cache = {rec["id"]: rec for rec in read_jsonl(cache_path)}
    else: cache = {}
    break_ctr = 0
    pbar = tqdm(
        enumerate(prompts), 
        total=len(prompts), 
        desc="getting ChatGPT CoT"
    )
    for ii,rec in pbar:
        if ii < start_point: continue
        if rec["id"] in cache: 
            break_ctr += 1
            continue
        pbar.set_description(f"CoTs obtained for ({break_ctr})")
        prompt = rec["prompt"]
        tokens = encoder.encode(prompt, disallowed_special=())
        if len(tokens)>2400: continue # 3000-600
        # print(prompt)
        if model == "o3-mini":
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=3000
            )
            response = response.choices[0].message.content
        else:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024
            )
            response = response.choices[0].message.content

        with open(cache_path, "a") as f:
            write_rec = {}
            write_rec.update(rec)
            write_rec["response"] = response
            f.write(json.dumps(write_rec)+"\n")
            break_ctr += 1
        if break_ctr >= 1000000: break

# main
if __name__ == "__main__":
    openai.api_key = os.environ["ORACLE_PROJECT_COT_API_KEY"]
    client = openai.OpenAI(api_key=os.environ["ORACLE_PROJECT_COT_API_KEY"])

    ### CODE CONSTRUCT LIST COT GENERATION

    # sft_train_data = json.load(open("./data/ruff_meta_linting/all_idioms/train.json"))
    # code_idiom_specs = load_ruff_idiom_specs("./data/ruff_pages_new")
    
    # code_construct_cot_gen_prompts = generate_code_construct_for_meta_task_cot_prompts(sft_data=sft_train_data, code_idiom_specs=code_idiom_specs)
    # code_construct_cot_gen_data = []
    # for k,v in code_construct_cot_gen_prompts.items():
    #     print(k, len(v))
    #     code_construct_cot_gen_data.append({"id": k, "prompt": list(v)[0]})

    # with open("./data/ruff_meta_linting/cot_gen/code_construct_prompts_for_meta_tasks.json", "w") as f:
    #     json.dump(code_construct_cot_gen_data, f, indent=4)
    # generate_cots(code_construct_cot_gen_data, task="code_construct_all_idioms", model="gpt-4o")
    
    ### SCAN FILE COT GENERATION

    code_construct_cots = {rec["id"]: rec["response"] for rec in read_jsonl("data/ruff_meta_linting/cot_gen/gpt-4o-code_construct_all_idioms-cot-gen-cache_start_0.jsonl")}
    sft_train_data = json.load(open("./data/ruff_meta_linting/all_idioms/train.json"))
    # sft_train_data = intelligent_downsample(sft_train_data)
    code_idiom_specs = load_ruff_idiom_specs("./data/ruff_pages_new")
    stack_data = load_stack_dump("./data/STACK-V2", as_dict=True)
    
    generate_idiom_det_and_loc_cot_prompts(
        sft_train_data, stack_data, code_idiom_specs, code_construct_cots, 
        "./data/ruff_meta_linting/cot_gen/train_all_idioms_cot.jsonl"
    )
    prompts = read_jsonl("./data/ruff_meta_linting/cot_gen/train_all_idioms_cot.jsonl")

    # estimate_gpt_cost([rec["prompt"] for rec in prompts], output_tokens=600, output_cost_per_million=0.15, input_cost_per_million=0.6)
    # # estimate_gpt_cost([rec["prompt"] for rec in prompts], output_tokens=600, output_cost_per_million=1.1, input_cost_per_million=4.4)

    try: start_point = int(sys.argv[1])
    except IndexError: start_point = 0
    # # generate_cots(prompts, task="loc_and_det_cot_v3", model="o3-mini", start_point=start_point)
    generate_cots(prompts, task="loc_and_det_cot_v3", model="gpt-4o-mini", start_point=start_point)