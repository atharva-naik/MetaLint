# script to generate CoTs where the model first complies a list of code constructs to be inspected for all the idioms in the meta-task. Given the list of code constructs for each code construct it finds instances of it in the file and analyzes if the construct applies.

import os
import sys
import json
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

Given these idioms compile an exhaustive list of code constructs that should be closely analyzed within a given Python code to figure out whether the idiom is present in the code."""

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

Given these code constructs, do a top to bottom analysis of the file looking at only the relevant code constructs mentioned in "### Code Comstructs" and arrive at the same violations as the ones given in "### Ground Truth Idiom Violations:". However do not make any reference to ground truth and pretend like you came up with these results on your own. Also try to brief and do not repeat the code file in your analysis only mention relevant lines with line numbers.

Step by step analysis:"""

def add_line_no_to_stack_file(stack_file: str):
    stack_file_with_lineno = []
    for lineno, line in enumerate(stack_file.split("\n")):
        stack_file_with_lineno.append(f"{str(lineno+1).rjust(3)} {line}")

    return "\n".join(stack_file_with_lineno)

def generate_code_construct_for_meta_task_cot_prompts(sft_data, code_idiom_specs: dict) -> list[str]:
    code_construct_cot_gen_prompts = defaultdict(lambda: set())
    for rec in tqdm(sft_data):
        # blob_id = rec["id"].split("_")[-1].strip()
        idiom_codes_list = [code.strip() for code in rec["source"].split("/")[-1].split("-") if code not in ["ANN001", "ANN201"]]
        
        LIST_OF_IDIOM_SPECS = "\n\n".join([idiom_spec_extractor_for_ruff(code_idiom_specs[idiom_code]) for idiom_code in idiom_codes_list if idiom_code not in ["ANN001", "ANN201"]])

        prompt = CODE_CONSTRUCT_COMPILATION_PROMPT.format(
            LIST_OF_IDIOM_SPECS=LIST_OF_IDIOM_SPECS
        )
        code_construct_cot_gen_prompts[rec['source']].add(prompt)

    return code_construct_cot_gen_prompts

def generate_code_construct_for_meta_task_cot_prompts(sft_data, code_idiom_specs: dict) -> list[str]:
    code_construct_cot_gen_prompts = defaultdict(lambda: set())
    for rec in tqdm(sft_data):
        # blob_id = rec["id"].split("_")[-1].strip()
        idiom_codes_list = [code.strip() for code in rec["source"].split("/")[-1].split("-") if code not in ["ANN001", "ANN201"]]
        
        LIST_OF_IDIOM_SPECS = "\n\n".join([idiom_spec_extractor_for_ruff(code_idiom_specs[idiom_code]) for idiom_code in idiom_codes_list if idiom_code not in ["ANN001", "ANN201"]])

        prompt = CODE_CONSTRUCT_COMPILATION_PROMPT.format(
            LIST_OF_IDIOM_SPECS=LIST_OF_IDIOM_SPECS
        )
        code_construct_cot_gen_prompts[rec['source']].add(prompt)

    return code_construct_cot_gen_prompts

def estimate_gpt4o_mini_cost(input_texts, output_tokens=512, input_cost_per_million=0.15, output_cost_per_million=0.6):
    """
    Estimates the cost of using GPT-4o-mini on a given dataset.
    
    Parameters:
    - input_texts (list of str): List of input prompts.
    - output_tokens (int): Fixed number of output tokens per input (default: 500).
    - input_cost_per_million (float): Cost per million input tokens ($0.25 by default).
    - output_cost_per_million (float): Cost per million output tokens ($1.25 by default).
    
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
        
        prompt = COT_GENERATION_PROMPT_V2.format(
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
                max_tokens=512
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

    # # CODE CONSTRUCT LIST COT GENERATION
    # sft_train_data = json.load(open("./data/ruff_meta_linting/train_v4.json"))
    # code_idiom_specs = load_ruff_idiom_specs("./data/ruff_pages")
    
    # code_construct_cot_gen_prompts = generate_code_construct_for_meta_task_cot_prompts(sft_data=sft_train_data, code_idiom_specs=code_idiom_specs)
    # code_construct_cot_gen_data = []
    # for k,v in code_construct_cot_gen_prompts.items():
    #     print(k, len(v))
    #     code_construct_cot_gen_data.append({"id": k, "prompt": list(v)[0]})

    # with open("./data/ruff_meta_linting/cot_gen/code_construct_prompts_for_meta_tasks.json", "w") as f:
    #     json.dump(code_construct_cot_gen_data, f, indent=4)
    # generate_cots(code_construct_cot_gen_data, task="code_construct", model="gpt-4o")

    # with open("./data/ruff_meta_linting/cot_gen/code_construct_cots_for_meta_tasks.json", "w") as f:
    #     json.dump(code_construct_cot_gen_cots, f, indent=4)
    
    # code_construct_cots = {rec["id"]: rec["response"] for rec in read_jsonl("data/ruff_meta_linting/cot_gen/gpt-4o-code_construct-cot-gen-cache.jsonl")}
    # sft_train_data = json.load(open("./data/ruff_meta_linting/train_v4_new_format_with_lineno.json"))
    # code_idiom_specs = load_ruff_idiom_specs("./data/ruff_pages")
    # stack_data = load_stack_dump("./data/STACK-V2", as_dict=True)
    
    # generate_idiom_det_and_loc_cot_prompts(sft_train_data, stack_data, code_idiom_specs, code_construct_cots, "./data/ruff_meta_linting/cot_gen/train_v4_cot_v2.jsonl")
    
    start_point = int(sys.argv[1])
    prompts = read_jsonl("./data/ruff_meta_linting/cot_gen/train_v4_cot_v2.jsonl")
    generate_cots(prompts, task="loc_and_det_cot", model="gpt-4o-mini", start_point=start_point)