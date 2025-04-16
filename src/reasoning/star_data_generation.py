# self-taught reasoner with rationalization inspired pipeline.

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

from src.dpo.reward_model_newformat import evaluate_instance
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

SAMPLE_OUTPUT_FORMAT = """### Final Idiom Violations Found

**Idiom ERA001 Violations:**

{"line": " 12 \t\t#event = forms.ModelChoiceField(queryset=Inquiry.objects.filter(owner=kwargs.pop('user')))", "span": "#event = forms.ModelChoiceField(queryset=Inquiry.objects.filter(owner=kwargs.pop('user')))", "fix": null}

**Idiom C901 Violations:**

NO VIOLATIONS FOUND

**Idiom I001 Violations:**

{"line": "  1 from django import forms\n  2 from django.forms.models import inlineformset_factory\n  3 from .models import Request\n  4 from inquiry.models import *", "span": "from django import forms\nfrom django.forms.models import inlineformset_factory\nfrom .models import Request\nfrom inquiry.models import *", "fix": [{"before": "from django import forms\nfrom django.forms.models import inlineformset_factory\nfrom .models import Request\nfrom inquiry.models import *\n\n\n\n", "after": "from django import forms\nfrom django.forms.models import inlineformset_factory\nfrom inquiry.models import *\n\nfrom .models import Request\n\n\n"}]}

**Idiom I002 Violations:**

NO VIOLATIONS FOUND

**Idiom BLE001 Violations:**

NO VIOLATIONS FOUND"""

ANSWER_GENERATION_PROMPT_NO_GT = """Look at the following list of code idiom specifications with definitions and examples:

{LIST_OF_IDIOM_SPECS}

Given these idioms, your task is to look at a code file and detect violations of the above idioms, and flag them like a linter. You should also suggest a fix if possible. Report the results per idiom specification mentioned above and just say 'NO VIOLATIONS FOUND' if no violations are found for a given idiom. Do not detect any idioms not specified above.

Code file:
{CODE_FILE}

You should report your answer strictly in this format under a section called "### Final Idiom Violations Found" as shown below:

{SAMPLE_OUTPUT_FORMAT}

Now that you understand the output format, below are the code constructs you need to look at to detect the specified idioms.

### Code Constructs:

{CODE_CONSTRUCTS_FOR_META_TASK}

Given these code constructs, do a line by line analysis of the code file, but focus only on the relevant code constructs mentioned in "### Code Constructs" and find all idiom violations. While doing your analysis go through each line of code only once and check for each line if any of the 5 idioms apply. Do not do a separate analysis per idiom. Let's say line 6 employs a construct like a `for` loop which is releavant for idiom B007, while line 21 uses `exec()` which is relevant for idiom S102, your analysis should look roughly like:

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
- Strictly follow the output format shown above and include a "### Final Idiom Violations Found" section after you finish your analysis. Remember that if you don't include this section your answer will be invalid even if the analysis is correct !!
- Do not repeat the whole code file in your analysis. 
- Make sure to analyze all lines potentially containing idiom violations based on the "### Code Constructs".
- Don't literally mention every line, only mention ones that contain the constructs mentioned in the "### Code Constructs".

Step by step analysis:"""

RATIONALIZATION_PROMPT_WITH_GT = """Your answer was: 

{REASONER_INCORRECT_ANSWER}

However the expected "ground truth" answer according to a linter is:

{LINTER_CORRECT_ANSWER}

Please amend your analysis accordingly to arrive at the same answer as the ground truth, but do not mention your previous analysis or the ground truth answer in the fixed analysis. Now directly give the fixed analysis and the "### Final Idiom Violations Found" section with output that matches the ground truth linter output:

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

def get_gpt_response(messages: list[dict], model: str, max_tokens: int=512) -> str:
    if model == "o3-mini":
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=3000
        )
        response = response.choices[0].message.content
    else:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
        )
        response = response.choices[0].message.content

    return response

def generate_star_SFT_demonstrations(
        sft_data, stack_data: dict, code_idiom_specs: dict, 
        code_construct_cots: dict, task: str, 
        start_point: int=0, model: str="gpt-4o-mini",
    ) -> list[str]:
    encoder = tiktoken.encoding_for_model(model)
    cache_path = f"./data/ruff_meta_linting/cot_gen/{model}-{task}-cot-gen-cache_start_{start_point}.jsonl"
    print(cache_path)
    if os.path.exists(cache_path):
        cache = {rec["id"]: rec for rec in read_jsonl(cache_path)}
    else: cache = {}

    pbar = tqdm(
        enumerate(sft_data),
        total=len(sft_data)
    )
    skipped_files = 0
    failure_to_get_CoT = 0
    for ii,rec in pbar:
        if ii < start_point: continue
        if rec["id"] in cache: continue

        # get code file and check if it is super long or has some really long lines.
        is_weird_file = False
        is_long_file = False
        blob_id = rec["id"].split("_")[-1].strip()
        raw_file = stack_data[blob_id]["content"]
        is_long_file = len(raw_file.split("\n")) > 200
        for line in raw_file.split("\n"):
            if len(line) > 1000: 
                # print(line[:100])
                is_weird_file = True
        if is_weird_file or is_long_file: 
            skipped_files += 1
            continue

        # get list of idiom specs and idiom violations.
        blob_id = rec["id"].split("_")[-1].strip()
        idiom_codes_list = [code.strip() for code in rec["source"].split("/")[-1].split("-")]
        CODE_FILE = add_line_no_to_stack_file(raw_file)
        
        LIST_OF_IDIOM_SPECS = "\n\n".join([idiom_spec_extractor_for_ruff(code_idiom_specs[idiom_code]) for idiom_code in idiom_codes_list])
        
        IDIOM_VIOLATIONS = rec["messages"][-1]["content"]
        CODE_CONSTRUCTS_FOR_META_TASK = code_construct_cots[rec['source']]
        
        prompt = ANSWER_GENERATION_PROMPT_NO_GT.format(
            LIST_OF_IDIOM_SPECS=LIST_OF_IDIOM_SPECS, CODE_FILE=CODE_FILE, 
            CODE_CONSTRUCTS_FOR_META_TASK=CODE_CONSTRUCTS_FOR_META_TASK,
            SAMPLE_OUTPUT_FORMAT=SAMPLE_OUTPUT_FORMAT,
        )
        # tokens = encoder.encode(prompt, disallowed_special=())
        # if len(tokens)>2400: continue # 3000-600
        # # print(prompt)
        
        messages = [{"role": "user", "content": prompt}]
        model_response = get_gpt_response(messages=messages, model=model, max_tokens=1024)
        ground_truth = rec["messages"][-1]["content"]
        try: reward = evaluate_instance(model_response, ground_truth)
        except AttributeError: reward = 0
        
        print(reward)
        if reward == 1: pass
            # print(model_response)
            # print(ground_truth)
        else:
            # print(model_response)
            REASONER_INCORRECT_ANSWER = load_linter_results(model_response)
            LINTER_CORRECT_ANSWER = ground_truth
            print("Reasoner failed, moving on to rationalization!")
            messages = [
                {"role": "user", "content": prompt},
                {"role": "system", "content": model_response},
                {"role": "user", "content": RATIONALIZATION_PROMPT_WITH_GT.format(
                    REASONER_INCORRECT_ANSWER=REASONER_INCORRECT_ANSWER,
                    LINTER_CORRECT_ANSWER=LINTER_CORRECT_ANSWER,
                )},
            ]
            model_response = get_gpt_response(messages=messages, model=model, max_tokens=2048)
            # print(model_response)
            # print(model_response)
            # print(ground_truth)
            try: reward = evaluate_instance(model_response, ground_truth)
            except AttributeError: reward = 0
            if reward != 1: 
                failure_to_get_CoT += 1
                print(f"SKIPPED DATA instance: {ii}")
                pbar.set_description(f"CoTs skipped >= ({failure_to_get_CoT})")
                continue
        # result = load_linter_results(response)
        # expected_response = rec["messages"][-1]["content"]
        # expected_result = load_linter_results(expected_response)
        with open(cache_path, "a") as f:
            write_rec = {
                "id": rec["id"],
                "source": rec["source"],
                "code_file": raw_file,
                "prompt": prompt,
                "ground_truth": ground_truth,
                "model_response": model_response,
            }
            f.write(json.dumps(write_rec)+"\n")
    print(f"skipped: {skipped_files} files")

# main
if __name__ == "__main__":
    openai.api_key = os.environ["ORACLE_PROJECT_COT_API_KEY"]
    client = openai.OpenAI(api_key=os.environ["ORACLE_PROJECT_COT_API_KEY"])

    ### CODE CONSTRUCT LIST COT GENERATION

    # sft_train_data = json.load(open("./data/ruff_meta_linting/train_v4.json"))
    # code_idiom_specs = load_ruff_idiom_specs("./data/ruff_pages")
    
    # code_construct_cot_gen_prompts = generate_code_construct_for_meta_task_cot_prompts(sft_data=sft_train_data, code_idiom_specs=code_idiom_specs)
    # code_construct_cot_gen_data = []
    # for k,v in code_construct_cot_gen_prompts.items():
    #     print(k, len(v))
    #     code_construct_cot_gen_data.append({"id": k, "prompt": list(v)[0]})

    # with open("./data/ruff_meta_linting/cot_gen/code_construct_prompts_for_meta_tasks.json", "w") as f:
    #     json.dump(code_construct_cot_gen_data, f, indent=4)
    # generate_cots(code_construct_cot_gen_data, task="code_construct_v2", model="gpt-4o")
    
    ### SCAN FILE COT GENERATION

    code_construct_cots = {rec["id"]: rec["response"] for rec in read_jsonl("data/ruff_meta_linting/cot_gen/gpt-4o-code_construct_v2-cot-gen-cache_start_0.jsonl")}
    sft_train_data = json.load(open("./data/ruff_meta_linting/train_v4_new_format_with_lineno.json"))
    # sft_train_data = intelligent_downsample(sft_train_data)
    # exit()
    code_idiom_specs = load_ruff_idiom_specs("./data/ruff_pages")
    stack_data = load_stack_dump("./data/STACK-V2", as_dict=True)
    
    try: start_point = int(sys.argv[1])
    except IndexError: start_point = 0

    generate_star_SFT_demonstrations(
        sft_data=sft_train_data, stack_data=stack_data, 
        code_idiom_specs=code_idiom_specs, 
        code_construct_cots=code_construct_cots,
        task="star", model="gpt-4o-mini", start_point=0, 
    )