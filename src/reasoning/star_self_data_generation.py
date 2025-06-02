import os
import sys
import json
import pathlib
import argparse
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

module_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent)
sys.path.append(module_path)

from src.datautils import read_jsonl
from src.dpo.reward_model_newformat_no_span import evaluate_instance
# from src.datautils import generate_cot_gen_prompts, load_ruff_idiom_specs, load_stack_dump, read_jsonl, idiom_spec_extractor_for_ruff

SAMPLE_OUTPUT_FORMAT_VIOLATIONS = """### Final Idiom Violations Found

**Idiom XYZ Violations:**

{"line": " 12 \\t\\t#event = forms.ModelChoiceField(queryset=Inquiry.objects.filter(owner=kwargs.pop('user')))", "fix": null}
{"line": "  1 from django import forms\\n  2 from django.forms.models import inlineformset_factory\\n  3 from .models import Request\\n  4 from inquiry.models import *", "fix": [{"before": "from django import forms\\nfrom django.forms.models import inlineformset_factory\\nfrom .models import Request\\nfrom inquiry.models import *\\n\\n\\n\\n", "after": "from django import forms\\nfrom django.forms.models import inlineformset_factory\\nfrom inquiry.models import *\\n\\nfrom .models import Request\\n\\n\\n"}]}
"""

SAMPLE_OUTPUT_FORMAT_NO_VIOLATIONS = """### Final Idiom Violations Found

**Idiom XYZ Violations:**

NO VIOLATIONS FOUND
"""

def add_cot_gen_instr_to_prompt(input_prompt: str):
    assert input_prompt.strip("\n").endswith("Violations per idiom:")
    input_prompt = input_prompt.strip("\n").removesuffix("Violations per idiom:")
    input_prompt += f"""
# OUTPUT FORMAT
    
I want you to generate your output under a section called "### Final Idiom Violations Found".

Structure you response for a given idiom XYZ as follows for cases with violations:

{SAMPLE_OUTPUT_FORMAT_VIOLATIONS}

and as follows for cases with violations:

{SAMPLE_OUTPUT_FORMAT_NO_VIOLATIONS}

# CHAIN OF THOUGHT REASONING FORMAT

Additionally I want you structure your chain-of-thought or thinking process as follows:
1. LIST CONSTRUCT: First you must analyze the idiom and list the code constructs that need to be analyzed.
2. FILE SCAN: Then based on the identified code constructs you must do a line by line top to bottom analysis of the code file to find violations of the idioms and suggest fixes wherever possible.

## LIST CONSTRUCT FORMAT

In this stage you will compile an exhaustive list of code constructs that should be closely analyzed as well as conditions that should be applied and fixes to be suggested (if provided in the list) for the given idiom. Please note that for the conditions to be applied you should only include conditions explicitly mentioned in the definition. However you can try and extrapolate the examples provided to broaden the search of code constructs while strictly enforcing the definitions provided. Remember to make the conditions exactly as specific as the definitions no more and no less.

Structure the LIST CONSTRUCT stage as follows:

Idiom XYZ: invalid-function-name

Code Constructs to Analyze:
- Function definitions, particularly the names used in these definitions.

Condition:
- The function name does not follow the snake_case naming convention.

Fix:
- Rename the function to follow snake_case by using lowercase letters with words separated by underscores for improved readability.

Example of Identified Code Construct:
```python
def myFunction():
    pass
```

Example Fix:
```python
def my_function():
    pass
```

Additional Notes:
- MixedCase naming for functions is only allowed in contexts where it’s already the prevailing style for backward compatibility.
- Specific function names can be excluded from this rule using the `lint.pep8-naming.ignore-names` or `lint.pep8-naming.extend-ignore-names` configuration options, allowing for flexibility in certain scenarios.

## FILE SCAN FORMAT

Structure the FILE SCAN stage as follows:

Line 6: Detected `for` loop relevant to idiom XYZ:
 - Conditions for idiom XYZ state that ...
 - Here ...
 - In conclusion B007 is not violated!

Line 21: Detected `exec` call which is relevant to idiom XYZ:
 - Conditions for idiom XYZ state that ...
 - Here ...
 - In conclusion XYZ is violated!

# FINAL INSTRUCTIONS

- Follow the reasoning format mentioned in '# CHAIN OF THOUGHT REASONING FORMAT' strictly and cover the lines in a sorted order from the lowest line number to the highest.
- Strictly follow the output format shown in '# OUTPUT FORMAT' and include a "### Final Idiom Violations Found" section after you finish your analysis. Remember that if you don't include this section your answer will be invalid even if the analysis is correct !!
- Do not repeat the whole code file in your analysis. 
- Make sure to conduct the LIST CONSTRUCT stage before the FILE SCAN stage and to analyze all lines potentially containing idiom violations based on the code constructs enumerated in the LIST CONSTRUCT stage.
- Don't literally mention every line, only mention lines that contain the constructs mentioned in the LIST CONSTRUCT stage.
    
Violations per idiom:"""

    return input_prompt

RATIONALIZATION_PROMPT_WITH_GT = """So my original answer was: 

{REASONER_INCORRECT_ANSWER}

Let me check if my answer is correct. Ok it seems like my answer was wrong and the correct answer is:

{LINTER_CORRECT_ANSWER}

Let me try a different approach to get the violations per idiom.

Violations per idiom:"""

def get_args():
    parser = argparse.ArgumentParser(description="Run inference with different settings.")
    parser.add_argument("--model_name", type=str, required=True, help="which model is to be queried")
    parser.add_argument("--train_file", type=str, help="path to the file containing training data",
                        default="data/ruff_meta_linting/train_v5.json")
    parser.add_argument("--write_path", type=str, required=True, help="name of the file where predictions should be written")
    parser.add_argument("--port", type=int, default=8002, help="Port where vLLM server is being served")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of parallel threads/workers to be used for querying vLLM.")

    return parser.parse_args()

args = get_args()
# vLLM server details
PORT = args.port
VLLM_SERVER_URL = f"http://0.0.0.0:{PORT}/v1/chat/completions"
MAX_RETRIES = 5
MAX_SEQ_LEN = 32768
MAX_NEW_TOKENS = 2048
NUM_WORKERS = args.num_workers
WRITE_EVERY_N = 1  # flush every N completed responses
PATIENCE_LIMIT = 5

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

def generate_response(index: int, rec: dict, model_name: str):
    user_prompt = rec['messages'][0]['content']
    gt_response = rec['messages'][1]['content']

    if len(user_prompt) > MAX_SEQ_LEN - MAX_NEW_TOKENS:
        user_prompt = user_prompt[:15000] + user_prompt[-15000:]

    messages = [{"role": "user", "content": user_prompt}]
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 0.7,
        "top_p": 0.95,
        "seed": 42
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(VLLM_SERVER_URL, json=payload)
            response.raise_for_status()
            model_response = response.json()["choices"][0]["message"]["content"]

            rationalized = False
            STaR_failed = False
            try: reward = evaluate_instance(model_response, gt_response)
            except AttributeError: reward = 0
            IMPATIENCE = 0
            
            while reward != 1 and IMPATIENCE < PATIENCE_LIMIT:
                IMPATIENCE += 1
                # print(f"failed attempt {IMPATIENCE}/{PATIENCE_LIMIT} for {index}")
                messages = [{"role": "user", "content": user_prompt}]
                payload = {
                    "model": model_name,
                    "messages": messages,
                    "max_tokens": MAX_NEW_TOKENS,
                    "temperature": [0.5, 0.7, 0.9][IMPATIENCE%3],
                    "top_p": 0.95,
                    "seed": 42
                }
                response = requests.post(VLLM_SERVER_URL, json=payload)
                response.raise_for_status()
                model_response = response.json()["choices"][0]["message"]["content"]
                try: reward = evaluate_instance(model_response, gt_response)
                except AttributeError: reward = 0

            if IMPATIENCE == PATIENCE_LIMIT:
                rationalized = True
                corrected_model_response = [model_response]
                print(f"\x1b[31;1mfailed to get correct answer for {index}! Moving on to rationalization\x1b[0m")
                REASONER_INCORRECT_ANSWER = load_linter_results(model_response)
                LINTER_CORRECT_ANSWER = gt_response
                
                messages = [
                    {"role": "user", "content": user_prompt},
                    {"role": "system", "content": model_response},
                    {"role": "system", "content": RATIONALIZATION_PROMPT_WITH_GT.format(
                        REASONER_INCORRECT_ANSWER=REASONER_INCORRECT_ANSWER,
                        LINTER_CORRECT_ANSWER=LINTER_CORRECT_ANSWER,
                    )},
                ]
                payload = {
                    "model": model_name,
                    "messages": messages,
                    "max_tokens": MAX_NEW_TOKENS,
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "seed": 42
                }
                response = requests.post(VLLM_SERVER_URL, json=payload)
                response.raise_for_status()
                model_response = response.json()["choices"][0]["message"]["content"]
                
                try: reward = evaluate_instance(model_response, gt_response)
                except AttributeError: reward = 0
                IMPATIENCE = 0

                while reward != 1 and IMPATIENCE < PATIENCE_LIMIT:
                    IMPATIENCE += 1
                    # print(f"failed attempt {IMPATIENCE}/{PATIENCE_LIMIT} for {index}")
                    REASONER_INCORRECT_ANSWER = load_linter_results(model_response)
                    LINTER_CORRECT_ANSWER = gt_response
                    messages = [
                        {"role": "user", "content": user_prompt},
                        {"role": "system", "content": model_response},
                        {"role": "system", "content": RATIONALIZATION_PROMPT_WITH_GT.format(
                            REASONER_INCORRECT_ANSWER=REASONER_INCORRECT_ANSWER,
                            LINTER_CORRECT_ANSWER=LINTER_CORRECT_ANSWER,
                        )},
                    ]
                    payload = {
                        "model": model_name,
                        "messages": messages,
                        "max_tokens": MAX_NEW_TOKENS,
                        "temperature": [0.5, 0.7, 0.9][IMPATIENCE%3],
                        "top_p": 0.95,
                        "seed": 42
                    }
                    response = requests.post(VLLM_SERVER_URL, json=payload)
                    response.raise_for_status()
                    model_response = response.json()["choices"][0]["message"]["content"]
                    try: reward = evaluate_instance(model_response, gt_response)
                    except AttributeError: reward = 0

                if reward == 1: 
                    corrected_model_response.append(model_response)
                    print(f"\x1b[32;1mRationalization worked for {index}!\x1b[0m")
                else: 
                    corrected_model_response.append(gt_response) # just add model response without CoT as it is.
                    STaR_failed = True
                    print(f"\x1b[31;1mRationalization failed for {index}!\x1b[0m")
            else: corrected_model_response = [model_response]

            return index, {
                "id": rec["id"],
                "source": rec["source"],
                "model_response": corrected_model_response,
                "ground_truth": gt_response,
                "error": False,
                "rationalized": rationalized,
                "STaR_failed": STaR_failed,
                "reward": reward,
            }
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise RuntimeError(f"Failed after {MAX_RETRIES} retries on index {index}: {e}")
            continue

def main(args):
    model_name = args.model_name
    write_path = args.write_path
    train_data = json.load(open(args.train_file))
    for i in range(len(train_data)):
        train_data[i]['messages'][0]['content'] = add_cot_gen_instr_to_prompt(train_data[i]['messages'][0]['content'])
    print(train_data[i]['messages'][0]['content'])
    skip_index_till = 0

    if not os.path.exists(write_path):
        open(write_path, "w").close()
    else:
        existing_preds = read_jsonl(write_path)
        skip_index_till = len(existing_preds)
        if skip_index_till > 0:
            print(f"Resuming from index {skip_index_till}")

    pending_data = train_data[skip_index_till:]
    results_buffer = {}
    next_write_index = skip_index_till

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor, open(write_path, "a") as f_out:
        futures = {
            executor.submit(generate_response, i + skip_index_till, rec, model_name): i + skip_index_till
            for i, rec in enumerate(pending_data)
        }

        for future in tqdm(as_completed(futures), total=len(futures)):
            index, result = future.result()
            results_buffer[index] = result

            # Check if we can write contiguous results starting from next_write_index
            written_count = 0
            while next_write_index in results_buffer:
                f_out.write(json.dumps(results_buffer[next_write_index]) + "\n")
                del results_buffer[next_write_index]
                next_write_index += 1
                written_count += 1

            # Optional: flush every N writes
            if written_count >= WRITE_EVERY_N:
                f_out.flush()

        # Final flush for any remaining buffered results
        while next_write_index in results_buffer:
            f_out.write(json.dumps(results_buffer[next_write_index]) + "\n")
            next_write_index += 1
        f_out.flush()

if __name__ == "__main__":
    main(args)

    # python src/reasoning/star_self_data_generation.py --model_name Qwen/Qwen3-4B --write_path "data/ruff_meta_linting/cot_gen/qwen3-4b-transfer-cot-gen-cache-star.jsonl" --train_file "data/ruff_meta_linting/train_v5.json"