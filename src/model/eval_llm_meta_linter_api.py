import os
import sys
import json
import openai
import pathlib
import argparse
import requests
from tqdm import tqdm
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

module_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent)
sys.path.append(module_path)

from src.datautils import read_jsonl

SAMPLE_OUTPUT_FORMAT_VIOLATIONS = """### Final Idiom Violations Found

**Idiom XYZ Violations:**

{"line": " 12 \\t\\t#event = forms.ModelChoiceField(queryset=Inquiry.objects.filter(owner=kwargs.pop('user')))", "fix": null}
{"line": "  1 from django import forms\\n  2 from django.forms.models import inlineformset_factory\\n  3 from .models import Request\\n  4 from inquiry.models import *", "fix": [{"before": "from django import forms\\nfrom django.forms.models import inlineformset_factory\\nfrom .models import Request\\nfrom inquiry.models import *\\n\\n\\n\\n", "after": "from django import forms\\nfrom django.forms.models import inlineformset_factory\\nfrom inquiry.models import *\\n\\nfrom .models import Request\\n\\n\\n"}]}
"""

SAMPLE_OUTPUT_FORMAT_NO_VIOLATIONS = """### Final Idiom Violations Found

**Idiom XYZ Violations:**

NO VIOLATIONS FOUND
"""

def add_output_instr_to_prompt(input_prompt: str):
    assert input_prompt.strip("\n").endswith("Violations per idiom:")
    input_prompt = input_prompt.strip("\n").removesuffix("Violations per idiom:")
    input_prompt += f"""
# OUTPUT FORMAT
    
I want you to generate your output under a section called "### Final Idiom Violations Found".

Structure you response for a given idiom XYZ as follows for cases with violations:

{SAMPLE_OUTPUT_FORMAT_VIOLATIONS}

and as follows for cases with violations:

{SAMPLE_OUTPUT_FORMAT_NO_VIOLATIONS}

Violations per idiom:"""
    
    return input_prompt

def get_args():
    parser = argparse.ArgumentParser(description="Run inference with different settings.")
    parser.add_argument("--lineno", action="store_true", help="Include line numbers in the code prompt during inference.")
    parser.add_argument("--model_name", type=str, required=True, help="which model is to be queried")
    parser.add_argument("--test_file", type=str, help="path to the file containing test data",
                        default="data/ruff_meta_linting/test_v4_new_format_with_lineno.json")
    parser.add_argument("--write_path", type=str, required=True, help="name of the file where predictions should be written")
    parser.add_argument("--untrained_mode", action="store_true", help="used when inferencing untrained model to obtain proper output format.")
    parser.add_argument("--no_think", action="store_true", help="Disable chain-of-thought reasoning globally.")
    parser.add_argument("--port", type=int, default=8002, help="Port where vLLM server is being served")
    parser.add_argument("--num_workers", type=int, default=12, help="Number of parallel threads/workers to be used for querying vLLM.")

    return parser.parse_args()

args = get_args()
MAX_RETRIES = 5
MAX_SEQ_LEN = 32768
MAX_NEW_TOKENS = 8192
NUM_WORKERS = args.num_workers
WRITE_EVERY_N = 1  # flush every N completed responses

def generate_response(client, index: int, rec: dict, model_name: str, no_think: bool):
    user_prompt = rec['messages'][0]['content']
    gt_response = rec['messages'][1]['content']

    if len(user_prompt) > MAX_SEQ_LEN - MAX_NEW_TOKENS:
        user_prompt = user_prompt[:15000] + user_prompt[-15000:]

    messages = [{"role": "user", "content": user_prompt}]
    # payload = {
    #     "model": model_name,
    #     "messages": messages,
    #     "max_tokens": MAX_NEW_TOKENS,
    #     "temperature": 0.7,
    #     "top_p": 0.95,
    #     "seed": 42
    # }
    # if no_think:
    #     payload["chat_template_kwargs"] = {"enable_thinking": False}

    for attempt in range(MAX_RETRIES):
        try:
            if model_name in ["o3-mini", "o4-mini"]:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_completion_tokens=3000
                )
            elif model_name in ["gpt-5", "gpt-5-2025-08-07"]:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    reasoning_effort="high",
                    max_completion_tokens=8192,
                )
            else:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=1024
                )
            model_response = response.choices[0].message.content

            return index, {
                "id": rec["id"],
                "source": rec["source"],
                "model_response": model_response,
                "ground_truth": gt_response,
                "error": False,
            }
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise RuntimeError(f"Failed after {MAX_RETRIES} retries on index {index}: {e}")
            continue

def main(args):
    openai.api_key = os.environ["ORACLE_PROJECT_COT_API_KEY"]
    client = openai.OpenAI(api_key=os.environ["ORACLE_PROJECT_COT_API_KEY"])

    model_name = args.model_name
    write_path = args.write_path

    test_data = json.load(open(args.test_file))
    if args.untrained_mode:
        for i in range(len(test_data)):
            test_data[i]['messages'][0]['content'] = add_output_instr_to_prompt(test_data[i]['messages'][0]['content'])
    skip_index_till = 0

    if not os.path.exists(write_path):
        open(write_path, "w").close()
    else:
        existing_preds = read_jsonl(write_path)
        skip_index_till = len(existing_preds)
        if skip_index_till > 0:
            print(f"Resuming from index {skip_index_till}")

    pending_data = test_data[skip_index_till:]
    results_buffer = {}
    next_write_index = skip_index_till

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor, open(write_path, "a") as f_out:
        futures = {
            executor.submit(generate_response, client, i + skip_index_till, rec, model_name, args.no_think): i + skip_index_till
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

    # PEP benchmark evals

    # python src/model/eval_llm_meta_linter_api.py --model_name gpt-5 --write_path "data/pep_benchmark_preds_v2/gpt_5_untrained_preds.jsonl" --test_file "data/pep_benchmark/test_pep_v2.json" --untrained_mode