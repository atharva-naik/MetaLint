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

# vLLM server details
PORT = 8002
VLLM_SERVER_URL = f"http://0.0.0.0:{PORT}/v1/chat/completions"
MAX_RETRIES = 5
MAX_SEQ_LEN = 32768
MAX_NEW_TOKENS = 2048
NUM_WORKERS = 12
WRITE_EVERY_N = 20  # flush every N completed responses

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

    return parser.parse_args()

def generate_response(index: int, rec: dict, model_name: str, no_think: bool):
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

    if no_think:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(VLLM_SERVER_URL, json=payload)
            response.raise_for_status()
            model_response = response.json()["choices"][0]["message"]["content"]
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

def main():
    args = get_args()
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
            executor.submit(generate_response, i + skip_index_till, rec, model_name, args.no_think): i + skip_index_till
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
    main()
    # python src/model/eval_llm_meta_linter_vllm_futures.py --model_name alignment-handbook/model_checkpoints/qwen2.5coder-3b-instruct-sft-all-idioms-subtask-cot-star/checkpoint-1500/ --write_path data/meta_linting_preds_vllm/qwen2.5coder_3b_instruct_sft_preds_1500_all_idioms_subtask_cot_star.jsonl  --test_file data/ruff_meta_linting/all_idioms/test.json                            

    # python src/model/eval_llm_meta_linter_vllm_futures.py --model_name alignment-handbook/model_checkpoints/qwen2.5coder-3b-instruct-sft-all-idioms-subtask-cot-star/checkpoint-2000/ --write_path data/meta_linting_preds_vllm/qwen2.5coder_3b_instruct_sft_preds_2000_all_idioms_subtask_cot_star.jsonl  --test_file data/ruff_meta_linting/all_idioms/test.json

    # python src/model/eval_llm_meta_linter_vllm_futures.py --model_name "alignment-handbook/model_checkpoints/qwen2.5coder-3b-instruct-dpo-transfer-v4-subtask-cot-star/checkpoint-200/" --write_path data/meta_linting_preds_vllm/qwen2.5coder_3b_instruct_dpo_preds_200_transfer_v4_subtask_cot_star.jsonl --test_file data/ruff_meta_linting/test_v4_new_format_with_lineno.json   

    # python src/model/eval_llm_meta_linter_vllm_futures.py --model_name alignment-handbook/model_checkpoints/qwen3-4b-instruct-sft-trasfer-v5-lineno/checkpoint-6000/ --write_path "data/meta_linting_preds_vllm/qwen3_4b_sft_preds_6000_transfer_v5_lineno.jsonl" --test_file "data/ruff_meta_linting/test_v5.json"

    # python src/model/eval_llm_meta_linter_vllm_futures.py --model_name Qwen/Qwen3-4B --write_path "data/meta_linting_preds_vllm/qwen3_4b_untrained_preds_transfer_v5_lineno.jsonl" --test_file "data/ruff_meta_linting/test_v5.json" --untrained_mode

    # python src/model/eval_llm_meta_linter_vllm_futures.py --model_name Qwen/Qwen3-4B --write_path "data/meta_linting_preds_vllm/qwen3_4b_untrained_no_think_preds_transfer_v5_lineno.jsonl" --test_file "data/ruff_meta_linting/test_v5.json" --untrained_mode --no_think