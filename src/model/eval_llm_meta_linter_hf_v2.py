import os
import sys
import json
import torch
import pathlib
import argparse
import requests
from tqdm import tqdm
from transformers import pipeline

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
MAX_SEQ_LEN = 32768
MAX_NEW_TOKENS = 8192
NUM_WORKERS = args.num_workers
WRITE_EVERY_N = 1  # flush every N completed responses


def generate_response(pipe, rec: dict, model_name: str):
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
    outputs = pipe(
        messages,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
        seed=42,
    )
    model_response = outputs[0]["generated_text"][-1]

    return {
        "id": rec["id"],
        "source": rec["source"],
        "model_response": model_response,
        "ground_truth": gt_response,
        "error": False,
    }

def main(args):
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

    pipe = pipeline(
        "text-generation",
        model=model_name,
        torch_dtype="auto",
        device_map="auto",
    )

    with open(write_path, "a") as f_out:
        for i, rec in tqdm(enumerate(pending_data), total=len(pending_data)):
            result = generate_response(model, rec, model_name)
            f_out.write(json.dumps(result) + "\n")

if __name__ == "__main__":
    main(args)

    # python src/model/eval_llm_meta_linter_hf_v2.py --model_name openai/gpt-oss-120b --write_path "data/pep_benchmark_preds_v2/gpt_oss_120b_untrained.jsonl" --test_file "data/pep_benchmark/test_pep_v2.json" --untrained_mode