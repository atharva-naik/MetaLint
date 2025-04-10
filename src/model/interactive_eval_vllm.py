import os
import sys
import json
import asyncio
import pathlib
import argparse
import requests
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

module_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent)
sys.path.append(module_path)

from src.datautils import read_jsonl, META_LINTING_PROMPT_V2, META_LINTING_PROMPT_ZERO_SHOT

# vLLM server details
VLLM_SERVER_URL = "http://0.0.0.0:8001/v1/chat/completions"
MAX_RETRIES = 5
MAX_SEQ_LEN = 32768
MAX_NEW_TOKENS = 2048

def idiom_spec_extractor_for_pep(idiom_spec):
    if "example" in idiom_spec:
        Example = f"\n\nExample:\n{idiom_spec['example']}"
    else: Example = ""
    return f"# Idiom {idiom_spec['code']} ({idiom_spec['name']})\n\nDefinition: {idiom_spec['what-it-does']}\n\nRationale: {idiom_spec['why-is-this-bad']}"+Example

def add_lineno_to_code(code: str):
    code_with_lineno = []
    for lineno, line in enumerate(code.split("\n")):
        code_with_lineno.append(f"{str(lineno+1).rjust(3)} {line}")

    return "\n".join(code_with_lineno)

def parse_response(response: str):
    cot, report = response.split("### Final Idiom Violations Found")
    return cot.strip("\n"), report.strip("\n")

def get_args():
    parser = argparse.ArgumentParser(description="Run inference with different settings.")
    # parser.add_argument("--cot", action="store_true", help="Run inference with Chain-of-Thought (CoT) prompting.")
    # parser.add_argument("--subtask_cot", action="store_true", help="Run inference with Chain-of-Thought (CoT) prompting.")
    parser.add_argument("--model_name", type=str, required=True, help="which model is to be queried")
    parser.add_argument("--zero_shot", action="store_true", help="use the zero shot prompt")
    parser.add_argument('--peps', nargs='+', help='list of peps', type=str)
    # parser.add_argument("--step", type=int, default=2000, help="Number of training steps the model has undergone (default: 2000).")
    
    return parser.parse_args()

async def generate_response(user_prompt: str, model_name: str):
    """ Sends a request to vLLM for generating a response """
    messages = [
        {"role": "user", "content": user_prompt}
    ]
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": MAX_NEW_TOKENS,
    }

    for _ in range(MAX_RETRIES):
        try:
            response = requests.post(VLLM_SERVER_URL, json=payload).json()
            model_response = response["choices"][0]["message"]["content"]
            return {
                "model_response": model_response,
                "error": False,
            }
        except Exception as e: pass
            # print(response)
            # print(e)

    return {
        "model_response": f"Got Error: {str(e)}",
        "error": True,
    }
    
async def eval_repl(model_name: str, peps: list[str], pep_idiom_specs: dict, zero_shot: bool=False):
    """ Interactively query the model to detect PEP violations for a file passed by the user. """
    LIST_OF_IDIOM_SPECS = "\n\n".join([idiom_spec_extractor_for_pep(pep_idiom_specs[idiom_code]) for idiom_code in peps])
    while True:
        filepath = input("Enter path to code file to be analyzed: ").strip()
        if filepath == "exit": exit()
        elif os.path.exists(filepath):
            CODE_FILE_WITH_LINENO = add_lineno_to_code(open(filepath).read())
            
            prompt_template = META_LINTING_PROMPT_ZERO_SHOT if zero_shot else META_LINTING_PROMPT_V2
            user_prompt = prompt_template.format(LIST_OF_IDIOM_SPECS=LIST_OF_IDIOM_SPECS, CODE_FILE=CODE_FILE_WITH_LINENO)
            print(user_prompt)
            
            response = await generate_response(user_prompt=user_prompt, model_name=model_name)
            try: cot, report = parse_response(response["model_response"])
            except ValueError:
                cot = ""
                report = response["model_response"]

            print(f"\x1b[34;1mChain of Thought Analysis:\x1b[0m\n")
            print(cot)

            print(f"\x1b[34;1mPEP Violation Report:\x1b[0m\n")
            print(report)


        else: continue

if __name__ == "__main__":
    args = get_args()

    model_name = args.model_name
    pep_list = args.peps
    zero_shot = args.zero_shot
    pep_idiom_specs = {str(rec["code"]).strip(): rec for rec in pd.read_csv("./data/pep_idiom_specs/hard_pep_idiom_specs.csv").to_dict('records')}

    asyncio.run(eval_repl(model_name=model_name, peps=pep_list, pep_idiom_specs=pep_idiom_specs, zero_shot=args.zero_shot))
    # python src/model/interactive_eval_vllm.py --model_name "alignment-handbook/model_checkpoints/qwen2.5coder-3b-instruct-sft-trasfer-v4-subtask-cot-v2-lite/checkpoint-500/" --peps 506 557 634 655