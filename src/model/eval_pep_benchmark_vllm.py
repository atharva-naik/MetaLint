import os
import sys
import json
import asyncio
import pathlib
import argparse
import requests
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score
from transformers import AutoModelForCausalLM, AutoTokenizer

module_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent)
sys.path.append(module_path)

from src.datautils import read_jsonl, META_LINTING_PROMPT_V2, META_LINTING_PROMPT_ZERO_SHOT, SAMPLE_OUTPUT
from src.metrics.meta_linting.idiom_detection_and_localization_v3 import load_linter_results

# vLLM server details
PORT = 8002
VLLM_SERVER_URL = f"http://0.0.0.0:{PORT}/v1/chat/completions"
MAX_RETRIES = 5
MAX_SEQ_LEN = 32768
MAX_NEW_TOKENS = 2048

def compute_idiom_wise_pr(data, pep_list):
    idiom_binary_presence_pred = {idiom_code: [0 for _ in range(len(data))] for idiom_code in pep_list}
    idiom_binary_presence_gt = {idiom_code: [0 for _ in range(len(data))] for idiom_code in pep_list}
    idiom_precisions, idiom_recalls = {}, {}
    # tool_group_freq = defaultdict(lambda: 0)
    # tool_group_freq["all"] = 0
    for index,rec in enumerate(data):
        model_resp = rec["model_response"]
        gt = rec["ground_truth"]
        for violation in model_resp:
            if "code" not in violation: continue
            idiom_code = violation["code"]
            if idiom_code in pep_list:
                idiom_binary_presence_pred[idiom_code][index] = 1

        for violation in gt:
            if "code" not in violation: continue
            idiom_code = violation["code"]
            idiom_binary_presence_gt[idiom_code][index] = 1

    for idiom_code in pep_list:
        idiom_precisions[idiom_code] = precision_score(idiom_binary_presence_gt[idiom_code], idiom_binary_presence_pred[idiom_code], zero_division=0)
        idiom_recalls[idiom_code] = recall_score(idiom_binary_presence_gt[idiom_code], idiom_binary_presence_pred[idiom_code], zero_division=0) 

    return idiom_precisions, idiom_recalls

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
    parser.add_argument("--online", action="store_true", help="start in interactive mode")
    parser.add_argument("--offline", action="store_true", required=True, help="evaluate offline on a benchmark")
    parser.add_argument("--model_name", type=str, required=True, help="which model is to be queried")
    parser.add_argument("--save_path", type=str, required=True, help="path to save predictions")
    parser.add_argument("--zero_shot", action="store_true", help="use the zero shot prompt")
    parser.add_argument('--peps', nargs='+', help='list of peps', type=str)
    # parser.add_argument("--step", type=int, default=2000, help="Number of training steps the model has undergone (default: 2000).")
    
    return parser.parse_args()

def add_linenos_to_line_field(line_field, start_lineno: int, end_lineno: int):
    split_line_field = line_field.split("\n")
    i = 0
    for lineno in range(start_lineno, end_lineno+1):
        split_line_field[i] = f"{str(lineno).rjust(3)} {split_line_field[i]}"
        i += 1

    return "\n".join(split_line_field)

def load_pep_benchmark(annotations_folder: str, files_folder: str) -> list[tuple[str, dict]]:
    """load benchmark in a format similar to ruff training data
    
    E.g. [{'line': "  7     password = ''.join(random.sample(characters, length))", 'span': 'random', 'fix': None, 'code': '506'}]
    """
    # annotations_folder = os.path.join(folder, "dummy_annotations")
    # files_folder = os.path.join(folder, "dummy_files")
    data = []
    for filename in os.listdir(annotations_folder):
        blob_id,_ = os.path.splitext(filename)
        code_file = os.path.join(files_folder, f"{blob_id}.py")
        annotation_file = os.path.join(annotations_folder, f"{blob_id}.json")
        code_file = open(code_file).read()
        annotations = []#json.load(open(annotation_file))
        # annotations = [
        # {
        #     "line": add_linenos_to_line_field(
        #         v["line"], v["start_lineno"], v["end_lineno"]
        #     ), 
        #     "span": v["span"], 
        #     "fix": v["fix"],
        #     "code": v["pep"],
        # } for v in annotations["violations"]]
        for v in json.load(open(annotation_file))["violations"]:
            try:
                annotations.append({
                    "line": add_linenos_to_line_field(
                        v["line"], v["start_lineno"], v["end_lineno"]
                    ), 
                    "span": v["span"], 
                    "fix": v["fix"],
                    "code": v["pep"],
                })
            except IndexError as e:
                print(e)
                print(v)
                print(blob_id)
                exit()
        data.append((blob_id, code_file, annotations))

    return data

async def generate_response(user_prompt: str, model_name: str):
    """ Sends a request to vLLM for generating a response """
    messages = [
        {"role": "user", "content": user_prompt}
    ]
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": MAX_NEW_TOKENS,
        # "temperature": 1,   # enables sampling
        # "top_p": 0.95,         # nucleus sampling
        # "seed": 42          # ensures reproducibility
        "temperature": 0.7, # greedy decoding for reproducibility
        "top_p": 0.95,
        "seed": 42  # seed for reproducibility
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
            user_prompt = prompt_template.format(LIST_OF_IDIOM_SPECS=LIST_OF_IDIOM_SPECS, CODE_FILE=CODE_FILE_WITH_LINENO, SAMPLE_OUTPUT=SAMPLE_OUTPUT)
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

async def eval_benchmark(benchmark, model_name: str, peps: list[str], pep_idiom_specs: dict, save_path: str, zero_shot: bool=False):
    """ Offline query the model to detect PEP violations for a files in a benchmark path passed by the user. """
    LIST_OF_IDIOM_SPECS = "\n\n".join([idiom_spec_extractor_for_pep(pep_idiom_specs[idiom_code]) for idiom_code in peps])
    resp_and_gt = []
    with open(save_path, "w") as f: pass
    for blob_id, code_file, gt in tqdm(benchmark):
        
        CODE_FILE_WITH_LINENO = add_lineno_to_code(code_file)            
        prompt_template = META_LINTING_PROMPT_ZERO_SHOT if zero_shot else META_LINTING_PROMPT_V2
        user_prompt = prompt_template.format(LIST_OF_IDIOM_SPECS=LIST_OF_IDIOM_SPECS, CODE_FILE=CODE_FILE_WITH_LINENO, SAMPLE_OUTPUT=SAMPLE_OUTPUT)
        
        # print(user_prompt)
        response = await generate_response(user_prompt=user_prompt, model_name=model_name)
        try: cot, report = parse_response(response["model_response"])
        except ValueError:
            cot = None
            report = response["model_response"]
        report = report.replace("```json\n{", "{").replace("}\n```", "}")
        # print(report)
        predictions = load_linter_results(report)
        resp_and_gt.append({
            "model_response": predictions,
            "ground_truth": gt,
        })
        # print(f"predicted {blob_id}: {predictions}")
        with open(save_path, "a") as f:
            f.write(json.dumps({
                "blob_id": blob_id, 
                "code_file": code_file, 
                "model_response": response["model_response"],
                "chain_of_thought": cot,
                "report": report,
                "predictions": predictions, 
                "ground_truth": gt
            })+"\n")
        # print("GT:", gt)
    print(compute_idiom_wise_pr(resp_and_gt, pep_list=peps))

if __name__ == "__main__":
    args = get_args()

    model_name = args.model_name
    pep_list = args.peps
    zero_shot = args.zero_shot
    pep_idiom_specs = {str(rec["code"]).strip(): rec for rec in pd.read_csv("./data/pep_idiom_specs/hard_pep_idiom_specs.csv").to_dict('records')}
    
    if args.online:
        asyncio.run(eval_repl(model_name=model_name, peps=pep_list, pep_idiom_specs=pep_idiom_specs, zero_shot=args.zero_shot))
    if args.offline:
        pep_benchmark = load_pep_benchmark(
            annotations_folder="./data/pep_benchmark/annotations/",
            files_folder="./data/pep_benchmark/code_files/"
            # annotations_folder="./data/pep_benchmark/dummy_annotations/",
            # files_folder="./data/pep_benchmark/dummy_files/"
        )
        asyncio.run(eval_benchmark(
            benchmark=pep_benchmark, 
            model_name=model_name, peps=pep_list, pep_idiom_specs=pep_idiom_specs, 
            zero_shot=args.zero_shot, 
            save_path=args.save_path,
        ))
    # python src/model/interactive_eval_vllm.py --online --model_name "alignment-handbook/model_checkpoints/qwen2.5coder-3b-instruct-sft-trasfer-v4-subtask-cot-v2-lite/checkpoint-500/" --peps 506 557 634 655

    # python src/model/interactive_eval_vllm.py --offline --model_name "alignment-handbook/model_checkpoints/qwen2.5coder-3b-instruct-sft-trasfer-v4-subtask-cot-v2-lite/checkpoint-500/" --peps 506 557 634 655

    # python src/model/interactive_eval_vllm.py --offline --model_name "Qwen/Qwen2.5-Coder-3B-Instruct/" --peps 506 557 634 655