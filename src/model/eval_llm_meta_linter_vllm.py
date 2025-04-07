import os
import sys
import json
import asyncio
import pathlib
import argparse
import requests
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

module_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent)
sys.path.append(module_path)

from src.datautils import read_jsonl

# vLLM server details
VLLM_SERVER_URL = "http://localhost:8001/v1/chat/completions"
MAX_RETRIES = 5

def get_args():
    parser = argparse.ArgumentParser(description="Run inference with different settings.")
    parser.add_argument("--cot", action="store_true", help="Run inference with Chain-of-Thought (CoT) prompting.")
    parser.add_argument("--subtask_cot", action="store_true", help="Run inference with Chain-of-Thought (CoT) prompting.")
    parser.add_argument("--lineno", action="store_true", help="Include line numbers in the code prompt during inference.")
    parser.add_argument("--step", type=int, default=2000, help="Number of training steps the model has undergone (default: 2000).")
    
    return parser.parse_args()

async def generate_response(rec: dict, model_name: str):
    """ Sends a request to vLLM for generating a response """
    user_prompt = rec['messages'][0]['content']
    gt_response = rec['messages'][1]['content']
    # truncate extremely long user prompts.
    if len(user_prompt) > 800000:
        user_prompt = rec['messages'][0]['content'][:100000]+rec['messages'][0]['content'][-100000:]
    messages = [
        {"role": "user", "content": user_prompt}
    ]
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": 2048,
    }

    for _ in range(MAX_RETRIES):
        try:
            response = requests.post(VLLM_SERVER_URL, json=payload).json()
            model_response = response["choices"][0]["message"]["content"]

            return {
                "id": rec["id"],
                "source": rec["source"],
                "model_response": model_response,
                "ground_truth": gt_response,
                "error": False,
            }
            break
        except Exception as e:
            print(e)
            exit()
            return {
                "id": rec["id"],
                "source": rec["source"],
                "model_response": f"Got Error: {str(e)}",
                "ground_truth": gt_response,
                "error": True,
            }
    
async def process_records(pbar, model_name: str, skip_index_till: int=-1):
    """ Processes dataset records asynchronously and writes responses to a file """
    # for i, idx in enumerate(random_indices):
    #     result = await generate_response(idx)
    #     results.append(result)
        
    #     print(f"Generated {i+1}/{total} responses")
        
    #     # Save results periodically (every 10 responses)
    #     if (i+1) % 10 == 0 or i == total-1:
    #         with open(output_file, 'r') as f:
    #             data = json.load(f)
            
    #         data["data"] = results if "data" not in data else data["data"] + results
            
    #         with open(output_file, 'w') as f:
    #             json.dump(data, f, indent=4)
            
    #         results = []
            
    # print(f"Completed processing all {total} records.")
    for index,rec in pbar:
        if index < skip_index_till: continue
        elif index == skip_index_till:
            print(f"resuming inference from index: {index}")
        result = await generate_response(rec, model_name=model_name)
        # text = tokenizer.apply_chat_template(
        #     messages,
        #     tokenize=False,
        #     add_generation_prompt=True
        # )
        # model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # generated_ids = model.generate(
        #     **model_inputs,
        #     max_new_tokens=2048,
        # )
        # generated_ids = [
        #     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        # ]
        # response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        model_preds.append(result)
        with open(write_path, "a") as f:
            f.write(json.dumps(result)+"\n")

if __name__ == "__main__":
    args = get_args()
    train_steps = args.step
    cot = args.cot
    lineno = args.lineno
    subtask_cot = args.subtask_cot
    # model_name = f"alignment-handbook/model_checkpoints/qwen2.5coder-3b-instruct-sft-v2-data/checkpoint-{train_steps}"
    if cot == True: 
        write_path = f"./data/meta_linting_preds_vllm/qwen2.5coder_3b_instruct_sft_preds_{train_steps}_transfer_v4_cot.jsonl"
        model_name = f"alignment-handbook/model_checkpoints/qwen2.5coder-3b-instruct-sft-trasfer-v4-cot/checkpoint-{train_steps}"
    elif subtask_cot:
        write_path = f"./data/meta_linting_preds_vllm/qwen2.5coder_3b_instruct_sft_preds_{train_steps}_transfer_v4_subtask_cot.jsonl"
        model_name = f"alignment-handbook/model_checkpoints/qwen2.5coder-3b-instruct-sft-trasfer-v4-subtask-cot/checkpoint-{train_steps}"
    elif lineno:
        write_path = f"./data/meta_linting_preds_vllm/qwen2.5coder_3b_instruct_sft_preds_{train_steps}_transfer_v4_lineno.jsonl"
        model_name = f"alignment-handbook/model_checkpoints/qwen2.5coder-3b-instruct-sft-trasfer-v4-lineno/checkpoint-{train_steps}"        
    else: 
        write_path = f"./data/meta_linting_preds_vllm/qwen2.5coder_3b_instruct_sft_preds_{train_steps}_transfer_v4.jsonl"
        model_name = f"alignment-handbook/model_checkpoints/qwen2.5coder-3b-instruct-sft-trasfer-v4/checkpoint-{train_steps}"
    #"Qwen/Qwen2.5-7B-Instruct-1M"
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name, torch_dtype="auto",
    #     device_map="auto"
    # )
    # tokenizer = AutoTokenizer.from_pretrained(model_name)

    # test_data = json.load(open("data/ruff_meta_linting/test_v3.json"))
    if lineno: test_data = json.load(open("data/ruff_meta_linting/test_v4_new_format_with_lineno.json"))
    else: test_data = json.load(open("data/ruff_meta_linting/test_v4.json"))
    # test_data = json.load(open("data/ruff_meta_linting/hardness_experiment/test.json"))
    model_preds = []
    
    skip_index_till: int = -1
    if not os.path.exists(write_path):
        f = open(write_path, "w")
        # print('f = open(write_path, "w")')
        # exit()
    else: 
        preds = read_jsonl(write_path)
        skip_index_till = len(preds)
        # exit()

    pbar = tqdm(enumerate(test_data), total=len(test_data))
    asyncio.run(process_records(pbar, model_name=model_name, skip_index_till=skip_index_till))