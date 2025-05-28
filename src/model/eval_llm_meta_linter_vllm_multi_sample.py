import os
import sys
import json
import pathlib
import argparse
import requests
from tqdm import tqdm
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed

module_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent)
sys.path.append(module_path)

from src.datautils import read_jsonl
from src.dpo.reward_model_newformat import evaluate_instance
from src.metrics.meta_linting.idiom_detection_and_localization_v3 import load_linter_results

# vLLM server details
PORT = 8002
VLLM_SERVER_URL = f"http://0.0.0.0:{PORT}/v1/chat/completions"
MAX_RETRIES = 5
MAX_SEQ_LEN = 32768
MAX_NEW_TOKENS = 2048
NUM_WORKERS = 16
WRITE_EVERY_N = 2

def get_args():
    parser = argparse.ArgumentParser(description="Run inference with different settings.")
    parser.add_argument("--model_name", type=str, required=True, help="which SFT model is to be queried")
    parser.add_argument("--write_path", type=str, required=True, help="name of the file where predictions should be written")
    parser.add_argument("--M", type=int, default=5, help="Number of samples per prompt")
    parser.add_argument("--train_file", type=str, required=True, help='path to training data to be used for training')
    parser.add_argument("--skip_no_violations", action="store_true", help="Comma-separated temperatures, one for each sample")
    parser.add_argument("--skip_index_offset", default=0, help="add an offset to the start index", type=int)
    parser.add_argument("--temp", type=str, default='0,0.3,0.5,0.7,1', help="Comma-separated temperatures, one for each sample")
    return parser.parse_args()

def generate_response(prompt_index: int, sample_index: int, rec: dict, model_name: str, temperature: float):
    user_prompt = rec['messages'][0]['content']
    gt_response = rec['messages'][1]['content']
    # print(len(user_prompt))
    if len(user_prompt) > MAX_SEQ_LEN - MAX_NEW_TOKENS:
        user_prompt = user_prompt[:10000] + user_prompt[-10000:]
        print(len(user_prompt))

    messages = [{"role": "user", "content": user_prompt}]
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": temperature,
        "top_p": 0.95,
        "seed": 42 + sample_index  # Different seed per sample
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(VLLM_SERVER_URL, json=payload, timeout=30)
            response.raise_for_status()
            model_response = response.json()["choices"][0]["message"]["content"]
            return (prompt_index, sample_index), {
                "id": rec["id"],
                "source": rec["source"],
                # "prompt": rec["user_prompt"],
                "sample_index": sample_index,
                "model_response": model_response,
                "ground_truth": gt_response,
                "error": False,
            }
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise RuntimeError(f"Failed after {MAX_RETRIES} retries on prompt {prompt_index} sample {sample_index}: {e}")
            continue


def main():
    args = get_args()
    model_name = args.model_name
    write_path = args.write_path
    M = args.M
    temperatures = list(map(float, args.temp.split(",")))

    assert len(temperatures) == M, "Length of temperatures must match M"

    if os.path.exists(args.train_file): # use local train file.
        train_data = json.load(open(args.train_file))
    else: train_data = load_dataset(args.train_file)['train'] # load train split from datasets module.
    print(f"read {len(train_data)} SFT training instances")
    if args.skip_no_violations: # skip data with NO VIOLATIONs for idioms (hack to improve recall).
        train_data = [rec for rec in train_data if len(load_linter_results(rec["messages"][1]["content"])) != 0]
        print(f"{len(train_data)} SFT training instances with VIOLATIONS")
    skip_index_till = 0

    if not os.path.exists(write_path):
        open(write_path, "w").close()
    else:
        # existing_preds = read_jsonl(write_path)
        # # automatic skip index + manual skip index offset added.
        # skip_index_till = int(args.skip_index_offset)+len(existing_preds)
        # print(f"Resuming from prompt index {skip_index_till}")
        existing_preds = read_jsonl(write_path)
        # Only count fully written prompts (those with M model_responses)
        num_fully_written = sum(1 for rec in existing_preds if "model_responses")
        skip_index_till = int(args.skip_index_offset) + num_fully_written
        print(f"Resuming from prompt index {skip_index_till}")

    pending_data = train_data[skip_index_till:]
    results_buffer = {}
    next_write_index = skip_index_till

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor, open(write_path, "a") as f_out:
        futures = {
            executor.submit(generate_response, i + skip_index_till, sample_idx, rec, model_name, temperatures[sample_idx]): (i + skip_index_till, sample_idx)
            for i, rec in enumerate(pending_data)
            for sample_idx in range(M)
        }

        for future in tqdm(as_completed(futures), total=len(futures)):
            (prompt_idx, sample_idx), result = future.result()

            if prompt_idx not in results_buffer:
                results_buffer[prompt_idx] = [None] * M
            results_buffer[prompt_idx][sample_idx] = result

            # Check if we can write contiguous prompts starting from next_write_index
            written_count = 0
            while next_write_index in results_buffer and all(results_buffer[next_write_index]):
                f_out.write(json.dumps({
                    "id": results_buffer[next_write_index][0]["id"],
                    "source": results_buffer[next_write_index][0]["source"],
                    # "prompt": results_buffer[next_write_index][0]["prompt"],
                    "ground_truth": results_buffer[next_write_index][0]["ground_truth"],
                    "model_responses": [(
                        sample["model_response"], 
                        evaluate_instance(
                            sample["model_response"], 
                            results_buffer[next_write_index][0]["ground_truth"]
                        )
                    ) for sample in results_buffer[next_write_index]]
                }) + "\n")
                del results_buffer[next_write_index]
                next_write_index += 1
                written_count += 1

            if written_count >= WRITE_EVERY_N:
                f_out.flush()

        # Final flush
        while next_write_index in results_buffer and all(results_buffer[next_write_index]):
            f_out.write(json.dumps({
                "id": results_buffer[next_write_index][0]["id"],
                "source": results_buffer[next_write_index][0]["source"],
                # "prompt": results_buffer[next_write_index][0]["prompt"],
                "ground_truth": results_buffer[next_write_index][0]["ground_truth"],
                "model_responses": [(
                        sample["model_response"], 
                        evaluate_instance(
                            sample["model_response"], 
                            results_buffer[next_write_index][0]["ground_truth"]
                        )
                    ) for sample in results_buffer[next_write_index]]
            }) + "\n")
            next_write_index += 1
        f_out.flush()

if __name__ == "__main__":
    main()

    # python src/model/eval_llm_meta_linter_vllm_multi_sample.py --model_name "alignment-handbook/model_checkpoints/qwen2.5coder-3b-instruct-sft-trasfer-v4-subtask-cot-star/checkpoint-2000/" --write_path "data/dpo_self_samples/qwen2.5coder_3b_instruct_transfer_v4_subtask_cot_star_SFT_step_2000.jsonl" --train_file data/ruff_meta_linting/train_v4_new_format_with_lineno.json --skip_no_violations
    
    # python src/model/eval_llm_meta_linter_vllm_multi_sample.py --model_name "alignment-handbook/model_checkpoints/qwen2.5coder-3b-instruct-sft-trasfer-v4-subtask-cot-star/checkpoint-2000/" --write_path "data/dpo_self_samples/qwen2.5coder_3b_instruct_transfer_v4_subtask_cot_star_SFT_step_2000_from_25000.jsonl" --train_file data/ruff_meta_linting/train_v4_new_format_with_lineno.json --skip_no_violations --skip_index_offset 25000

    # python src/model/eval_llm_meta_linter_vllm_multi_sample.py --model_name "alignment-handbook/model_checkpoints/qwen3-4b-instruct-sft-trasfer-v5-lineno/checkpoint-4000/" --write_path "data/dpo_self_samples/qwen3_4b_transfer_v5_lineno_SFT_step_4000.jsonl" --train_file data/ruff_meta_linting/train_v5.json --skip_no_violations