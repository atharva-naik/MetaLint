import os
import sys
import json
import openai
import pathlib
import tiktoken
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

module_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent)
sys.path.append(module_path)

from src.datautils import generate_cot_gen_prompts, load_ruff_idiom_specs, load_stack_dump, read_jsonl

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

# main
if __name__ == "__main__":
    openai.api_key = os.environ["ORACLE_PROJECT_COT_API_KEY"]
    client = openai.OpenAI(api_key=os.environ["ORACLE_PROJECT_COT_API_KEY"])

    sft_train_data = json.load(open("./data/ruff_meta_linting/train_v4.json"))
    stack_data = load_stack_dump("./data/STACK-V2", as_dict=True)
    code_idiom_specs = load_ruff_idiom_specs("./data/ruff_pages")
    
    cot_gen_prompts = generate_cot_gen_prompts(sft_train_data, stack_data, code_idiom_specs, save_path="./data/ruff_meta_linting/cot_gen/train_v4.jsonl")
    
    # print(f"${estimate_gpt4o_mini_cost([rec['prompt'] for rec in cot_gen_prompts]):.2f}")
    # print(f"${estimate_gpt4o_mini_cost([rec['prompt'] for rec in cot_gen_prompts if rec["no_violations"] == False]):.2f}")
    model = "gpt-4o-mini"
    encoder = tiktoken.encoding_for_model("gpt-4o-mini")
    cache_path = f"./data/ruff_meta_linting/cot_gen/{model}-cot-gen-cache.jsonl"
    if os.path.exists(cache_path):
        cache = {rec["id"]: rec for rec in read_jsonl(cache_path)}
    else: cache = {}
    break_ctr = 0
    pbar = tqdm(cot_gen_prompts, desc="getting ChatGPT CoT")
    for rec in pbar:
        if rec["id"] in cache: 
            break_ctr += 1
            continue
        if rec["no_violations"] == False:
            pbar.set_description(f"getting CoTs ({break_ctr}/2000)")
            prompt = rec["prompt"]
            tokens = encoder.encode(prompt, disallowed_special=())
            if len(tokens)>2488: continue # 3000-512
            # print(prompt)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512
            )
            response = response.choices[0].message.content
            # print(response)
            with open(cache_path, "a") as f:
                write_rec = {}
                write_rec.update(rec)
                write_rec["response"] = response
                f.write(json.dumps(write_rec)+"\n")
                break_ctr += 1
            if break_ctr >= 40000: break
            # break