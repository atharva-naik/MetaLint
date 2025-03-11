import os
import sys
import json
import pathlib
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

module_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent)
sys.path.append(module_path)

from src.datautils import read_jsonl

if __name__ == "__main__":
    try: train_steps: int=int(sys.argv[1])
    except IndexError: train_steps: int=2000
    # model_name = f"alignment-handbook/model_checkpoints/qwen2.5coder-3b-instruct-sft-v2-data/checkpoint-{train_steps}"
    model_name = f"alignment-handbook/model_checkpoints/qwen2.5coder-3b-instruct-sft-idiom-hardness-v3/checkpoint-{train_steps}"
    #"Qwen/Qwen2.5-7B-Instruct-1M"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # test_data = json.load(open("data/ruff_meta_linting/test_v3.json"))
    test_data = json.load(open("data/ruff_meta_linting/hardness_experiment/test.json"))
    model_preds = []
    write_path = f"./data/meta_linting_preds/qwen2.5coder_3b_instruct_sft_preds_{train_steps}-idiom-hardness-v3.jsonl"
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
    for index,rec in pbar:
        if index < skip_index_till: continue
        elif index == skip_index_till:
            print(f"resuming inference from index: {index}")
        user_prompt = rec['messages'][0]['content']

        # truncate extremely long user prompts.
        # print(len(user_prompt))
        if len(user_prompt) > 800000:
            user_prompt = rec['messages'][0]['content'][:100000]+rec['messages'][0]['content'][-100000:]
        messages = [
            {"role": "user", "content": user_prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=1024,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # print(response)
        # print(rec['messages'][1]['content'])
        model_preds.append({
            'model_response': response,
            "ground_truth": rec['messages'][1]['content'],
        })
        with open(write_path, "a") as f:
            f.write(json.dumps({
            'model_response': response,
            "ground_truth": rec['messages'][1]['content'],
        })+"\n")