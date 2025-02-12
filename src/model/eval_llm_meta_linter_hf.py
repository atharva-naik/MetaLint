import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

if __name__ == "__main__":
    model_name = "alignment-handbook/model_checkpoints/qwen2.5coder-3b-instruct-sft/checkpoint-2000"
    #"Qwen/Qwen2.5-7B-Instruct-1M"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    test_data = json.load(open("data/ruff_meta_linting/test_v2.json"))
    model_preds = []
    write_path = "./data/meta_linting_preds/qwen2.5coder_3b_instruct_sft_preds.jsonl"
    f = open(write_path, "w")
    for rec in tqdm(test_data):
        messages = [
            {"role": "user", "content": rec['messages'][0]['content']}
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
            f.write(json.dumps(model_preds[-1])+"\n")