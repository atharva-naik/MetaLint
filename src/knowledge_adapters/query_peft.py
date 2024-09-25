# script to do inference with/query LLM + PEFT trained knowledge adapter.
import os
import torch
import transformers
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_prompt_inference(prompt, eos_token="</s>"):
    return prompt+"\n"#+response+' '+eos_token

model_name = "ise-uiuc/Magicoder-S-DS-6.7B"
# "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name, load_in_8bit=True, device_map="auto",
)

tokenizer.add_special_tokens({"pad_token": "<PAD>"})
model.resize_token_embeddings(len(tokenizer))

peft_model_id = "experiments/ise-uiuc/Magicoder-S-DS-6.7B-3.11-klora/checkpoint-4000"
# "experiments/ise-uiuc/Magicoder-S-DS-6.7B-3.9-klora/checkpoint-3000"
# "experiments/llama3-8b-py3.10-klora/final"
peft_model = PeftModel.from_pretrained(model, peft_model_id, torch_dtype=torch.float16, offload_folder="experiments/lora_results/temp")


# input_prompt = generate_prompt_inference("Can you list all the changes introduced in Python 3.10?")

TEST_PROMPTS = [
    # "Describe a release summary of Python 3.11.",
    # "Describe \"summary-release-highlights\" introduced in Python 3.11.",
    "Describe summary release highlights introduced in Python 3.11."
    # "Can you list all the changes introduced in Python 3.10?",
    # "Can you list all the changes introduced in Python 3.11?",
    # "Can you list all the changes introduced in Python 2.6?",
    # "Describe the 'new-features - pep-657-fine-grained-error-locations-in-tracebacks' in Python 3.11.",
    # "Describe the 'new-features - pep-657-fine-grained-error-locations-in-tracebacks'.",
]
for prompt in TEST_PROMPTS:
    input_prompt = generate_prompt_inference(prompt)
    input_tokens = tokenizer(input_prompt, return_tensors="pt")["input_ids"].to("cuda")
    with torch.cuda.amp.autocast():
        generation_output = peft_model.generate(
            input_ids=input_tokens,
            max_new_tokens=500,
            do_sample=True,
            top_k=10,
            top_p=0.9,
            temperature=0.3,
            repetition_penalty=1.15,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
        )
    op = tokenizer.decode(generation_output[0], skip_special_tokens=True)
    print(op)