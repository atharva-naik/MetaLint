import sys
import torch
import transformers
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from collections import defaultdict
from peft import (
        get_peft_model, 
        prepare_model_for_kbit_training, 
        LoraConfig
    )
from trl import SFTTrainer
from datasets import Dataset
from src.datautils import load_python_whatsnew_dataset

# def check_python_version(required_version):
#     """Checks if the Python version matches the required version."""
#     current_version = sys.version_info
#     if (current_version.major, current_version.minor) != required_version:
#         raise ValueError(f"Python {required_version[0]}.{required_version[1]} is required, but you are using "
#                          f"{current_version.major}.{current_version.minor}.{current_version.micro}")
def parse_arguments():
    parser = argparse.ArgumentParser(description="PEFT Fine-Tuning Task Arguments Parser")

    # Required arguments
    parser.add_argument('--train_dataset', type=str, required=True, help='Path to the training dataset')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate for training')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save the output model')
    parser.add_argument('--model_name', type=str, default="meta-llama/Meta-Llama-3-8B", help="base model used for PEFT.")

    # Optional arguments
    parser.add_argument('--model_saving_strategy', type=str, default="epoch", 
                        choices=["epoch", "step"], help='Strategy to save the model: "epoch" or "step"')
    parser.add_argument('--train_steps', type=int, default=1000, help='Maximum number of training steps (use instead of epochs)')
    parser.add_argument('--save_steps', type=int, default=50, help='Number of steps after which to save')
    parser.add_argument('--epochs', type=int, help='Number of training epochs (use instead of steps)')

    # Python version check
    parser.add_argument('--python_version', type=str, required=True, 
                        help='The Python version (docs and PEPs) for training the knowledge expert')

    args = parser.parse_args()

    # Split and validate Python version
    version_parts = args.python_version.split('.')
    if len(version_parts) != 2:
        raise ValueError("Python version must be in the format 'major.minor', e.g., '3.8'.")
    
    # required_version = (int(version_parts[0]), int(version_parts[1]))
    # check_python_version(required_version)

    # Ensure either train_steps or epochs is provided
    if args.train_steps is None and args.epochs is None:
        raise ValueError("Either --train_steps or --epochs must be provided.")
    
    if args.train_steps is not None and args.epochs is not None:
        raise ValueError("You must provide only one of --train_steps or --epochs, not both.")

    return args

def generate_prompt(prompt, response, eos_token="</s>"):
    # instruction = "Summarize the following:\n"
    # input = f"{dialogue}\n"
    # summary = f"Summary: {summary + ' ' + eos_token if summary else ''} "
    # prompt = (" ").join([instruction, input, summary])
    return prompt+"\n"+response+' '+eos_token

if __name__ == "__main__":
    args = parse_arguments()

    # model_name = "meta-llama/Llama-2-7b-hf" # "meta-llama/Meta-Llama-3-8B" # "meta-llama/Llama-2-7b-hf"
    # model_name = "deepseek-ai/deepseek-coder-7b-instruct"
    model_name = args.model_name
    model = AutoModelForCausalLM.from_pretrained(
        model_name, load_in_8bit=True, device_map="auto",
    )
    # quantization_type=BitsAndBytesConfig()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"training knowledge adapter for {args.model_name} for Python {args.python_version}")

    # load and filter out data corresponding to a specific Python version.
    data = load_python_whatsnew_dataset("./data/Python-docs/whatsnew.jsonl")
    py_version_wise_data = defaultdict(lambda: [])
    for rec in data:
        py_version_wise_data[rec['python_version']].append(rec)
    
    data = Dataset.from_list(py_version_wise_data[args.python_version])
    print(len(data))
    # exit()
    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # this should be set for finutning and batched inference
    tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    model.resize_token_embeddings(len(tokenizer))

    # Loading in 8 bit ..."
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    output_dir = args.output_dir
    per_device_train_batch_size = args.batch_size
    gradient_accumulation_steps = 4
    per_device_eval_batch_size = 4
    eval_accumulation_steps = 4
    optim = "paged_adamw_32bit"
    save_steps = args.save_steps
    logging_steps = args.save_steps # 10
    learning_rate = args.learning_rate
    max_grad_norm = 0.3
    max_steps = args.train_steps
    warmup_ratio = 0.03
    evaluation_strategy="steps"
    lr_scheduler_type = "constant"

    training_args = transformers.TrainingArguments(
                output_dir=output_dir,
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                optim=optim,
                evaluation_strategy=evaluation_strategy,
                save_steps=save_steps,
                learning_rate=learning_rate,
                logging_steps=logging_steps,
                max_grad_norm=max_grad_norm,
                max_steps=max_steps,
                warmup_ratio=warmup_ratio,
                group_by_length=True,
                lr_scheduler_type=lr_scheduler_type,
                ddp_find_unused_parameters=False,
                eval_accumulation_steps=eval_accumulation_steps,
                per_device_eval_batch_size=per_device_eval_batch_size,
            )

    def formatting_func(prompt):
        output = []

        for d, s in zip(prompt["prompt"], prompt["response"]):
            op = generate_prompt(d, s)
            output.append(op)

        return output

    # same train and eval data because we want to overfit to the data.
    trainer = SFTTrainer(
        model=model,
        train_dataset=data,
        eval_dataset=data,
        peft_config=lora_config,
        formatting_func=formatting_func,
        max_seq_length=1024,
        tokenizer=tokenizer,
        args=training_args
    )

    # We will also pre-process the model by upcasting the layer norms in float 32 for more stable training
    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)

    trainer.train()
    trainer.save_model(f"{output_dir}/final")

    # python -m src.knowledge_adapters.train_peft --train_dataset ./data/Python-docs/whatsnew.jsonl --output_dir experiments/llama2-7b-py3.10-klora --python_version 3.10