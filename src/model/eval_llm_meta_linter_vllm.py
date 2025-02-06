# script to evaluate the performance of trained LLMs on the meta-linting task:

import os
import sys
import json
import pathlib

sys.path.append(str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent))

from src.datautils import META_LINTING_PROMPT
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# main
if __name__ == "__main__":
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("alignment-handbook/model_checkpoints/qwen2.5-7b-1M-instruct-sft/checkpoint-2000")

    # Pass the default decoding hyperparameters of Qwen2.5-7B-Instruct
    # max_tokens is for the maximum length for generation.
    sampling_params = SamplingParams(temperature=0.1, top_p=0.95, 
                                     #repetition_penalty=1.05, 
                                     max_tokens=2048)

    # Input the model name or path. See below for parameter explanation (after the example of openai-like server).
    llm = LLM(model="alignment-handbook/model_checkpoints/qwen2.5-7b-1M-instruct-sft/checkpoint-2000",
        tensor_parallel_size=4,
        max_model_len=1010000,
        enable_chunked_prefill=True,
        max_num_batched_tokens=131072,
        enforce_eager=True,
        # quantization="fp8", # Enabling FP8 quantization for model weights can reduce memory usage.
    )

    test_data = json.load(open("data/ruff_meta_linting/test.json"))
    messages = [
        {"role": "user", "content": test_data[0]['messages'][0]['content']}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # generate outputs
    outputs = llm.generate([text], sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        # print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        print(generated_text)