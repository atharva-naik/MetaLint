import json
import os
import random
import asyncio
import requests
from datasets import load_dataset

# vLLM server details
VLLM_SERVER_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "Qwen/Qwen2.5-Coder-3B-Instruct"

# Load dataset
dataset = load_dataset("teledia-cmu/ruff-meta-linting-v2-5-idiom-subsets")

# Get random sample of 500 indices
total_size = len(dataset['train'])
random_indices = random.sample(range(total_size), 500)

# Output file
output_file = 'baseline_responses_vllm.json'
if not os.path.exists(output_file):
    with open(output_file, 'w') as f:
        json.dump({"data": []}, f)

async def generate_response(idx):
    """ Sends a request to vLLM for generating a response """
    record = dataset['train'][idx]
    record_id = record['id']
    source = record.get('source', '')
    prompt = record['messages'][0]['content']
    chosen_content = record['messages'][1]['content']  # Ground truth response

    chosen = [
        {"content": prompt, "role": "user"}, 
        {"content": chosen_content, "role": "assistant"}
    ]
    
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": 1024
    }

    try:
        response = requests.post(VLLM_SERVER_URL, json=payload).json()
        rejected_content = response["choices"][0]["message"]["content"]

        rejected = [
            {"content": rejected_content, "role": "assistant"}
        ]

        return {
            "id": record_id,
            "source": source,
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        }
    except Exception as e:
        print(f"Error processing record {record_id}: {e}")
        return {
            "id": record_id,
            "source": source,
            "prompt": prompt,
            "chosen": chosen,
            "rejected": None
        }

async def process_records():
    """ Processes dataset records asynchronously and writes responses to a file """
    results = []
    total = len(random_indices)
    
    for i, idx in enumerate(random_indices):
        result = await generate_response(idx)
        results.append(result)
        
        print(f"Generated {i+1}/{total} responses")
        
        # Save results periodically (every 10 responses)
        if (i+1) % 10 == 0 or i == total-1:
            with open(output_file, 'r') as f:
                data = json.load(f)
            
            data["data"] = results if "data" not in data else data["data"] + results
            
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=4)
            
            results = []
            
    print(f"Completed processing all {total} records.")

# Run async processing
asyncio.run(process_records())