import json
import numpy as np
import re
import os

FAULTY_RESULT_CTR = 0

def parse_json_response(response_str: str) -> list[dict]:
    results = []
    for line in response_str.split('\n'):
        line = line.strip()
        if line:
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                try:
                    line = line.encode().decode('unicode_escape')
                    results.append(json.loads(line))
                except (json.JSONDecodeError, UnicodeError):
                    global FAULTY_RESULT_CTR
                    FAULTY_RESULT_CTR += 1
                    print(f"JSON decoding error for {FAULTY_RESULT_CTR} linter result predictions")
                    return []  # Return empty list on JSON decode error
    return results

def parse_idiom_violations(text):
    # Parse the new format with idiom sections
    idioms = {}
    current_idiom = None
    violations = []
    
    for line in text.split('\n'):
        if line.startswith('**Idiom ') and line.endswith(':**'):
            # Save previous idiom if exists
            if current_idiom:
                idioms[current_idiom] = violations
                violations = []
            
            # Extract new idiom name
            current_idiom = re.search(r'\*\*Idiom (.*?) Violations:\*\*', line).group(1)
        elif line.strip() == 'NO VIOLATIONS FOUND':
            violations = 'NO VIOLATIONS FOUND'
        elif line.startswith('{') and line.endswith('}'):
            try:
                violation = json.loads(line)
                violations.append(violation)
            except json.JSONDecodeError:
                continue
    
    # Save the last idiom
    if current_idiom:
        idioms[current_idiom] = violations
    
    return idioms

def calculate_metrics(matrix):
    rows, cols = matrix.shape
    if rows == 0 or cols == 0:
        return 0, 0, 0
        
    row_counter = sum(1 for row in matrix if any(row))
    precision = row_counter / rows if rows > 0 else 0
    
    col_counter = sum(1 for col in matrix.T if any(col))
    recall = col_counter / cols if cols > 0 else 0
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def evaluate_instance(model_response, ground_truth):
    # Parse idioms from both responses
    model_idioms = parse_idiom_violations(model_response)
    truth_idioms = parse_idiom_violations(ground_truth)
    
    # Track metrics for each idiom
    idiom_rewards = []
    
    # Process each idiom
    for idiom in set(model_idioms.keys()) | set(truth_idioms.keys()):
        model_violations = model_idioms.get(idiom, 'NO VIOLATIONS FOUND')
        truth_violations = truth_idioms.get(idiom, 'NO VIOLATIONS FOUND')
        
        # Both have no violations - perfect match
        if model_violations == 'NO VIOLATIONS FOUND' and truth_violations == 'NO VIOLATIONS FOUND':
            idiom_rewards.append(1.0)
            continue
            
        # One has violations, other doesn't - no match
        if model_violations == 'NO VIOLATIONS FOUND' or truth_violations == 'NO VIOLATIONS FOUND':
            idiom_rewards.append(0.0)
            continue
        
        # Create matrices for line and span matching
        model_lines = [v.get('line', '') for v in model_violations]
        truth_lines = [v.get('line', '') for v in truth_violations]
        model_spans = [v.get('span', '') for v in model_violations]
        truth_spans = [v.get('span', '') for v in truth_violations]
        
        # Create line matrix
        line_matrix = np.zeros((len(model_lines), len(truth_lines)))
        for i, model_line in enumerate(model_lines):
            for j, truth_line in enumerate(truth_lines):
                if model_line == truth_line:
                    line_matrix[i, j] = 1
        
        # Create span matrix
        span_matrix = np.zeros((len(model_spans), len(truth_spans)))
        for i, model_span in enumerate(model_spans):
            for j, truth_span in enumerate(truth_spans):
                if model_span == truth_span:
                    span_matrix[i, j] = 1
        
        # Calculate metrics
        line_precision, line_recall, line_f1 = calculate_metrics(line_matrix)
        span_precision, span_recall, span_f1 = calculate_metrics(span_matrix)
        
        # Calculate reward for this idiom
        idiom_reward = line_f1 * span_f1
        idiom_rewards.append(idiom_reward)
    
    # Calculate final reward as average across idioms
    final_reward = sum(idiom_rewards) / len(idiom_rewards) if idiom_rewards else 0
    return final_reward

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                global FAULTY_RESULT_CTR
                FAULTY_RESULT_CTR += 1
                print(f"JSON decoding error for {FAULTY_RESULT_CTR} JSONL file lines")
    return data

def main():
    # Load and process the dataset
    file_path = "../../data/meta_linting_preds/qwen2.5coder_3b_instruct_sft_preds_8000_transfer_v4_lineno.jsonl"
    data = read_jsonl(file_path)
    
    rewards = []
    
    for idx, instance in enumerate(data):
        model_response = instance['model_response']
        ground_truth = instance['ground_truth']
        
        reward = evaluate_instance(model_response, ground_truth)
        rewards.append(reward)
        
        print(f"Row {idx+1} done.")
    
    # Save results to a file
    results_dict = {
        "Reward": rewards
    }
    
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'reward_metrics.json')
    
    with open(output_file, 'w') as f:
        f.write("{\n")
        for idx, (key, value) in enumerate(results_dict.items()):
            f.write(f"  \"{key}\": [{', '.join(map(str, value))}]")
            if idx < len(results_dict) - 1:
                f.write(",\n")
            else:
                f.write("\n")
        f.write("}")
    
    print(f"Metrics saved to {output_file}")
    print(f"Total JSON decoding errors: {FAULTY_RESULT_CTR}")

if __name__ == "__main__":
    main()
