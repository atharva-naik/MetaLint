import json
import numpy as np
from typing import List, Dict

FAULTY_RESULT_CTR = 0

def parse_json_response(response_str: str) -> List[Dict]:
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
                    return []  # Return empty list on JSON decode error
    return results

def read_jsonl(file_path: str) -> List[Dict]:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data

def create_comparison_matrix(model_json: List[Dict], truth_json: List[Dict], field: str) -> np.ndarray:
    model_values = []
    truth_values = []
    model_codes = []
    truth_codes = []

    for model_viol in model_json:
        model_code = model_viol['code']
        model_codes.extend([model_code] * len(model_viol['code_spans_and_lines']))
        model_values.extend([entry[field] for entry in model_viol['code_spans_and_lines']])

    for truth_viol in truth_json:
        truth_code = truth_viol['code']
        truth_codes.extend([truth_code] * len(truth_viol['code_spans_and_lines']))
        truth_values.extend([entry[field] for entry in truth_viol['code_spans_and_lines']])

    m = len(model_values)
    n = len(truth_values)
    matrix = np.zeros((m, n))

    for i, model_value in enumerate(model_values):
        for j, truth_value in enumerate(truth_values):
            if model_value == truth_value and model_codes[i] == truth_codes[j]:
                matrix[i, j] = 1

    return matrix

def calculate_metrics(matrix: np.ndarray) -> tuple[float, float, float]:
    rows, cols = matrix.shape
    row_counter = sum(1 for row in matrix if any(row))
    precision = row_counter / rows if rows > 0 else 0
    
    col_counter = sum(1 for col in matrix.T if any(col))
    recall = col_counter / cols if cols > 0 else 0
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def main():
    file_path = "../../data/meta_linting_preds/qwen2.5coder_3b_instruct_sft_preds_8000-idiom-hardness-v3.jsonl"
    data = read_jsonl(file_path)
    
    num_rows = len(data)
    
    line_precision_matrix = np.zeros(num_rows)
    line_recall_matrix = np.zeros(num_rows)
    line_f1_matrix = np.zeros(num_rows)
    span_precision_matrix = np.zeros(num_rows)
    span_recall_matrix = np.zeros(num_rows)
    span_f1_matrix = np.zeros(num_rows)
    reward_matrix = np.zeros(num_rows)

    for idx, row in enumerate(data):
        model_resp = row['model_response']
        ground_truth = row['ground_truth']
        
        if model_resp == "NO VIOLATIONS FOUND" and ground_truth == "NO VIOLATIONS FOUND":
            line_precision_matrix[idx] = 1.0
            line_recall_matrix[idx] = 1.0
            line_f1_matrix[idx] = 1.0
            span_precision_matrix[idx] = 1.0
            span_recall_matrix[idx] = 1.0
            span_f1_matrix[idx] = 1.0
            reward_matrix[idx] = 1.0
        elif model_resp == "NO VIOLATIONS FOUND" or ground_truth == "NO VIOLATIONS FOUND":
            line_precision_matrix[idx] = 0.0
            line_recall_matrix[idx] = 0.0
            line_f1_matrix[idx] = 0.0
            span_precision_matrix[idx] = 0.0
            span_recall_matrix[idx] = 0.0
            span_f1_matrix[idx] = 0.0
            reward_matrix[idx] = 0.0
        else:
            try:
                if isinstance(model_resp, str):
                    model_json = parse_json_response(model_resp) if '\n' in model_resp else [json.loads(model_resp)]
                else:
                    model_json = [model_resp] if isinstance(model_resp, dict) else model_resp
                    
                if isinstance(ground_truth, str):
                    truth_json = parse_json_response(ground_truth) if '\n' in ground_truth else [json.loads(ground_truth)]
                else:
                    truth_json = [ground_truth] if isinstance(ground_truth, dict) else truth_json
                
                if not model_json or not truth_json:  # Skip if either is empty due to JSON decode error
                    continue
                
                line_matrix = create_comparison_matrix(model_json, truth_json, 'line')
                span_matrix = create_comparison_matrix(model_json, truth_json, 'span')

                line_precision, line_recall, line_f1 = calculate_metrics(line_matrix)
                span_precision, span_recall, span_f1 = calculate_metrics(span_matrix)

                line_precision_matrix[idx] = line_precision
                line_recall_matrix[idx] = line_recall
                line_f1_matrix[idx] = line_f1
                span_precision_matrix[idx] = span_precision
                span_recall_matrix[idx] = span_recall
                span_f1_matrix[idx] = span_f1
                reward_matrix[idx] = line_f1 * span_f1
            except json.JSONDecodeError:
                continue  # Skip row on JSON decode error
        
        print(f"Row {idx+1} done.")

    # Save results to a file
    import os
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'reward_matrices.json')

    results_dict = {
        "Line Precision": line_precision_matrix.tolist(),
        "Line Recall": line_recall_matrix.tolist(),
        "Line F1": line_f1_matrix.tolist(),
        "Span Precision": span_precision_matrix.tolist(),
        "Span Recall": span_recall_matrix.tolist(),
        "Span F1": span_f1_matrix.tolist(),
        "Reward": reward_matrix.tolist()
    }

    zero_count = sum(1 for val in reward_matrix if val == 0)
    one_count = sum(1 for val in reward_matrix if val == 1)
    other_count = len(reward_matrix) - zero_count - one_count

    print(f"Reward values of 0: {zero_count}")
    print(f"Reward values of 1: {one_count}")
    print(f"Reward values otherwise: {other_count}")

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

if __name__ == "__main__":
    main()