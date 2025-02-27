import os
import sys
import json
import pathlib
import numpy as np
from fuzzywuzzy import fuzz
from typing import Dict, List, Tuple

module_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent.parent)
sys.path.append(module_path)

from src.datautils import read_jsonl_safe

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
                    print(f"JSON decoding error for {FAULTY_RESULT_CTR} linter result predictions")
                    continue
    return results

def merge_same_code_violations(violations: list[dict]) -> list[dict]:
    code_map = {}
    for violation in violations:
        code = violation.get('code')
        if code not in code_map:
            code_map[code] = violation.copy()
        else:
            code_map[code]['code_spans_and_lines'].extend(violation.get('code_spans_and_lines', []))
    return list(code_map.values())

def calculate_metrics(matrix: np.ndarray) -> Tuple[float, float, float]:
    rows, cols = matrix.shape
    precision_counter = sum(1 for row in matrix if any(row))
    precision = precision_counter / rows if rows > 0 else 0
    
    recall_counter = sum(1 for col in matrix.T if any(col))
    recall = recall_counter / cols if cols > 0 else 0
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def evaluate_responses(file_path: str) -> Dict[str, Dict[str, float]]:
    data = read_jsonl_safe(file_path)
    metrics_by_code = {}
    
    for idx, row in enumerate(data):
        model_resp = row['model_response']
        ground_truth = row['ground_truth']
        
        if model_resp == "NO VIOLATIONS FOUND" or ground_truth == "NO VIOLATIONS FOUND":
            continue
            
        try:
            if isinstance(model_resp, str):
                model_json = parse_json_response(model_resp) if '\n' in model_resp else [json.loads(model_resp)]
            else:
                model_json = [model_resp] if isinstance(model_resp, dict) else model_resp
                
            if isinstance(ground_truth, str):
                truth_json = parse_json_response(ground_truth) if '\n' in ground_truth else [json.loads(ground_truth)]
            else:
                truth_json = [ground_truth] if isinstance(ground_truth, dict) else truth_json
            
            model_violations = merge_same_code_violations(model_json)
            truth_violations = merge_same_code_violations(truth_json)
            
            for model_viol in model_violations:
                model_code = model_viol.get('code')
                matching_truth = next((t for t in truth_violations if t.get('code') == model_code), None)
                
                if not matching_truth:
                    continue
                    
                model_spans = model_viol.get('code_spans_and_lines', [])
                truth_spans = matching_truth.get('code_spans_and_lines', [])
                
                if not model_spans or not truth_spans:
                    continue
                    
                # Create matrices for both exact and fuzzy matching
                line_matrix = np.zeros((len(model_spans), len(truth_spans)))
                span_matrix = np.zeros((len(model_spans), len(truth_spans)))
                fuzzy_line_matrix = np.zeros((len(model_spans), len(truth_spans)))
                fuzzy_span_matrix = np.zeros((len(model_spans), len(truth_spans)))
                
                for i, model_span in enumerate(model_spans):
                    for j, truth_span in enumerate(truth_spans):
                        # Exact matching
                        if model_span.get('line') == truth_span.get('line'):
                            line_matrix[i, j] = 1
                        if model_span.get('span') == truth_span.get('span'):
                            span_matrix[i, j] = 1
                        
                        # Fuzzy matching with 90% threshold
                        fuzzy_line_ratio = fuzz.ratio(str(model_span.get('line', '')), str(truth_span.get('line', '')))
                        fuzzy_span_ratio = fuzz.ratio(str(model_span.get('span', '')), str(truth_span.get('span', '')))
                        
                        fuzzy_line_matrix[i, j] = 1 if fuzzy_line_ratio >= 90 else 0
                        fuzzy_span_matrix[i, j] = 1 if fuzzy_span_ratio >= 90 else 0
                
                # Calculate metrics
                line_precision, line_recall, line_f1 = calculate_metrics(line_matrix)
                span_precision, span_recall, span_f1 = calculate_metrics(span_matrix)
                fuzzy_line_precision, fuzzy_line_recall, fuzzy_line_f1 = calculate_metrics(fuzzy_line_matrix)
                fuzzy_span_precision, fuzzy_span_recall, fuzzy_span_f1 = calculate_metrics(fuzzy_span_matrix)
                
                if model_code not in metrics_by_code:
                    metrics_by_code[model_code] = {
                        'line_precision': [], 'line_recall': [], 'line_f1': [],
                        'span_precision': [], 'span_recall': [], 'span_f1': [],
                        'fuzzy_line_precision': [], 'fuzzy_line_recall': [], 'fuzzy_line_f1': [],
                        'fuzzy_span_precision': [], 'fuzzy_span_recall': [], 'fuzzy_span_f1': []
                    }
                
                metrics = metrics_by_code[model_code]
                # Store all metrics
                metrics['line_precision'].append(line_precision)
                metrics['line_recall'].append(line_recall)
                metrics['line_f1'].append(line_f1)
                metrics['span_precision'].append(span_precision)
                metrics['span_recall'].append(span_recall)
                metrics['span_f1'].append(span_f1)
                metrics['fuzzy_line_precision'].append(fuzzy_line_precision)
                metrics['fuzzy_line_recall'].append(fuzzy_line_recall)
                metrics['fuzzy_line_f1'].append(fuzzy_line_f1)
                metrics['fuzzy_span_precision'].append(fuzzy_span_precision)
                metrics['fuzzy_span_recall'].append(fuzzy_span_recall)
                metrics['fuzzy_span_f1'].append(fuzzy_span_f1)
                
        except json.JSONDecodeError as e:
            print(f"JSON decode error in row {idx}: {str(e)}")
            continue
    
    # Calculate final averages
    final_metrics = {}
    for code, metrics in metrics_by_code.items():
        final_metrics[code] = {
            metric: sum(values) / len(values)
            for metric, values in metrics.items()
        }
    
    return final_metrics

# main
if __name__ == "__main__":
    # Get script directory and construct path
    
    try: train_steps: int=int(sys.argv[1])
    except IndexError: train_steps: int=2000
    # model_preds = read_jsonl_safe(f"data/meta_linting_preds/qwen2.5coder_3b_instruct_sft_preds_{train_steps}.jsonl")
    file_path = f"data/meta_linting_preds/qwen2.5coder_3b_instruct_sft_preds_{train_steps}-v2-data.jsonl"

    # Get results
    results = evaluate_responses(file_path)

    # Create dictionaries for metrics
    metrics_dict = {
        'exact_line': {'P': {}, 'R': {}, 'F': {}},
        'exact_span': {'P': {}, 'R': {}, 'F': {}},
        'fuzzy_line': {'P': {}, 'R': {}, 'F': {}},
        'fuzzy_span': {'P': {}, 'R': {}, 'F': {}}
    }

    # Organize results
    for code, metrics in results.items():
        metrics_dict['exact_line']['P'][code] = metrics['line_precision']
        metrics_dict['exact_line']['R'][code] = metrics['line_recall']
        metrics_dict['exact_line']['F'][code] = metrics['line_f1']
        
        metrics_dict['exact_span']['P'][code] = metrics['span_precision']
        metrics_dict['exact_span']['R'][code] = metrics['span_recall']
        metrics_dict['exact_span']['F'][code] = metrics['span_f1']
        
        metrics_dict['fuzzy_line']['P'][code] = metrics['fuzzy_line_precision']
        metrics_dict['fuzzy_line']['R'][code] = metrics['fuzzy_line_recall']
        metrics_dict['fuzzy_line']['F'][code] = metrics['fuzzy_line_f1']
        
        metrics_dict['fuzzy_span']['P'][code] = metrics['fuzzy_span_precision']
        metrics_dict['fuzzy_span']['R'][code] = metrics['fuzzy_span_recall']
        metrics_dict['fuzzy_span']['F'][code] = metrics['fuzzy_span_f1']

    # Print results
    for metric_type in ['exact_line', 'fuzzy_line', 'exact_span', 'fuzzy_span']:
        print(f"\n{metric_type.upper()} METRICS:")
        # print("Precision:", metrics_dict[metric_type]['P'])
        print(f"P: {np.mean(list(metrics_dict[metric_type]['P'].values())):.4f}")
        # print("Recall:", metrics_dict[metric_type]['R'])
        print(f"R: {np.mean(list(metrics_dict[metric_type]['R'].values())):.4f}")
        # print("F1:", metrics_dict[metric_type]['F'])
        print(f"F: {np.mean(list(metrics_dict[metric_type]['F'].values())):.4f}")
