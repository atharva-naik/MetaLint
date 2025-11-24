import os
import sys
import ast
import json
import pathlib
import pandas as pd

project_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent)
sys.path.append(project_path)

from src.datautils import read_jsonl
from src.metrics.meta_linting.idiom_detection_and_localization_v4 import load_linter_results

# main
if __name__ == "__main__":
    qwen_3_4b_preds = read_jsonl("data/pep_benchmark_preds_v2/qwen3_4b_dpo_run2_no_violations_0.05_400_transfer_v5_preds.jsonl")
    qwen_3_4b_cot_preds = read_jsonl("data/pep_benchmark_preds_v2/qwen3_4b_think_dpo_run3_no_violations_0.05_600_transfer_v5_preds.jsonl")

    loc_err_analysis = []
    for rec in qwen_3_4b_preds:
        pred_lines = load_linter_results(rec['model_response'])
        gt_lines = load_linter_results(rec['ground_truth'])
        pred_linenos = set()
        gt_linenos = set()

        pred_code_blocks = []
        gt_code_blocks = []
        for pred_rec in pred_lines:
            pred_code_blocks.append(pred_rec['line'])
            for line in pred_rec['line'].split("\n"):
                pred_linenos.add(line.split()[0].strip()) 
        for gt_rec in gt_lines:
            gt_code_blocks.append(gt_rec['line'])
            for line in gt_rec['line'].split("\n"):
                gt_linenos.add(line.split()[0].strip()) 
        
        # collect cases with no overlapping lines between gt and preds.
        if len(pred_linenos.intersection(gt_linenos)) == 0 and len(gt_linenos) !=0 and len(pred_linenos) != 0:
            loc_err_analysis.append({
                "id": rec['id'], 'source': rec['source'],
                'model_response': rec['model_response'],
                'gt_lines': "\n".join(gt_code_blocks), 'pred_lines': "\n".join(pred_code_blocks),
                'gt_linenos': gt_linenos, 'pred_linenos': pred_linenos,
            })

    print(len(loc_err_analysis))
    pd.DataFrame(loc_err_analysis).to_csv("data/pep_benchmark/loc_error_analysis/qwen3_4b_dpo_run2_no_violations_0.05_400_transfer_v5.csv", index=False)
