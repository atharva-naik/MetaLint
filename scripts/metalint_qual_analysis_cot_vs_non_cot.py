import os
import sys
import json
from pathlib import Path
from collections import Counter

# print(str(Path(os.path.abspath(__file__)).parent.parent))
sys.path.append(str(Path(os.path.abspath(__file__)).parent.parent))

from src.datautils import read_jsonl
from src.metrics.meta_linting.idiom_detection_and_localization_v4 import load_linter_results


# main
if __name__ == "__main__":
    metalint_CoT_preds = read_jsonl("data/pep_benchmark_preds_v2/qwen3_4b_think_dpo_run3_no_violations_0.05_600_transfer_v5_preds.jsonl")
    metalint_preds = read_jsonl("data/pep_benchmark_preds_v2/qwen3_4b_dpo_run2_no_violations_0.05_400_transfer_v5_preds.jsonl")
    CoT_detection_failures = []
    for rec_CoT, rec in zip(metalint_CoT_preds, metalint_preds):
        cot_resp = load_linter_results(rec_CoT["model_response"])
        resp = load_linter_results(rec["model_response"])
        assert rec["ground_truth"] == rec_CoT["ground_truth"]
        gt = load_linter_results(rec["ground_truth"])
        if cot_resp == [] and gt != [] and resp != []: # cases where CoT model predicts no violation even though there are 1 or more violations and the non CoT model also detects a violation.
            rec_CoT["cot_model_response"] = rec_CoT["model_response"]
            rec_CoT["non_cot_model_response"] = rec["model_response"]
            del rec_CoT["model_response"]
            CoT_detection_failures.append(rec_CoT)
    print(len(CoT_detection_failures))
    with open("CoT_detection_failure_qwen3_4b_think_dpo.json", "w") as f:
        json.dump(CoT_detection_failures, f, indent=4)
    PEP_dist = Counter([rec["source"].split("/")[-1] for rec in CoT_detection_failures]).most_common()
    print(PEP_dist)
    print(len(PEP_dist))