import json
import numpy as np
from scipy import stats

pairs_to_compare = {
    "Comparing MetaLint Models": [
        ("Qwen3-4B MetaLint (SFT)", "Qwen3-4B MetaLint (SFT+RS-DPO)"),
        ("Qwen3-4B MetaLint w CoT (RS-SFT)", "Qwen3-4B MetaLint w CoT (RS-SFT+RS-DPO)"),
        ("Qwen3-4B MetaLint (SFT)", "Qwen3-4B MetaLint w CoT (RS-SFT)"),
        ("Qwen3-4B MetaLint (SFT+RS-DPO)", "Qwen3-4B MetaLint w CoT (RS-SFT+RS-DPO)")
    ],
    "Comparing MetaLint Models against Untrained Variants": [
        ("Qwen3-4B", "Qwen3-4B MetaLint (SFT)"),
        ("Qwen3-4B", "Qwen3-4B MetaLint (SFT+RS-DPO)"),
        ("Qwen3-4B w CoT", "Qwen3-4B MetaLint w CoT (RS-SFT)"),
        ("Qwen3-4B w CoT", "Qwen3-4B MetaLint w CoT (RS-SFT+RS-DPO)"),
    ],
    "Comparing MetaLint Models against Baselines": [
        ("Qwen3-8B", "Qwen3-4B MetaLint (SFT+RS-DPO)"),
        ("Qwen3-8B w CoT", "Qwen3-4B MetaLint w CoT (RS-SFT+RS-DPO)"),
        ("Qwen3-14B", "Qwen3-4B MetaLint (SFT+RS-DPO)"),
        ("Qwen3-14B w CoT", "Qwen3-4B MetaLint w CoT (RS-SFT+RS-DPO)"),
        ("Qwen3-32B", "Qwen3-4B MetaLint (SFT+RS-DPO)"),
        ("Qwen3-32B w CoT", "Qwen3-4B MetaLint w CoT (RS-SFT+RS-DPO)"),
        ("R1-Distill-Qwen-7B", "Qwen3-4B MetaLint (SFT+RS-DPO)"),
        ("R1-Distill-Qwen-7B", "Qwen3-4B MetaLint w CoT (RS-SFT+RS-DPO)"),
        ("R1-Distill-Qwen-14B", "Qwen3-4B MetaLint (SFT+RS-DPO)"),
        ("R1-Distill-Qwen-14B", "Qwen3-4B MetaLint w CoT (RS-SFT+RS-DPO)"),
        ("R1-Distill-Qwen-32B", "Qwen3-4B MetaLint (SFT+RS-DPO)"),
        ("R1-Distill-Qwen-32B", "Qwen3-4B MetaLint w CoT (RS-SFT+RS-DPO)"),
        ("Qwen2.5-3B-Instruct", "Qwen3-4B MetaLint (SFT+RS-DPO)"),
        ("Qwen2.5-3B-Instruct", "Qwen3-4B MetaLint w CoT (RS-SFT+RS-DPO)"),
        ("Qwen2.5-7B-Instruct", "Qwen3-4B MetaLint (SFT+RS-DPO)"),
        ("Qwen2.5-7B-Instruct", "Qwen3-4B MetaLint w CoT (RS-SFT+RS-DPO)"),
        ("Qwen2.5-14B-Instruct", "Qwen3-4B MetaLint (SFT+RS-DPO)"),
        ("Qwen2.5-14B-Instruct", "Qwen3-4B MetaLint w CoT (RS-SFT+RS-DPO)"),
        ("Qwen2.5-32B-Instruct", "Qwen3-4B MetaLint (SFT+RS-DPO)"),
        ("Qwen2.5-32B-Instruct", "Qwen3-4B MetaLint w CoT (RS-SFT+RS-DPO)"),
        ("Qwen2.5Coder-3B-Instruct", "Qwen3-4B MetaLint (SFT+RS-DPO)"),
        ("Qwen2.5Coder-3B-Instruct", "Qwen3-4B MetaLint w CoT (RS-SFT+RS-DPO)"),
        ("Qwen2.5Coder-7B-Instruct", "Qwen3-4B MetaLint (SFT+RS-DPO)"),
        ("Qwen2.5Coder-7B-Instruct", "Qwen3-4B MetaLint w CoT (RS-SFT+RS-DPO)"),
        ("Qwen2.5Coder-14B-Instruct", "Qwen3-4B MetaLint (SFT+RS-DPO)"),
        ("Qwen2.5Coder-14B-Instruct", "Qwen3-4B MetaLint w CoT (RS-SFT+RS-DPO)"),
        ("Qwen2.5Coder-32B-Instruct", "Qwen3-4B MetaLint (SFT+RS-DPO)"),
        ("Qwen2.5Coder-32B-Instruct", "Qwen3-4B MetaLint w CoT (RS-SFT+RS-DPO)"),
        ("o3-mini", "Qwen3-4B MetaLint (SFT+RS-DPO)"),
        ("o3-mini", "Qwen3-4B MetaLint w CoT (RS-SFT+RS-DPO)"),
    ],
    "Effect of CoT": [
        ("Qwen3-4B", "Qwen3-4B w CoT"),
        ("Qwen3-8B", "Qwen3-8B w CoT"),
        ("Qwen3-14B", "Qwen3-14B w CoT"),
        ("Qwen3-32B", "Qwen3-32B w CoT"),
    ],
    "Effect of Model Scale": [
        ("Qwen3-4B", "Qwen3-8B"),
        ("Qwen3-8B", "Qwen3-14B"),
        ("Qwen3-14B", "Qwen3-32B"),
        ("Qwen3-4B w CoT", "Qwen3-8B w CoT"),
        ("Qwen3-8B w CoT", "Qwen3-14B w CoT"),
        ("Qwen3-14B w CoT", "Qwen3-32B w CoT"),
        ("R1-Distill-Qwen-7B", "R1-Distill-Qwen-14B"),
        ("R1-Distill-Qwen-14B", "R1-Distill-Qwen-32B"),
        ("Qwen2.5Coder-3B-Instruct", "Qwen2.5Coder-7B-Instruct"),
        ("Qwen2.5Coder-7B-Instruct", "Qwen2.5Coder-14B-Instruct"),
        ("Qwen2.5Coder-14B-Instruct", "Qwen2.5Coder-32B-Instruct"),
        ("Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct"),
        ("Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct"),
        ("Qwen2.5-14B-Instruct", "Qwen2.5-32B-Instruct"),
    ],
    "Comparing GPT Models": [
        ("GPT-4o", "GPT-4.1"),
        ("o3-mini", "o4-mini")
    ],
    "Comparing with o3-mini": [
        ("Qwen3-4B", "o3-mini"),
        ("Qwen3-4B w CoT", "o3-mini"),
    ]
}

file_paths = {
    "R1-Distill-Qwen-7B": "statistical_testing/deepseek_r1_distill_qwen_7b_untrained_think_st.json",
    "R1-Distill-Qwen-14B": "statistical_testing/deepseek_r1_distill_qwen_14b_untrained_think_st.json",
    "R1-Distill-Qwen-32B": "statistical_testing/deepseek_r1_distill_qwen_32b_untrained_think_st.json",
    
    "GPT-4.1": "statistical_testing/gpt_4.1_st.json",
    "GPT-4o": "statistical_testing/gpt_4o_st.json",
    "o3-mini": "statistical_testing/o3_mini_st.json", 
    "o4-mini": "statistical_testing/o4_mini_st.json",
    
    "Qwen2.5-3B-Instruct": "statistical_testing/qwen2.5_3b_instruct_untrained_st.json",
    "Qwen2.5-7B-Instruct": "statistical_testing/qwen2.5_7b_instruct_untrained_st.json",
    "Qwen2.5-14B-Instruct": "statistical_testing/qwen2.5_14b_instruct_untrained_st.json",
    "Qwen2.5-32B-Instruct": "statistical_testing/qwen2.5_32b_instruct_untrained_st.json",
    
    "Qwen2.5Coder-3B-Instruct": "statistical_testing/qwen2.5_coder_3b_instruct_st.json",
    "Qwen2.5Coder-7B-Instruct": "statistical_testing/qwen2.5_coder_7b_instruct_st.json",
    "Qwen2.5Coder-14B-Instruct": "statistical_testing/qwen2.5_coder_14b_instruct_st.json",
    "Qwen2.5Coder-32B-Instruct": "statistical_testing/qwen2.5_coder_32b_instruct_st.json",

    "Qwen3-4B": "statistical_testing/qwen3_4b_untrained_no_think_st.json",
    "Qwen3-8B": "statistical_testing/qwen3_8b_untrained_no_think_st.json",
    "Qwen3-14B": "statistical_testing/qwen3_14b_untrained_no_think_st.json",
    "Qwen3-32B": "statistical_testing/qwen3_32b_untrained_no_think_st.json",

    "Qwen3-4B w CoT": "statistical_testing/qwen3_4b_untrained_think_st.json",
    "Qwen3-8B w CoT": "statistical_testing/qwen3_8b_untrained_think_st.json",
    "Qwen3-14B w CoT": "statistical_testing/qwen3_14b_untrained_think_st.json",
    "Qwen3-32B w CoT": "statistical_testing/qwen3_32b_untrained_think_st.json",


    "Qwen3-4B MetaLint (SFT)": "statistical_testing/qwen3_4b_sft_preds_4000_transfer_v5_st.json",
    "Qwen3-4B MetaLint w CoT (RS-SFT)": "statistical_testing/qwen3_4b_think_sft_preds_6000_transfer_v5_st.json",
    "Qwen3-4B MetaLint (SFT+RS-DPO)": "statistical_testing/qwen3_4b_dpo_run2_no_violations_0.05_400_transfer_v5_st.json",
    "Qwen3-4B MetaLint w CoT (RS-SFT+RS-DPO)": "statistical_testing/qwen3_4b_think_dpo_run3_no_violations_0.05_600_transfer_v5_st.json",
}

def paired_statistical_tests(arr_1: np.ndarray, arr_2: np.ndarray):
    if arr_1.shape != arr_2.shape:
        raise ValueError("Input arrays must have the same shape.")

    # Paired t-test (assumes normality of differences)
    t_stat, t_pval = stats.ttest_rel(arr_1, arr_2)

    # Wilcoxon signed-rank test (non-parametric)
    try:
        w_stat, w_pval = stats.wilcoxon(arr_1, arr_2)
    except ValueError as e:
        w_stat, w_pval = None, None  # Wilcoxon may fail on constant inputs or ties

    return {
        # "t_stat": t_stat,
        # "t_pval": t_pval,
        "wilcoxon_stat": w_stat,
        "wilcoxon_pval": w_pval
    }

def load_data(path: str):
    data = json.load(open(path))
    det = np.array([int(i == j) for i,j in zip(data["det"]["predictions"], data["det"]["ground truth"])])
    p_loc = np.array(data["loc"]["p_line"])
    r_loc = np.array(data["loc"]["r_line"])

    return det, p_loc, r_loc 

def colorize(pval: float, stat: float) -> str:
    color = "\033[92m" if pval < alpha else "\033[91m"  # green or red
    reset = "\033[0m"
    return f"{color}{stat:.1f} ({pval:.2e}){reset}"

def format_stat_test_results(pair: tuple[str, str], d: dict, lp: dict, lr: dict, alpha: float = 0.05):
    return (
        f"{pair[0]} vs {pair[1]}:\t"
        f"{colorize(d['wilcoxon_pval'], d['wilcoxon_stat'])}\t"
        f"{colorize(lp['wilcoxon_pval'], lp['wilcoxon_stat'])}\t"
        f"{colorize(lr['wilcoxon_pval'], lr['wilcoxon_stat'])}"
    )

# main
if __name__ == "__main__":
    for table_name, pairs in pairs_to_compare.items():
        alpha = 0.05/len(pairs)
        print("\n\x1b[34;1m"+table_name+f" (𝛼={alpha})\x1b[0m")
        print("Model Comparison\tDetection\tLocalization P\tLocalization R")
        for pair in pairs:
            path_1 = file_paths[pair[0]]
            path_2 = file_paths[pair[1]]
            det_1, p_loc_1, r_loc_1 = load_data(path_1)
            det_2, p_loc_2, r_loc_2 = load_data(path_2)
            
            
            d = paired_statistical_tests(det_1, det_2)
            lp = paired_statistical_tests(p_loc_1, p_loc_2)
            lr = paired_statistical_tests(r_loc_1, r_loc_2)
            print(format_stat_test_results(pair, d, lp, lr, alpha=alpha))
            