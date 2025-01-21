# Analyze IDIOM/pattern matches.

import os
import ast
import sys
import json
import math
import numpy as np
from typing import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import matplotlib.pyplot as plt
from collections import defaultdict
from src.datautils import read_jsonl, load_stack_dump, strip_comments_and_docstrings

def analyze_pep_occurrence_stats(all_tree_patterns):
    all_matches = []
    pep_dist = defaultdict(lambda: 0)
    for rec in all_tree_patterns:
        for matched_pep, matches in rec['patterns'].items():
            for match in matches:
                pep_dist[matched_pep] += 1
                all_matches.append((matched_pep, match))
    Z = sum(pep_dist.values())
    pep_dist_abs = {k: v for k,v in pep_dist.items()}
    pep_dist = {k: round(100*v/Z, 4) for k,v in pep_dist.items()}
    pep_dist = dict(sorted(pep_dist.items(), reverse=True, key=lambda x: x[1]))
    pep_dist_abs = dict(sorted(pep_dist_abs.items(), reverse=True, key=lambda x: x[1]))
    
    return pep_dist, pep_dist_abs

def transform_pep_label(pep: str):
    return pep.replace("pep_", "PEP ")

def aggregate_violations_and_adherence(pep_dist: dict[str, Union[float, int]]):
    pep_dist_agg = defaultdict(lambda:0)
    for k,v in pep_dist.items():
        pep_dist_agg[k.replace("v","")] += v

    return dict(pep_dist_agg)

def plot_pep_histogram(data: dict[str, float], path: str, use_logscale: bool=False):
    # Extract data for plotting
    labels = list(data.keys())
    if use_logscale:
        values = [math.log(v) for v in list(data.values())]
    else: values = list(data.values())

    # Create line plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(labels, values, marker='o', label='Values', color='blue')
    ax.fill_between(labels, values, color='blue', alpha=0.3)

    # Adjust x-axis labels for readability
    plt.xticks(rotation=45, ha='right', fontsize=8)
    ax.set_xlabel("Labels")
    ax.set_ylabel("Values")
    ax.set_title("Distribution of Labels (Decreasing Order)")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend()

    plt.tight_layout()
    plt.savefig(path)

def plot_pep_dist(data: dict[str, float], path: str, threshold: float=1):
    # Process data: Group values less than 1 into "Others"
    # threshold = 1
    main_data = {transform_pep_label(k): v for k, v in data.items() if v >= threshold}
    others_value = sum(v for v in data.values() if v < threshold)
    if others_value > 0:
        main_data["Others"] = others_value

    # Prepare data for the pie chart
    labels = list(main_data.keys())
    sizes = list(main_data.values())

    # Define a custom color palette
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))

    # Create pie chart with a consistent color scheme
    fig, ax = plt.subplots(figsize=(10, 7))
    wedges, _ = ax.pie(sizes, wedgeprops=dict(width=0.4), startangle=140, colors=colors)

    # Add labels with arrows
    for i, (label, wedge) in enumerate(zip(labels, wedges)):
        theta = (wedge.theta2 + wedge.theta1) / 2
        x = 1.2 * np.cos(np.radians(theta))
        y = 1.2 * np.sin(np.radians(theta))
        ax.annotate(
            label,
            xy=(wedge.r * np.cos(np.radians(theta)), wedge.r * np.sin(np.radians(theta))),
            xytext=(x, y),
            arrowprops=dict(arrowstyle="->", color='black'),
            horizontalalignment='center',
            verticalalignment='center'
        )

    ax.set_title("Idiom Distribution")
    plt.savefig(path)

def pep_co_occurr(all_tree_patterns: dict[str, Union[dict, str]], stack_data=None):
    co_occurr_stats = defaultdict(lambda: 0)
    total_count = defaultdict(lambda: 0)
    for file_matches in all_tree_patterns:
        PEPs_found = set()
        for pep in file_matches["patterns"]:
            pep = pep.replace("v","")
            PEPs_found.add(pep)
        for pep in PEPs_found:
            total_count[pep] += 1

    examples = defaultdict(lambda: {})
    ctr = 0
    for file_matches in all_tree_patterns:
        PEPs_found = set()
        for pep in file_matches["patterns"]:
            pep = pep.replace("v","")
            PEPs_found.add(pep)
        PEPs_found = list(PEPs_found)
        for i in range(len(PEPs_found)):
            for j in range(i+1, len(PEPs_found)):
                PEPa = min(PEPs_found[i], PEPs_found[j])
                PEPb = max(PEPs_found[i], PEPs_found[j])
                co_occurr_stats[f"{PEPa}-{PEPb}"] += 1/math.sqrt(total_count[PEPa]*total_count[PEPb])
                if PEPa == "pep_525" and PEPb == "pep_567":
                    # print(file_matches['blob_id'])
                    # print(file_matches["patterns"][PEPa])
                    if stack_data:
                        examples[f"{PEPa}-{PEPb}"][ctr] = {'file': strip_comments_and_docstrings(stack_data[file_matches['blob_id']]['content']), 'patterns': file_matches['patterns']}
                        ctr += 1
    if stack_data:
        with open("./data/pep_co_occurr_examples.json", "w") as f:
            json.dump(dict(examples), f, indent=4)
                    
    co_occurr_stats = dict(co_occurr_stats)
    co_occurr_stats = dict(sorted(co_occurr_stats.items(), reverse=True, key=lambda x: x[1]))

    return co_occurr_stats

# main
if __name__ == "__main__":
    stack_data = {rec['blob_id']: rec for rec in load_stack_dump("./data/STACK-V2")}
    all_tree_patterns = read_jsonl("./data/pattern_mining/tree_patterns/all_patterns_streaming.jsonl")
    pep_dist, pep_dist_abs = analyze_pep_occurrence_stats(all_tree_patterns)
    pep_dist = aggregate_violations_and_adherence(pep_dist)
    
    # print(pep_dist)
    pep_dist_abs = aggregate_violations_and_adherence(pep_dist_abs)
    # print(pep_dist_abs)
    
    # make all plots
    plot_pep_dist(pep_dist, "plots/PEP_dist.png")
    plot_pep_histogram(pep_dist_abs, "plots/PEP_dist_tail.png")
    plot_pep_histogram(pep_dist_abs, "plots/PEP_dist_tail_log.png", use_logscale=True)
    
    co_occurr_stats = pep_co_occurr(all_tree_patterns, stack_data=stack_data)
    # print(co_occurr_stats)