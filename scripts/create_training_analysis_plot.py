import os
import sys
import json
import pathlib
import matplotlib.pyplot as plt

def plot_performance(data, path, plot_title: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ['P', 'R', 'F']
    titles = ['Precision', 'Recall', 'F-Score']
    colors = ['b', 'g', 'r']
    
    for i, (metric, title, color) in enumerate(zip(metrics, titles, colors)):
        y_values = data[metric]
        # print(x_values)
        x_values = [2000*(i+1) for i in range(len(y_values))]
        # print(x_values, y_values)
        axes[i].plot(x_values, y_values, marker='o', linestyle='-', color=color, label=metric)
        axes[i].set_title(title)
        axes[i].set_xlabel('Training Steps')
        axes[i].set_xticks(x_values, labels=x_values)
        # axes[i].set_ylabel('Score')
        axes[i].set_ylim(min(y_values) * 0.9, max(y_values) * 1.1)  # Adjust scale individually
        axes[i].legend()
    
    # fig.suptitle(plot_title)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(path)

# main
if __name__ == "__main__":
    # idiom detection plots: training analysis
    overall_data = {"P": [], "R": [], "F": []}
    not_data = {"P": [], "R": [], "F": []}
    net_data = {"P": [], "R": [], "F": []}
    fat_data = {"P": [], "R": [], "F": []}
    tool_data = {}
    for train_steps in [2000, 4000, 6000]:
        # print()
        evals_json = json.load(open(f"data/meta_linting_evals/idiom_detection/qwen2.5coder_3b_instruct_sft_preds_{train_steps}-v2-data.json"))
        for i,metric in enumerate(overall_data):
            # x = 2000*(i+1)
            overall_data[metric].append(round(evals_json['overall'][metric], 4))
        for i,metric in enumerate(not_data):
            # x = 2000*(i+1)
            not_data[metric].append(round(evals_json['NoT'][metric], 4))
        for i,metric in enumerate(net_data):
            # x = 2000*(i+1)
            net_data[metric].append(round(evals_json['NeT'][metric], 4))
        for i,metric in enumerate(not_data):
            # x = 2000*(i+1)
            fat_data[metric].append(round(evals_json['FaT'][metric], 4))
        for tool, metrics_dict in evals_json["tool_groups"].items():
            if train_steps == 2000:
                tool_data[tool] = {"P": [], "R": [], "F": []}
            for metric in metrics_dict:
                tool_data[tool][metric].append(round(metrics_dict[metric], 4))

    # print(overall_data)
    plot_performance(overall_data, f"data/meta_linting_evals/idiom_detection/plots/overall.png", plot_title="Overall")
    plot_performance(not_data, f"data/meta_linting_evals/idiom_detection/plots/no_transfer.png", plot_title="No Transfer")
    plot_performance(net_data, f"data/meta_linting_evals/idiom_detection/plots/near_transfer.png", plot_title="Near Transfer")
    plot_performance(fat_data, f"data/meta_linting_evals/idiom_detection/plots/far_transfer.png", plot_title="Far Transfer")
    for tool, tool_specific_data in tool_data.items():
        plot_performance(tool_specific_data, f"data/meta_linting_evals/idiom_detection/plots/{tool}.png", plot_title=f"{tool} Idiom Det.")