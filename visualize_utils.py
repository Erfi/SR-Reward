"""
Functions used to visualize the results of the model.
"""

import glob
import os
from pathlib import Path
from typing import Dict, List
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from eval_utils import recompose_config, normalize_reward
from scipy.stats import ttest_ind, mannwhitneyu


def get_normalized_mean_std(test_eval_paths: List[Path], expert_score, random_score):
    all_means = []
    for test_eval_path in test_eval_paths:
        data = np.load(test_eval_path)
        try:
            rewards = data["rewards"]
            mean = np.mean(rewards)
        except KeyError:
            mean = data["mean_reward"]
        all_means.append(mean)
    # MEDIAN and INTER-QUANTILE RANGE
    median = np.median(all_means)
    if len(all_means) == 0:
        interquartile_range = 0
    else:
        interquartile_range = np.percentile(all_means, 75) - np.percentile(
            all_means, 25
        )
    # MEAN and STD
    mean = np.mean(all_means)
    std = np.std(all_means)

    normalized_all_means = normalize_reward(
        np.array(all_means), expert_score, random_score
    )
    # NORMALIZED --- MEDIAN and INTER-QUANTILE RANGE
    median_normalized = np.median(normalized_all_means)
    if len(normalized_all_means) == 0:
        interquartile_normalized_range = 0
    else:
        interquartile_normalized_range = np.percentile(
            normalized_all_means, 75
        ) - np.percentile(normalized_all_means, 25)
    # NORMALIZED --- MEAN and STD
    mean_normalized = np.mean(normalized_all_means)
    std_normalized = np.std(normalized_all_means)
    return (
        median_normalized,
        interquartile_normalized_range,
        mean_normalized,
        std_normalized,
        normalized_all_means,
        median,
        interquartile_range,
        mean,
        std,
        all_means,
    )


def get_validation_normalized_mean_std(
    validation_eval_paths: List[Path], expert_score, random_score
):
    """
    The validation paths are evaluation.npz files.
    They should contain a "results" key with shape (N, M) where N is the number of validations steps
    and M is the number of rollouts per validation step.

    At each validation step we compute the mean over the rollouts (M) to get an array of shape (N) per file.
    If there are multiple files (S seeds), we append the results to get an array of (S, N).
    We then compute the mean and std over the seeds (S) to get an array of shape (N).
    """

    all_means = []
    for validation_eval_path in validation_eval_paths:
        data = np.load(validation_eval_path)
        rewards = data["results"]  # shape (N, M)
        mean = np.mean(rewards, axis=1)  # shape (N)
        all_means.append(mean)
    try:
        all_means = np.array(all_means).reshape(
            len(validation_eval_paths), -1
        )  # shape (S, N)
    except ValueError:  # handle uneven lengths by filling with mean
        max_length = max(len(x) for x in all_means)
        all_means = np.array(
            [
                np.concatenate([x, np.full(max_length - len(x), np.mean(x))])
                for x in all_means
            ]
        )  # shape (S, N)
    # MEDIAN and INTER-QUANTILE RANGE
    median = np.median(all_means, axis=0)  # shape (N)
    if len(all_means) == 0:
        interquartile_range = 0
    else:
        interquartile_range = np.percentile(all_means, 75, axis=0) - np.percentile(
            all_means, 25, axis=0
        )
    # MEAN and STD
    mean = np.mean(all_means, axis=0)  # shape (N)
    std = np.std(all_means, axis=0)  # shape (N)

    normalized_all_means = normalize_reward(all_means, expert_score, random_score)
    # NORMALIZED --- MEDIAN and INTER-QUANTILE RANGE
    median_normalized = np.median(normalized_all_means, axis=0)  # shape (N)
    if len(normalized_all_means) == 0:
        interquartile_normalized_range = 0
    else:
        interquartile_normalized_range = np.percentile(
            normalized_all_means, 75, axis=0
        ) - np.percentile(normalized_all_means, 25, axis=0)
    # NORMALIZED --- MEAN and STD
    mean_normalized = np.mean(normalized_all_means, axis=0)  # shape (N)
    std_normalized = np.std(normalized_all_means, axis=0)  # shape (N)
    return (
        median_normalized,
        interquartile_normalized_range,
        mean_normalized,
        std_normalized,
        normalized_all_means,
        median,
        interquartile_range,
        mean,
        std,
        all_means,
    )


def dataset_returns_plot(returns: np.ndarray, env_name: str):
    """
    Histogram of the return values.
    """
    os.makedirs("plots", exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.hist(returns, bins=100, color="indigo", alpha=0.8, rwidth=0.8)
    ax.set_xlabel("Return", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(env_name, fontsize=16)
    fig.tight_layout()
    fig.savefig(f"plots/{env_name}_returns.png")
    fig.savefig(f"plots/{env_name}_returns.pdf")


def baseline_plot(results: Dict):
    os.makedirs("plots", exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    y = np.arange(len(results))
    env_to_color = {
        "ant": "indigo",
        "halfcheetah": "orange",
        "hopper": "teal",
        "walker2d": "indianred",
        "door": "darkgreen",
        "hammer": "darkorange",
        "pen": "darkblue",
        "relocate": "darkred",
        "pickcube": "darkgreen",
        "stackcube": "darkorange",
        "turnfaucet": "darkblue",
        "navmap": "darkred",
    }
    colors = [env_to_color[k.split("_")[0]] for k in results.keys()]

    def create_label(key: str):
        key = key.split("_")
        if key[-1] == "default":
            end = "(SR-Reward)"
        elif key[-1] == "default-true-reward":
            end = "(True Reward)"
        if key[1] == "bc":
            return f"{key[0].capitalize()} BC"
        return f"{key[0].capitalize()} {key[1].upper()} {end}"

    labels = [create_label(k) for k in results.keys()]
    bars = ax.barh(
        y=y,
        width=[results[k]["mean"] for k in results.keys()],
        xerr=[results[k]["std"] for k in results.keys()],
        tick_label=labels,
        align="center",
        alpha=0.8,
        ecolor="black",
        capsize=6,
        color=colors,
        zorder=3,
    )
    bar_labels = [
        f"{results[k]['mean']:.2f} ± {results[k]['std']:.2f}" for k in results.keys()
    ]
    ax.bar_label(bars, labels=bar_labels, label_type="edge", padding=5)
    ax.tick_params(axis="x", labelrotation=90)
    ax.spines[["right", "left", "top", "bottom"]].set_visible(False)
    ax.set_xlabel("Normalized Return")
    fig.tight_layout()
    fig.savefig("plots/baseline.png")
    fig.savefig("plots/baseline.pdf")


def baseline_validation_plot(results: Dict):
    """
    Create a set of line plots one for each environment.
    Each plot contains the results for each algorithm with the mean and std of the seeds.
    """
    os.makedirs("plots", exist_ok=True)
    environments = [
        "ant",
        "halfcheetah",
        "hopper",
        "walker2d",
        "door",
        "hammer",
        "pen",
        "relocate",
        "pickcube",
        "stackcube",
        "turnfaucet",
    ]
    algorithms = [
        "bc",
        "sparseql",
        "fdvl",
    ]
    experiments = [
        "default",
        "default-true-reward",
    ]
    algo_experiment_colors = {
        "bc": {"default": "green"},
        "sparseql": {"default": "red", "default-true-reward": "darkorange"},
        "fdvl": {"default": "blue", "default-true-reward": "black"},
    }
    fig, ax = plt.subplots(6, 2, figsize=(20, 30))
    ax = ax.flatten()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for i, env in enumerate(environments):
        for j, algo in enumerate(algorithms):
            for k, experiment in enumerate(experiments):
                key = f"{env}_{algo}_{experiment}"
                if key not in results:
                    continue
                # Smoothen the mean and std using a window
                window = 5
                std_frac = 1.0
                mean = np.convolve(
                    results[key]["mean"], np.ones(window) / window, mode="valid"
                )
                std = (
                    np.convolve(
                        results[key]["std"], np.ones(window) / window, mode="valid"
                    )
                    * std_frac
                )
                # mean = results[key]["mean"]
                # std = results[key]["std"]
                ax[i].plot(
                    np.arange(len(mean)),
                    mean,
                    color=algo_experiment_colors[algo][experiment],
                    linewidth=2,
                    alpha=0.5,
                    label=f"{algo.upper()} {experiment.replace('default','').replace('-', ' ')}",
                )
                ax[i].fill_between(
                    np.arange(len(mean)),
                    mean - std,
                    mean + std,
                    color=algo_experiment_colors[algo][experiment],
                    alpha=0.1,
                )
        ax[i].legend(loc="upper left", fontsize=12)
        ax[i].set_title(f"{env.capitalize()}", size=16)
        ax[i].set_xlabel("Validation Step", size=12)
        ax[i].set_ylabel("Normalized Return", size=12)
        ax[i].spines[["right", "left", "top", "bottom"]].set_visible(False)
        ax[i].set_facecolor("ghostwhite")

    # Hide the last plot
    ax[-1].axis("off")
    fig.tight_layout()
    fig.savefig("plots/baseline_validation.png")
    fig.savefig("plots/baseline_validation.pdf")


def data_quality_lineplot(results: Dict, envs_group: str, std_fraction: float = 1.0):
    """
    Plot the data quality results for the mujoco or adroit environments.

    Args:
        results: The results dictionary.
        envs_group: The group of environments to plot. Either "mujoco" or "adroit".
    """
    if envs_group == "mujoco":
        envs = ["walker2d", "ant", "halfcheetah", "hopper"]
        quality_list = ["medium", "medium_expert", "expert"]
    elif envs_group == "adroit":
        envs = ["door", "hammer", "pen", "relocate"]
        quality_list = ["human", "expert"]
    else:
        raise ValueError("envs_group must be either 'mujoco' or 'adroit'")

    os.makedirs("plots", exist_ok=True)
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    algos = ["bc", "sparseql_true_reward", "sparseql"]
    colors = ["teal", "orange", "indigo"]
    plot_label_map = {
        "bc": "BC",
        "sparseql": "SR-Reward",
        "sparseql_true_reward": "True Reward",
    }
    for i, env in enumerate(envs):
        for algo, color in zip(algos, colors):
            if envs_group == "mujoco":
                result_list = [
                    results[f"{env}_{algo}_medium"]["mean"],
                    results[f"{env}_{algo}_medium_expert"]["mean"],
                    results[f"{env}_{algo}_expert"]["mean"],
                ]
                result_list_std = [
                    results[f"{env}_{algo}_medium"]["std"],
                    results[f"{env}_{algo}_medium_expert"]["std"],
                    results[f"{env}_{algo}_expert"]["std"],
                ]
            elif envs_group == "adroit":
                result_list = [
                    results[f"{env}_{algo}_human"]["mean"],
                    results[f"{env}_{algo}_expert"]["mean"],
                ]
                result_list_std = [
                    results[f"{env}_{algo}_human"]["std"],
                    results[f"{env}_{algo}_expert"]["std"],
                ]
            ax[i].plot(
                quality_list,
                result_list,
                marker="o",
                markersize=10,
                color=color,
                label=plot_label_map[algo],
                linewidth=3,
                alpha=0.8,
            )
            ax[i].fill_between(
                quality_list,
                np.array(result_list) - np.array(result_list_std) * std_fraction,
                np.array(result_list) + np.array(result_list_std) * std_fraction,
                color=color,
                alpha=0.1,
            )
        ax[i].set_yticks(np.arange(0, 151, 50))
        ax[i].spines[["right", "left", "top", "bottom"]].set_visible(False)
        ax[i].set_facecolor("ghostwhite")
        ax[i].set_ylim(0, 150)
        ax[i].set_title(f"{env.capitalize()}", size=16)
        ax[i].tick_params(axis="both", which="major", labelsize=14)
        if i == 0:
            ax[i].set_ylabel("Normalized Return", size=16)
        if i == len(envs) - 1:
            handles, labels = ax[i].get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                loc="lower center",
                ncols=3,
                fontsize=12,
                bbox_to_anchor=(0.5, -0.01),
            )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    fig.savefig(f"plots/data_quality_{envs_group}.png")
    fig.savefig(f"plots/data_quality_{envs_group}.pdf")


def data_quality_barplot(results: Dict, envs_group: str, std_fraction: float = 1.0):
    """
    Plot the data quality results for the mujoco or adroit environments as a bar plot.

    Args:
        results: The results dictionary.
        envs_group: The group of environments to plot. Either "mujoco" or "adroit".
    """
    if envs_group == "mujoco":
        envs = ["walker2d", "ant", "halfcheetah", "hopper"]
        quality_list = ["medium", "medium_expert", "expert"]
    elif envs_group == "adroit":
        envs = ["door", "hammer", "pen", "relocate"]
        quality_list = ["human", "expert"]
    else:
        raise ValueError("envs_group must be either 'mujoco' or 'adroit'")

    os.makedirs("plots", exist_ok=True)
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    algos = ["bc", "sparseql_true_reward", "sparseql"]
    colors = ["teal", "orange", "indigo"]
    plot_label_map = {
        "bc": "BC",
        "sparseql": "SR-Reward",
        "sparseql_true_reward": "True Reward",
    }
    # create grouped bar plot, grouping the bars by data size
    bar_width = 0.2
    for i, env in enumerate(envs):
        for j, algo in enumerate(algos):
            if envs_group == "mujoco":
                result_list = [
                    results[f"{env}_{algo}_medium"]["mean"],
                    results[f"{env}_{algo}_medium_expert"]["mean"],
                    results[f"{env}_{algo}_expert"]["mean"],
                ]
                result_list_std = [
                    results[f"{env}_{algo}_medium"]["std"],
                    results[f"{env}_{algo}_medium_expert"]["std"],
                    results[f"{env}_{algo}_expert"]["std"],
                ]
            elif envs_group == "adroit":
                result_list = [
                    results[f"{env}_{algo}_human"]["mean"],
                    results[f"{env}_{algo}_expert"]["mean"],
                ]
                result_list_std = [
                    results[f"{env}_{algo}_human"]["std"],
                    results[f"{env}_{algo}_expert"]["std"],
                ]
            ax[i].bar(
                np.arange(len(quality_list)) + j * bar_width,
                result_list,
                bar_width,
                color=colors[j],
                label=plot_label_map[algo],
            )
            ax[i].errorbar(
                np.arange(len(quality_list)) + j * bar_width,
                result_list,
                yerr=np.array(result_list_std) * std_fraction,
                fmt="none",
                ecolor="black",
                capsize=5,
                alpha=0.3,
            )
        ax[i].set_yticks(np.arange(0, 151, 50))
        ax[i].spines[["right", "left", "top", "bottom"]].set_visible(False)
        ax[i].set_facecolor("ghostwhite")
        ax[i].set_ylim(0, 150)
        ax[i].set_title(f"{env.capitalize()}", size=16)
        ax[i].tick_params(axis="both", which="major", labelsize=12)
        ax[i].set_xticks(np.arange(len(quality_list)) + bar_width)
        ax[i].set_xticklabels(quality_list)
        ax[i].set_xlabel("Data Quality", size=16)
        if i == 0:
            ax[i].set_ylabel("Normalized Return", size=16)
        if i == len(envs) - 1:
            handles, labels = ax[i].get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                loc="lower center",
                ncols=3,
                fontsize=12,
                bbox_to_anchor=(0.5, -0.01),
            )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    fig.savefig(f"plots/data_quality_{envs_group}_barplot.png")
    fig.savefig(f"plots/data_quality_{envs_group}_barplot.pdf")


def data_quality_boxplot(results: Dict, envs_group: str):
    os.makedirs("plots", exist_ok=True)
    if envs_group == "mujoco":
        envs = ["walker2d", "ant", "halfcheetah", "hopper"]
        quality_list = ["medium", "medium_expert", "expert"]
    elif envs_group == "adroit":
        envs = ["door", "hammer", "pen", "relocate"]
        quality_list = ["human", "expert"]

    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    algos = ["bc", "sparseql_true_reward", "sparseql"]
    colors = ["teal", "orange", "indigo"]
    plot_label_map = {
        "bc": "BC",
        "sparseql": "SR-Reward",
        "sparseql_true_reward": "True Reward",
    }
    box_width = 0.2
    for i, env in enumerate(envs):
        for j, algo in enumerate(algos):
            if envs_group == "mujoco":
                results_list = [
                    results[f"{env}_{algo}_medium"]["rewards"],
                    results[f"{env}_{algo}_medium_expert"]["rewards"],
                    results[f"{env}_{algo}_expert"]["rewards"],
                ]
            elif envs_group == "adroit":
                results_list = [
                    results[f"{env}_{algo}_human"]["rewards"],
                    results[f"{env}_{algo}_expert"]["rewards"],
                ]
            ax[i].boxplot(
                results_list,
                positions=np.arange(len(quality_list)) + j * (box_width + 0.03),
                widths=box_width,
                patch_artist=True,
                boxprops=dict(facecolor=colors[j], color=colors[j]),
                whiskerprops=dict(color=colors[j]),
                capprops=dict(color=colors[j]),
                medianprops=dict(color="ghostwhite", alpha=0.5),
                flierprops=dict(marker="o", markersize=1, color=colors[j], alpha=0.3),
            )
        ax[i].set_yticks(np.arange(0, 151, 50))
        ax[i].spines[["right", "left", "top", "bottom"]].set_visible(False)
        ax[i].set_facecolor("ghostwhite")
        ax[i].set_ylim(0, 150)
        ax[i].set_title(f"{env.capitalize()}", size=16)
        ax[i].tick_params(axis="both", which="major", labelsize=12)
        ax[i].set_xticks(np.arange(len(quality_list)) + box_width)
        ax[i].set_xticklabels(quality_list)
        ax[i].set_xlabel("Data Quality", size=16)
        if i == 0:
            ax[i].set_ylabel("Normalized Return", size=16)
        if i == len(envs) - 1:
            handles = [plt.Rectangle((0, 0), 1, 1, color=colors[j]) for j in range(3)]
            labels = [plot_label_map[algo] for algo in algos]
            fig.legend(
                handles,
                labels,
                loc="lower center",
                ncols=3,
                fontsize=12,
                bbox_to_anchor=(0.5, -0.01),
            )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    fig.savefig(f"plots/data_quality_{envs_group}_boxplot.png")
    fig.savefig(f"plots/data_quality_{envs_group}_boxplot.pdf")


def data_size_lineplot(results: Dict, envs_group: str, std_fraction: float = 1.0):
    os.makedirs("plots", exist_ok=True)
    if envs_group == "mujoco":
        envs = ["walker2d", "ant", "halfcheetah", "hopper"]
        data_size_list = [10, 50, 100, 500, 1000]
    elif envs_group == "adroit":
        envs = ["door", "hammer", "pen", "relocate"]
        data_size_list = [10, 50, 100, 500, 1000, 5000]

    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    algos = ["bc", "sparseql_true_reward", "sparseql"]
    colors = ["teal", "orange", "indigo"]
    plot_label_map = {
        "bc": "BC",
        "sparseql": "SR-Reward",
        "sparseql_true_reward": "True Reward",
    }
    for i, env in enumerate(envs):
        for algo, color in zip(algos, colors):
            if envs_group == "mujoco":
                results_list = [
                    results[f"{env}_{algo}_10"]["mean"],
                    results[f"{env}_{algo}_50"]["mean"],
                    results[f"{env}_{algo}_100"]["mean"],
                    results[f"{env}_{algo}_500"]["mean"],
                    results[f"{env}_{algo}_1000"]["mean"],
                ]
                results_list_std = [
                    results[f"{env}_{algo}_10"]["std"],
                    results[f"{env}_{algo}_50"]["std"],
                    results[f"{env}_{algo}_100"]["std"],
                    results[f"{env}_{algo}_500"]["std"],
                    results[f"{env}_{algo}_1000"]["std"],
                ]
            elif envs_group == "adroit":
                results_list = [
                    results[f"{env}_{algo}_10"]["mean"],
                    results[f"{env}_{algo}_50"]["mean"],
                    results[f"{env}_{algo}_100"]["mean"],
                    results[f"{env}_{algo}_500"]["mean"],
                    results[f"{env}_{algo}_1000"]["mean"],
                    results[f"{env}_{algo}_5000"]["mean"],
                ]
                results_list_std = [
                    results[f"{env}_{algo}_10"]["std"],
                    results[f"{env}_{algo}_50"]["std"],
                    results[f"{env}_{algo}_100"]["std"],
                    results[f"{env}_{algo}_500"]["std"],
                    results[f"{env}_{algo}_1000"]["std"],
                    results[f"{env}_{algo}_5000"]["std"],
                ]
            ax[i].plot(
                data_size_list,
                results_list,
                marker="o",
                markersize=10,
                color=color,
                label=plot_label_map[algo],
                linewidth=3,
                alpha=0.8,
            )
            ax[i].fill_between(
                data_size_list,
                np.array(results_list) - np.array(results_list_std) * std_fraction,
                np.array(results_list) + np.array(results_list_std) * std_fraction,
                color=color,
                alpha=0.1,
            )
        ax[i].set_yticks(np.arange(0, 201, 50))
        ax[i].spines[["right", "left", "top", "bottom"]].set_visible(False)
        ax[i].set_facecolor("ghostwhite")
        ax[i].set_ylim(0, 210)
        ax[i].set_title(f"{env.capitalize()}", size=16)
        ax[i].tick_params(axis="both", which="major", labelsize=14)
        ax[i].set_xscale("log")
        if i == 0:
            ax[i].set_ylabel("Normalized Return", size=16)
        if i == len(envs) - 1:
            handles, labels = ax[i].get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                loc="lower center",
                ncols=3,
                fontsize=12,
                bbox_to_anchor=(0.5, -0.01),
            )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    fig.savefig(f"plots/data_size_{envs_group}_lineplot.png")
    fig.savefig(f"plots/data_size_{envs_group}_lineplot.pdf")


def data_size_barplot(results: Dict, envs_group: str, std_fraction: float = 1.0):
    os.makedirs("plots", exist_ok=True)
    if envs_group == "mujoco":
        envs = ["walker2d", "ant", "halfcheetah", "hopper"]
        data_size_list = [10, 50, 100, 500]
    elif envs_group == "adroit":
        envs = ["door", "hammer", "pen", "relocate"]
        data_size_list = [10, 50, 100, 500]

    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    algos = ["bc", "sparseql_true_reward", "sparseql"]
    colors = ["teal", "orange", "indigo"]
    barcolors = ["teal", "orange", "indigo"]
    plot_label_map = {
        "bc": "BC",
        "sparseql": "SR-Reward",
        "sparseql_true_reward": "True Reward",
    }
    # create grouped bar plot, grouping the bars by data size
    bar_width = 0.2
    for i, env in enumerate(envs):
        for j, algo in enumerate(algos):
            if envs_group == "mujoco":
                results_list = [
                    results[f"{env}_{algo}_10"]["mean"],
                    results[f"{env}_{algo}_50"]["mean"],
                    results[f"{env}_{algo}_100"]["mean"],
                    results[f"{env}_{algo}_500"]["mean"],
                    # results[f"{env}_{algo}_1000"]["mean"],
                ]
                results_list_std = [
                    results[f"{env}_{algo}_10"]["std"],
                    results[f"{env}_{algo}_50"]["std"],
                    results[f"{env}_{algo}_100"]["std"],
                    results[f"{env}_{algo}_500"]["std"],
                    # results[f"{env}_{algo}_1000"]["std"],
                ]
            elif envs_group == "adroit":
                results_list = [
                    results[f"{env}_{algo}_10"]["mean"],
                    results[f"{env}_{algo}_50"]["mean"],
                    results[f"{env}_{algo}_100"]["mean"],
                    results[f"{env}_{algo}_500"]["mean"],
                    # results[f"{env}_{algo}_1000"]["mean"],
                    # results[f"{env}_{algo}_5000"]["mean"],
                ]
                results_list_std = [
                    results[f"{env}_{algo}_10"]["std"],
                    results[f"{env}_{algo}_50"]["std"],
                    results[f"{env}_{algo}_100"]["std"],
                    results[f"{env}_{algo}_500"]["std"],
                    # results[f"{env}_{algo}_1000"]["std"],
                    # results[f"{env}_{algo}_5000"]["std"],
                ]
            ax[i].bar(
                np.arange(len(data_size_list)) + j * bar_width,
                results_list,
                bar_width,
                color=colors[j],
                label=plot_label_map[algo],
            )
            ax[i].errorbar(
                np.arange(len(data_size_list)) + j * bar_width,
                results_list,
                yerr=np.array(results_list_std) * std_fraction,
                fmt="none",
                ecolor="black",
                capsize=5,
                alpha=0.3,
            )
        ax[i].set_yticks(np.arange(0, 201, 50))
        ax[i].spines[["right", "left", "top", "bottom"]].set_visible(False)
        ax[i].set_facecolor("ghostwhite")
        ax[i].set_ylim(0, 210)
        ax[i].set_title(f"{env.capitalize()}", size=16)
        ax[i].tick_params(axis="both", which="major", labelsize=12)
        ax[i].set_xticks(np.arange(len(data_size_list)) + bar_width)
        ax[i].set_xticklabels(data_size_list)
        ax[i].set_xlabel("Data Size", size=16)
        if i == 0:
            ax[i].set_ylabel("Normalized Return", size=16)
        if i == len(envs) - 1:
            handles, labels = ax[i].get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                loc="lower center",
                ncols=3,
                fontsize=12,
                bbox_to_anchor=(0.5, -0.01),
            )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    fig.savefig(f"plots/data_size_{envs_group}_barplot.png")
    fig.savefig(f"plots/data_size_{envs_group}_barplot.pdf")


def data_size_boxplot(results: Dict, envs_group: str):
    os.makedirs("plots", exist_ok=True)
    if envs_group == "mujoco":
        envs = ["walker2d", "ant", "halfcheetah", "hopper"]
        data_size_list = [10, 50, 100, 500, 1000]
    elif envs_group == "adroit":
        envs = ["door", "hammer", "pen", "relocate"]
        data_size_list = [10, 50, 100, 500, 1000, 5000]

    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    algos = ["bc", "sparseql_true_reward", "sparseql"]
    colors = ["teal", "orange", "indigo"]
    plot_label_map = {
        "bc": "BC",
        "sparseql": "SR-Reward",
        "sparseql_true_reward": "True Reward",
    }
    box_width = 0.2
    for i, env in enumerate(envs):
        for j, algo in enumerate(algos):
            if envs_group == "mujoco":
                results_list = [
                    results[f"{env}_{algo}_10"]["rewards"],
                    results[f"{env}_{algo}_50"]["rewards"],
                    results[f"{env}_{algo}_100"]["rewards"],
                    results[f"{env}_{algo}_500"]["rewards"],
                    results[f"{env}_{algo}_1000"]["rewards"],
                ]
            elif envs_group == "adroit":
                results_list = [
                    results[f"{env}_{algo}_10"]["rewards"],
                    results[f"{env}_{algo}_50"]["rewards"],
                    results[f"{env}_{algo}_100"]["rewards"],
                    results[f"{env}_{algo}_500"]["rewards"],
                    results[f"{env}_{algo}_1000"]["rewards"],
                    results[f"{env}_{algo}_5000"]["rewards"],
                ]
            ax[i].boxplot(
                results_list,
                positions=np.arange(len(data_size_list)) + j * (box_width + 0.03),
                widths=box_width,
                patch_artist=True,
                boxprops=dict(facecolor=colors[j], color=colors[j]),
                whiskerprops=dict(color=colors[j]),
                capprops=dict(color=colors[j]),
                medianprops=dict(color="ghostwhite", alpha=0.5),
                flierprops=dict(marker="o", markersize=1, color=colors[j], alpha=0.3),
            )
        ax[i].set_yticks(np.arange(0, 201, 50))
        ax[i].spines[["right", "left", "top", "bottom"]].set_visible(False)
        ax[i].set_facecolor("ghostwhite")
        ax[i].set_ylim(0, 210)
        ax[i].set_title(f"{env.capitalize()}", size=16)
        ax[i].tick_params(axis="both", which="major", labelsize=12)
        ax[i].set_xticks(np.arange(len(data_size_list)) + box_width)
        ax[i].set_xticklabels(data_size_list)
        ax[i].set_xlabel("Data Size", size=16)
        if i == 0:
            ax[i].set_ylabel("Normalized Return", size=16)
        if i == len(envs) - 1:
            handles = [plt.Rectangle((0, 0), 1, 1, color=colors[j]) for j in range(3)]
            labels = [plot_label_map[algo] for algo in algos]
            fig.legend(
                handles,
                labels,
                loc="lower center",
                ncols=3,
                fontsize=12,
                bbox_to_anchor=(0.5, -0.01),
            )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    fig.savefig(f"plots/data_size_{envs_group}_boxplot.png")
    fig.savefig(f"plots/data_size_{envs_group}_boxplot.pdf")


def bc_sr_barplot(results: Dict):
    os.makedirs("plots", exist_ok=True)
    envs = ["pickcube", "stackcube", "turnfaucet"]
    colors = ["teal", "indigo"]
    algos = ["bc", "sparseql"]
    fig, ax = plt.subplots(1, len(envs), figsize=(8, 4))
    for i, env in enumerate(envs):
        ax[i].bar(
            np.arange(2),
            [
                results[f"{env}_bc_default"]["mean"],
                results[f"{env}_sparseql_default"]["mean"],
            ],
            color=colors,
            tick_label=["BC", "SR-Reward"],
        )
        ax[i].errorbar(
            np.arange(2),
            [
                results[f"{env}_bc_default"]["mean"],
                results[f"{env}_sparseql_default"]["mean"],
            ],
            yerr=[
                results[f"{env}_bc_default"]["std"],
                results[f"{env}_sparseql_default"]["std"],
            ],
            fmt="none",
            ecolor="black",
            capsize=5,
            alpha=0.3,
        )
        ax[i].set_title(env.capitalize(), size=16)
        ax[i].set_yticks(np.arange(0, 151, 50))
        ax[i].spines[["right", "left", "top", "bottom"]].set_visible(False)
        ax[i].set_facecolor("ghostwhite")
        ax[i].set_ylim(0, 150)
        ax[i].tick_params(axis="both", which="major", labelsize=12)
        if i == 0:
            ax[i].set_ylabel("Normalized Return", size=16)
    fig.tight_layout()
    fig.savefig("plots/bc_sr_barplot.png")
    fig.savefig("plots/bc_sr_barplot.pdf")


def negative_sampling_barplot(results: Dict):
    """
    Results for the negative sampling experiments for Pickcube and Stackcube.
    """
    os.makedirs("plots", exist_ok=True)
    envs = ["pickcube", "stackcube", "turnfaucet"]
    colors = ["orange", "skyblue", "indigo"]
    algos = ["No N.S.", "N.S. (VINS)", "N.S. (ours)"]
    fig, ax = plt.subplots(1, len(envs), figsize=(12, 4))
    for i, env in enumerate(envs):
        ax[i].bar(
            np.arange(len(algos)),
            [
                results[f"{env}_sparseql_no-neg-sampling"]["mean"],
                results[f"{env}_sparseql_orig-neg-sampling"]["mean"],
                results[f"{env}_sparseql_default"]["mean"],
            ],
            color=colors,
            tick_label=algos,
        )
        ax[i].errorbar(
            np.arange(len(algos)),
            [
                results[f"{env}_sparseql_no-neg-sampling"]["mean"],
                results[f"{env}_sparseql_orig-neg-sampling"]["mean"],
                results[f"{env}_sparseql_default"]["mean"],
            ],
            yerr=[
                results[f"{env}_sparseql_no-neg-sampling"]["std"],
                results[f"{env}_sparseql_orig-neg-sampling"]["std"],
                results[f"{env}_sparseql_default"]["std"],
            ],
            fmt="none",
            ecolor="black",
            capsize=5,
            alpha=0.3,
        )
        ax[i].set_title(env.capitalize(), size=16)
        ax[i].set_yticks(np.arange(0, 151, 50))
        ax[i].spines[["right", "left", "top", "bottom"]].set_visible(False)
        ax[i].set_facecolor("ghostwhite")
        ax[i].set_ylim(0, 150)
        ax[i].tick_params(axis="both", which="major", labelsize=12)
        if i == 0:
            ax[i].set_ylabel("Normalized Return", size=16)
    fig.tight_layout()
    fig.savefig("plots/negative_sampling_barplot.png")
    fig.savefig("plots/negative_sampling_barplot.pdf")


def variance_ranking_barplot(results: Dict):
    """
    plot the line plot of coefficient of variation per dataset for each environment as well as
    coefficient of variation per environment for each evaluation result.
    """
    dataset_cv_map = {
        "walker2d": 0.69,
        "pen": 0.532,
        "hammer": 0.512,
        "ant": 0.47,
        "relocate": 0.1927,
        "door": 0.1448,
        "hopper": 0.119,
        "halfcheetah": 0.0867,
    }
    environment_cv_map = {}
    for env in dataset_cv_map.keys():
        environment_cv_map[env] = (
            results[f"{env}_sparseql_default"]["std"]
            / results[f"{env}_sparseql_default"]["mean"]
        )
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    bar_width = 0.2
    index = np.arange(len(dataset_cv_map))
    dataset_bars = ax.bar(
        index,
        list(dataset_cv_map.values()),
        bar_width,
        color="darkorange",
        alpha=0.8,
        label="Dataset",
    )
    environment_bars = ax.bar(
        index + bar_width,
        list(environment_cv_map.values()),
        bar_width,
        color="indigo",
        alpha=0.8,
        label="SR-Reward",
    )

    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(
        [key.capitalize() for key in dataset_cv_map.keys()],
        rotation=45,
        ha="right",
        fontsize=12,
    )

    ax.set_ylabel("Coefficient of Variation", fontsize=14)
    ax.legend(fontsize=12)
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_facecolor("ghostwhite")
    fig.tight_layout()
    fig.savefig("plots/variance_ranking_barplot.png")
    fig.savefig("plots/variance_ranking_barplot.pdf")


def single_env_barplot(results: Dict, env: str):
    """
    Plot the results for a single environment.
    """
    os.makedirs("plots", exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    algos = ["sparseql"]
    colors = ["indigo"]
    plot_label_map = {
        "sparseql": "SR-Reward",
    }
    bar_width = 0.1
    for j, algo in enumerate(algos):
        results_list = [
            results[f"{env}_{algo}_default"]["mean"],
        ]
        results_list_std = [
            results[f"{env}_{algo}_default"]["std"],
        ]
        ax.bar(
            np.arange(len(results_list)) + j * bar_width,
            results_list,
            bar_width,
            color=colors[j],
            label=plot_label_map[algo],
        )
        ax.errorbar(
            np.arange(len(results_list)) + j * bar_width,
            results_list,
            yerr=np.array(results_list_std) * 1.0,
            fmt="none",
            ecolor="black",
            capsize=5,
            alpha=0.3,
        )
    ax.set_yticks(np.arange(0, 201, 50))
    ax.spines[["right", "left", "top", "bottom"]].set_visible(False)
    ax.set_facecolor("ghostwhite")
    ax.set_ylim(0, 210)
    ax.set_title(f"{env.capitalize()}", size=16)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.set_xticks(np.arange(len(results_list)))
    ax.set_xticklabels(algos)
    ax.set_ylabel("Normalized Return", size=16)
    fig.tight_layout()
    fig.savefig(f"plots/{env}_barplot.png")
    fig.savefig(f"plots/{env}_barplot.pdf")


def return_histogram_plot(results: Dict):
    ncols = 5
    nrows = len(results) // ncols
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(100, 200))
    for i in range(nrows):
        for j in range(ncols):
            index = i * ncols + j
            key = list(results.keys())[index]
            ax[i][j].hist(results[key]["unnormalized_rewards"], bins=100)
            ax[i][j].set_title(key, size=50)
            ax[i][j].set_xlabel("Return")
            ax[i][j].set_ylabel("Frequency")
            ax[i][j].spines[["right", "top"]].set_visible(False)
            ax[i][j].set_facecolor("ghostwhite")

    fig.tight_layout()
    fig.savefig("plots/return_histogram.png")
    fig.savefig("plots/return_histogram.pdf")


def get_data_quality_results(envs_group: str):
    """
    Load the data size results.
    """
    results = {}
    if envs_group == "mujoco":
        envs = ["Ant", "HalfCheetah", "Hopper", "Walker2d"]
        quality_list = ["medium", "medium_expert", "expert"]
    elif envs_group == "adroit":
        envs = ["Door", "Hammer", "Pen", "Relocate"]
        quality_list = ["human", "expert"]
    experiments = ["data-quality-", "true-reward__data-quality-"]
    algos = ["bc", "sparseql"]
    for env in envs:
        config = recompose_config(overrides=[f"environment={env.lower()}"]).environment
        min_score = config.d4rl.min_score
        max_score = config.d4rl.max_score
        for algo in algos:
            for experiment in experiments:
                for quality in quality_list:
                    if algo == "bc" and experiment.startswith("true-reward"):
                        continue
                    if quality == "expert":
                        experiment_folder = (
                            "default"
                            if experiment == "data-quality-"
                            else "default-true-reward"
                        )
                        filenames = glob.glob(
                            str(
                                Path("models")
                                / algo
                                / env
                                / experiment_folder
                                / f"**/test_evaluations.npz"
                            ),
                            recursive=True,
                        )
                    else:
                        filenames = glob.glob(
                            str(
                                Path("models")
                                / algo
                                / env
                                / f"{experiment}{quality}"
                                / f"**/test_evaluations.npz"
                            ),
                            recursive=True,
                        )
                    if experiment.startswith("true-reward"):
                        key = f"{env.lower()}_{algo}_true_reward_{quality}"
                    else:
                        key = f"{env.lower()}_{algo}_{quality}"
                    (
                        median,
                        interquartile_range,
                        mean,
                        std,
                        rewards,
                        unnormalized_median,
                        unnormalized_interquartile_range,
                        unnormalized_mean,
                        unnormalized_std,
                        unnormalized_rewards,
                    ) = get_normalized_mean_std(filenames, max_score, min_score)
                    results[key] = {
                        "median": median,
                        "interquartile_range": interquartile_range,
                        "mean": mean,
                        "std": std,
                        "rewards": rewards,
                        "unnormalized_median": unnormalized_median,
                        "unnormalized_interquantile_range": unnormalized_interquartile_range,
                        "unnormalized_mean": unnormalized_mean,
                        "unnormalized_std": unnormalized_std,
                        "unnormalized_rewards": unnormalized_rewards,
                    }
    return results


def get_data_size_results(envs_group: str):
    """
    Load the data size results.
    """
    results = {}
    if envs_group == "mujoco":
        envs = ["Ant", "HalfCheetah", "Hopper", "Walker2d"]
        sizes = [10, 50, 100, 500, 1000]
        default_size = 1000
    elif envs_group == "adroit":
        envs = ["Door", "Hammer", "Pen", "Relocate"]
        sizes = [10, 50, 100, 500, 1000, 5000]
        default_size = 5000
    experiments = ["data-size-", "true-reward__data-size-"]
    algos = ["bc", "sparseql"]
    for env in envs:
        config = recompose_config(overrides=[f"environment={env.lower()}"]).environment
        min_score = config.d4rl.min_score
        max_score = config.d4rl.max_score
        for algo in algos:
            for experiment in experiments:
                for size in sizes:
                    if algo == "bc" and experiment.startswith("true-reward"):
                        continue
                    if size == default_size:
                        experiment_folder = (
                            "default"
                            if experiment == "data-size-"
                            else "default-true-reward"
                        )
                        filenames = glob.glob(
                            str(
                                Path("models")
                                / algo
                                / env
                                / experiment_folder
                                / f"**/test_evaluations.npz"
                            ),
                            recursive=True,
                        )
                    else:
                        filenames = glob.glob(
                            str(
                                Path("models")
                                / algo
                                / env
                                / f"{experiment}{size}"
                                / f"**/test_evaluations.npz"
                            ),
                            recursive=True,
                        )
                    if experiment.startswith("true-reward"):
                        key = f"{env.lower()}_{algo}_true_reward_{size}"
                    else:
                        key = f"{env.lower()}_{algo}_{size}"

                    (
                        median,
                        interquartile_range,
                        mean,
                        std,
                        rewards,
                        unnormalized_median,
                        unnormalized_interquartile_range,
                        unnormalized_mean,
                        unnormalized_std,
                        unnormalized_rewards,
                    ) = get_normalized_mean_std(filenames, max_score, min_score)
                    results[key] = {
                        "median": median,
                        "interquartile_range": interquartile_range,
                        "mean": mean,
                        "std": std,
                        "rewards": rewards,
                        "unnormalized_median": unnormalized_median,
                        "unnormalized_interquantile_range": unnormalized_interquartile_range,
                        "unnormalized_mean": unnormalized_mean,
                        "unnormalized_std": unnormalized_std,
                        "unnormalized_rewards": unnormalized_rewards,
                    }
    return results


def get_baseline_results():
    """
    Load the baseline (defaults)results.
    """
    results = {}

    envs = [
        "Ant",
        "HalfCheetah",
        "Hopper",
        "Walker2d",
        "Door",
        "Hammer",
        "Pen",
        "Relocate",
        "PickCube",
        "StackCube",
        "TurnFaucet",
        "NavMap",
    ]
    experiments = ["default", "default-true-reward"]
    algos = ["bc", "fdvl", "sparseql"]
    for env in envs:
        config = recompose_config(overrides=[f"environment={env.lower()}"]).environment
        min_score = config.d4rl.min_score
        max_score = config.d4rl.max_score
        for algo in algos:
            for experiment in experiments:
                if algo == "bc" and experiment == "default-true-reward":
                    continue
                if (
                    env in ["StackCube", "PickCube", "TurnFaucet"]
                    and experiment == "default-true-reward"
                ):
                    continue
                filenames = glob.glob(
                    str(
                        Path("models")
                        / algo
                        / env
                        / experiment
                        / f"**/test_evaluations.npz"
                    ),
                    recursive=True,
                )
                key = f"{env.lower()}_{algo}_{experiment}"
                (
                    median,
                    interquartile_range,
                    mean,
                    std,
                    rewards,
                    unnormalized_median,
                    unnormalized_interquartile_range,
                    unnormalized_mean,
                    unnormalized_std,
                    unnormalized_rewards,
                ) = get_normalized_mean_std(filenames, max_score, min_score)
                results[key] = {
                    "median": median,
                    "interquartile_range": interquartile_range,
                    "mean": mean,
                    "std": std,
                    "rewards": rewards,
                    "unnormalized_median": unnormalized_median,
                    "unnormalized_interquantile_range": unnormalized_interquartile_range,
                    "unnormalized_mean": unnormalized_mean,
                    "unnormalized_std": unnormalized_std,
                    "unnormalized_rewards": unnormalized_rewards,
                }
    return results


def get_baseline_validation_results():
    """
    Load the baseline (defaults) training results.
    These are the validation results that were captured during training.
    We would like to combine the results at every validation step over all available seeds
    and report the mean and std.
    The validation results are in the evaluations.npz file.
    """
    results = {}

    envs = [
        "Ant",
        "HalfCheetah",
        "Hopper",
        "Walker2d",
        "Door",
        "Hammer",
        "Pen",
        "Relocate",
        "PickCube",
        "StackCube",
        "TurnFaucet",
    ]
    experiments = ["default", "default-true-reward"]
    algos = ["bc", "fdvl", "sparseql"]
    for env in envs:
        config = recompose_config(overrides=[f"environment={env.lower()}"]).environment
        min_score = config.d4rl.min_score
        max_score = config.d4rl.max_score
        for algo in algos:
            for experiment in experiments:
                if algo == "bc" and experiment == "default-true-reward":
                    continue
                if (
                    env in ["StackCube", "PickCube", "TurnFaucet"]
                    and experiment == "default-true-reward"
                ):
                    continue
                filenames = glob.glob(
                    str(
                        Path("models") / algo / env / experiment / f"**/evaluations.npz"
                    ),
                    recursive=True,
                )
                key = f"{env.lower()}_{algo}_{experiment}"
                (
                    median,
                    interquartile_range,
                    mean,
                    std,
                    rewards,
                    unnormalized_median,
                    unnormalized_interquartile_range,
                    unnormalized_mean,
                    unnormalized_std,
                    unnormalized_rewards,
                ) = get_validation_normalized_mean_std(filenames, max_score, min_score)
                results[key] = {
                    "median": median,
                    "interquartile_range": interquartile_range,
                    "mean": mean,
                    "std": std,
                    "rewards": rewards,
                    "unnormalized_median": unnormalized_median,
                    "unnormalized_interquantile_range": unnormalized_interquartile_range,
                    "unnormalized_mean": unnormalized_mean,
                    "unnormalized_std": unnormalized_std,
                    "unnormalized_rewards": unnormalized_rewards,
                }
    return results


def get_bc_sr_results():
    """
    To compare BC (defaults) and SR-Reward (defaults) results for stackcube and pickcube environments.
    """
    results = {}

    envs = ["StackCube", "PickCube", "TurnFaucet"]
    experiments = ["default"]
    algos = ["bc", "sparseql"]
    for env in envs:
        config = recompose_config(overrides=[f"environment={env.lower()}"]).environment
        min_score = config.d4rl.min_score
        max_score = config.d4rl.max_score
        for algo in algos:
            for experiment in experiments:
                filenames = glob.glob(
                    str(
                        Path("models")
                        / algo
                        / env
                        / experiment
                        / f"**/test_evaluations.npz"
                    ),
                    recursive=True,
                )
                key = f"{env.lower()}_{algo}_{experiment}"
                (
                    median,
                    interquartile_range,
                    mean,
                    std,
                    rewards,
                    unnormalized_median,
                    unnormalized_interquartile_range,
                    unnormalized_mean,
                    unnormalized_std,
                    unnormalized_rewards,
                ) = get_normalized_mean_std(filenames, max_score, min_score)
                results[key] = {
                    "mean": mean,
                    "std": std,
                    "rewards": rewards,
                    "unnormalized_mean": unnormalized_mean,
                    "unnormalized_std": unnormalized_std,
                    "unnormalized_rewards": unnormalized_rewards,
                }
    return results


def get_negative_sampling_results():
    """
    Get results for defaults and no_neg_sampling and orig-neg-sampling experiments for pickcube and stackcube environments.
    """
    results = {}

    envs = ["StackCube", "PickCube", "TurnFaucet"]
    experiments = ["default", "no-neg-sampling", "orig-neg-sampling"]
    algos = ["sparseql"]
    for env in envs:
        config = recompose_config(overrides=[f"environment={env.lower()}"]).environment
        min_score = config.d4rl.min_score
        max_score = config.d4rl.max_score
        for algo in algos:
            for experiment in experiments:
                filenames = glob.glob(
                    str(
                        Path("models")
                        / algo
                        / env
                        / experiment
                        / f"**/test_evaluations.npz"
                    ),
                    recursive=True,
                )
                key = f"{env.lower()}_{algo}_{experiment}"
                (
                    median,
                    interquartile_range,
                    mean,
                    std,
                    rewards,
                    unnormalized_median,
                    unnormalized_interquartile_range,
                    unnormalized_mean,
                    unnormalized_std,
                    unnormalized_rewards,
                ) = get_normalized_mean_std(filenames, max_score, min_score)
                results[key] = {
                    "mean": mean,
                    "std": std,
                    "rewards": rewards,
                    "unnormalized_mean": unnormalized_mean,
                    "unnormalized_std": unnormalized_std,
                    "unnormalized_rewards": unnormalized_rewards,
                }
    return results


def get_negative_sampling_sensitivity_results():
    """
    Get results for negative sampling sensitivity analysis where we
    vary the noise and sigma hyperpaprameters for negative sampling for stack cube environment.
    """

    results = {}
    envs = ["StackCube"]
    noise_levels = [0.01, 0.03, 0.05, 0.1, 0.5, 1.0, 2.0]
    sigma_levels = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
    experiments = [
        f"default-noise_{noise}-sigma_{sigma}"
        for noise in noise_levels
        for sigma in sigma_levels
    ]
    algos = ["sparseql"]
    for env in envs:
        config = recompose_config(overrides=[f"environment={env.lower()}"]).environment
        min_score = config.d4rl.min_score
        max_score = config.d4rl.max_score
        for algo in algos:
            for experiment in experiments:
                filenames = glob.glob(
                    str(
                        Path("models")
                        / algo
                        / env
                        / experiment
                        / f"**/test_evaluations.npz"
                    ),
                    recursive=True,
                )
                key = f"{env.lower()}_{algo}_{experiment}"
                (
                    median,
                    interquartile_range,
                    mean,
                    std,
                    rewards,
                    unnormalized_median,
                    unnormalized_interquartile_range,
                    unnormalized_mean,
                    unnormalized_std,
                    unnormalized_rewards,
                ) = get_normalized_mean_std(filenames, max_score, min_score)
                results[key] = {
                    "mean": mean,
                    "std": std,
                    "rewards": rewards,
                    "unnormalized_mean": unnormalized_mean,
                    "unnormalized_std": unnormalized_std,
                    "unnormalized_rewards": unnormalized_rewards,
                }
    return results


def get_offline_online_results():
    """
    Reviewer 2 would like to see the results for the offline and online experiments.
    We use TD3 on Half cheetah, once with the true reward and once with the SR-Reward.
    We expect the SR-Reward to not be as good as the true reward, because of
    estimation error for OOD during online interactions.
    """
    results = {}
    envs = ["HalfCheetah"]
    experiments = ["default", "no-neg-sampling", "default-true-reward"]
    algos = ["td3"]
    for env in envs:
        config = recompose_config(overrides=[f"environment={env.lower()}"]).environment
        min_score = config.d4rl.min_score
        max_score = config.d4rl.max_score
        for algo in algos:
            for experiment in experiments:
                filenames = glob.glob(
                    str(
                        Path("models")
                        / algo
                        / env
                        / experiment
                        / f"**/test_evaluations.npz"
                    ),
                    recursive=True,
                )
                key = f"{env.lower()}_{algo}_{experiment}"
                (
                    median,
                    interquartile_range,
                    mean,
                    std,
                    rewards,
                    unnormalized_median,
                    unnormalized_interquartile_range,
                    unnormalized_mean,
                    unnormalized_std,
                    unnormalized_rewards,
                ) = get_normalized_mean_std(filenames, max_score, min_score)
                results[key] = {
                    "mean": mean,
                    "std": std,
                    "rewards": rewards,
                    "unnormalized_mean": unnormalized_mean,
                    "unnormalized_std": unnormalized_std,
                    "unnormalized_rewards": unnormalized_rewards,
                }
    return results


def plot_reward_test(dir1: Path, dir2: Path):
    """
    Plots the reward vs noise levels for the results in test_reward.npz files in two directories:
    one with and one without negative sampling.
    """

    def load_results(directory: Path):
        npz_file = directory / "test_reward.npz"
        data = np.load(npz_file)
        return data["rewards"], data["noises"]

    rewards1, noises1 = load_results(dir1)
    rewards2, noises2 = load_results(dir2)

    # Determine labels based on dir names
    label1 = (
        "Without Negative Sampling"
        if "no-neg-sampling" in dir1.parent.parent.name
        else "With Negative Sampling"
    )
    label2 = (
        "Without Negative Sampling"
        if "no-neg-sampling" in dir2.parent.parent.name
        else "With Negative Sampling"
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor("ghostwhite")
    ax.spines[["right", "left", "top", "bottom"]].set_visible(False)
    ax.set_yticks(np.arange(0, 210, 50))
    ax.tick_params(axis="both", which="major", labelsize=18)

    ax.plot(
        noises1,
        rewards1,
        marker="o",
        markersize=10,
        label=label1,
        alpha=0.8,
        linewidth=3,
        color="orange",
    )
    ax.plot(
        noises2,
        rewards2,
        marker="o",
        markersize=10,
        label=label2,
        alpha=0.8,
        linewidth=3,
        color="indigo",
    )

    ax.set_xlabel("Noise Fraction", fontsize=20)
    ax.set_ylabel("Return", fontsize=20)
    ax.legend(fontsize=18)
    fig.tight_layout()
    fig.savefig("./plots/reward_vs_noise.png")
    fig.savefig("./plots/reward_vs_noise.pdf")


def plot_instant_reward(directory: Path):
    """
    Loads the instant_rewards_on_expert_traj list from the specified directory and plots it
    """

    def load_instant_rewards(directory: Path):
        npz_file = directory / "instant_rewards_on_expert_traj.npz"
        data = np.load(npz_file, allow_pickle=True)
        return data["instant_rewards"].tolist()

    # Load instant rewards
    instant_rewards = load_instant_rewards(directory)

    # ploting the first traj. for the time being
    first_reward_list = instant_rewards[2]
    max_len = max(len(reward) for reward in instant_rewards)

    # pad to max len by repeating the last element
    for i, reward in enumerate(instant_rewards):
        if len(reward) < max_len:
            last_value = reward[-1]
            instant_rewards[i].extend([last_value] * (max_len - len(reward)))

    # Plot the first rewards list
    plt.figure(figsize=(10, 6))
    plt.plot(
        first_reward_list,
        marker="o",
        label="Instant Reward Throughout Expert Trajectory for a single traj",
    )
    plt.xlabel("trajectory time step")
    plt.ylabel("SR-Reward")
    plt.legend()
    plt.savefig("./plots/instant_sr_rewards_single_expert_traj.png")
    plt.show()

    rewards_array = np.array(instant_rewards)
    mean_rewards = np.mean(rewards_array, axis=0)
    std_rewards = np.std(rewards_array, axis=0)

    # Plot the mean with std across all expert trajs
    plt.figure(figsize=(10, 6))
    plt.plot(mean_rewards, label="Mean Reward", color="blue")
    plt.fill_between(
        range(max_len),
        mean_rewards - std_rewards,
        mean_rewards + std_rewards,
        color="blue",
        alpha=0.2,
        label="1 std",
    )

    plt.xlabel("trajectory time step")
    plt.ylabel("SR-Reward")
    plt.legend()
    plt.savefig("./plots/instant_sr_rewards_all_expert_traj.png")
    plt.show()

    # calculate and plot the gradient of the reward arrays
    gradients = np.gradient(rewards_array, axis=1)
    mean_gradients = np.mean(gradients, axis=0)
    std_gradients = np.std(gradients, axis=0)

    # Plot the mean gradient
    plt.figure(figsize=(10, 6))
    plt.plot(mean_gradients, label="Mean Reward Gradient", color="green")
    plt.fill_between(
        range(max_len),
        mean_gradients - std_gradients,
        mean_gradients + std_gradients,
        color="green",
        alpha=0.2,
        label="1 std",
    )

    plt.xlabel("Timestep")
    plt.ylabel("Reward Gradient")
    plt.legend()
    plt.title("Mean Reward Gradient")
    plt.savefig("./plots/reward_gradient_with_std.png")
    plt.show()


def print_t_test_results(results: Dict):
    print("\n=== T-Test Results ===")
    best_algos = {}
    for env in results:
        env_name = env.split("_", 1)[0]
        algo_name = env.split("_", 1)[1]
        if env_name not in best_algos:
            best_algos[env_name] = (algo_name, results[env]["mean"])
        else:
            if results[env]["mean"] > best_algos[env_name][1]:
                best_algos[env_name] = (algo_name, results[env]["mean"])
    for env in results:
        env_name = env.split("_", 1)[0]
        algo_name = env.split("_", 1)[1]
        best_algo, best_mean = best_algos[env_name]
        if algo_name != best_algo:
            t_stat, p_value = ttest_ind(
                results[f"{env_name}_{best_algo}"]["rewards"],
                results[env]["rewards"],
                equal_var=False,
            )
            if p_value < 0.05:
                print(
                    f"{env_name}: {best_algo} (mean: {best_mean:.3f}, std: {results[f'{env_name}_{best_algo}']['std']:.3f}) vs {algo_name} (mean: {results[env]['mean']:.3f}, std: {results[env]['std']:.3f}) --- Significant difference (p-value: {p_value:.5f}) <---"
                )
            else:
                print(
                    f"{env_name}: {best_algo} (mean: {best_mean:.3f}, std: {results[f'{env_name}_{best_algo}']['std']:.3f}) vs {algo_name} (mean: {results[env]['mean']:.3f}, std: {results[env]['std']:.3f}) --- No significant difference (p-value: {p_value:.5f})"
                )


def print_man_whitney_u_test_results(results: Dict):
    print("\n=== Mann-Whitney U Test Results ===")
    best_algos = {}
    for env in results:
        env_name = env.split("_", 1)[0]
        algo_name = env.split("_", 1)[1]
        if env_name not in best_algos:
            best_algos[env_name] = (algo_name, results[env]["median"])
        else:
            if results[env]["median"] > best_algos[env_name][1]:
                best_algos[env_name] = (algo_name, results[env]["median"])
    for env in results:
        env_name = env.split("_", 1)[0]
        algo_name = env.split("_", 1)[1]
        best_algo, best_mean = best_algos[env_name]
        if algo_name != best_algo:
            if len(results[env]["rewards"]) == 0:
                print(f"{env} has no rewards")
                continue
            u_stat, p_value = mannwhitneyu(
                results[f"{env_name}_{best_algo}"]["rewards"],
                results[env]["rewards"],
                alternative="two-sided",
            )
            if p_value < 0.05:
                print(
                    f"{env_name}: {best_algo} (median: {best_mean:.3f}, iq-range: {results[f'{env_name}_{best_algo}']['interquartile_range']:.3f}) vs {algo_name} (median: {results[env]['median']:.3f}, iq-range: {results[env]['interquartile_range']:.3f}) --- Significant difference (p-value: {p_value:.5f}) <---"
                )
            else:
                print(
                    f"{env_name}: {best_algo} (median: {best_mean:.3f}, iq-range: {results[f'{env_name}_{best_algo}']['interquartile_range']:.3f}) vs {algo_name} (median: {results[env]['median']:.3f}, iq-range: {results[env]['interquartile_range']:.3f}) --- No significant difference (p-value: {p_value:.5f})"
                )


def negative_sampling_sensitivity_plot(results: Dict):
    """
    Scatter plot with noise in the x-axis and sigma in the y-axis.
    The color of the points is determined by the mean reward (return)
    This is only used for StackCube environment.
    """
    GRID = True
    os.makedirs("plots", exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor("ghostwhite")
    ax.spines[["right", "left", "top", "bottom"]].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=18)
    noise_levels = [0.01, 0.03, 0.05, 0.1, 0.5]  # , 1.0, 2.0]
    sigma_levels = [0.1, 0.3, 0.5, 1.0]  # , 2.0, 5.0]
    if GRID:
        ax.set_xticks(np.arange(len(noise_levels)))
        ax.set_xticklabels(noise_levels)
        ax.set_yticks(np.arange(len(sigma_levels)))
        ax.set_yticklabels(sigma_levels)

    for i in range(len(noise_levels)):
        for j in range(len(sigma_levels)):
            key = f"stackcube_sparseql_default-noise_{noise_levels[i]}-sigma_{sigma_levels[j]}"
            if key in results:
                mean_reward = results[key]["mean"] + 1e-1
                if np.isnan(mean_reward):
                    continue
                if noise_levels[i] == 0.03 and sigma_levels[j] == 0.3:
                    color = "red"
                else:
                    color = "indigo"
                if GRID:
                    ax.scatter(
                        i,
                        j,
                        s=mean_reward * 14,
                        alpha=0.7,
                        c=color,
                    )
                    ax.annotate(
                        f"{mean_reward:.2f}",
                        (i, j),
                        textcoords="offset points",
                        xytext=(0, 20),
                        ha="center",
                        fontsize=13,
                        color="black",
                        alpha=0.7,
                    )
                else:
                    ax.scatter(
                        noise_levels[i],
                        sigma_levels[j],
                        s=mean_reward * 10,
                        alpha=0.7,
                        c=color,
                    )
                    ax.annotate(
                        f"{mean_reward:.2f}",
                        (noise_levels[i], sigma_levels[j]),
                        textcoords="offset points",
                        xytext=(0, 20),
                        ha="center",
                        fontsize=13,
                        color="black",
                        alpha=0.7,
                    )

    ax.set_xlabel(r"$\beta$", fontsize=20)
    ax.set_ylabel(r"$\sigma$", fontsize=20)
    fig.tight_layout()
    fig.savefig("plots/negative_sampling_sensitivity_plot.png")
    fig.savefig("plots/negative_sampling_sensitivity_plot.pdf")


def offline_online_plot(results: Dict):
    """
    Plot the offline and online results for the HalfCheetah environment using TD3.
    Online means that we are using the reward from the environment as standard RL.
    Offline means that we are using the reward from SR-Reward and runnign RL with that reward.
    """
    os.makedirs("plots", exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_facecolor("ghostwhite")
    ax.spines[["right", "left", "top", "bottom"]].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=18)

    bar_width = 0.05
    results_list = [
        results["halfcheetah_td3_default-true-reward"]["mean"],
        results["halfcheetah_td3_default"]["mean"],
        results["halfcheetah_td3_no-neg-sampling"]["mean"],
    ]
    results_list_std = [
        results["halfcheetah_td3_default-true-reward"]["std"],
        results["halfcheetah_td3_default"]["std"],
        results["halfcheetah_td3_no-neg-sampling"]["std"],
    ]
    x_locations = np.array([0, 0.1, 0.2])
    ax.bar(
        x_locations,
        results_list,
        bar_width,
        color="indigo",
        label="TD3",
    )
    ax.errorbar(
        x_locations,
        results_list,
        yerr=np.array(results_list_std) * 1.0,
        fmt="none",
        ecolor="black",
        capsize=5,
        alpha=0.3,
    )

    ax.set_yticks(np.arange(0, 101, 50))
    ax.spines[["right", "left", "top", "bottom"]].set_visible(False)
    ax.set_facecolor("ghostwhite")
    ax.set_ylim(0, 110)
    ax.set_title(f"HalfCheetah | Offline-to-Online Reward", size=14)

    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.set_xticks(x_locations)

    ax.set_xticklabels(
        [
            "Online\n(True Reward)",
            "Online\n(SR-Reward + Neg Sampling)",
            "Online\n(SR-Reward)",
        ],
        rotation=0,
        ha="center",
        fontsize=12,
    )
    ax.set_ylabel("Normalized Return", size=16)
    ax.legend(fontsize=12)
    fig.tight_layout()

    fig.savefig("plots/halfcheetah_td3_offline_online.png")
    fig.savefig("plots/halfcheetah_td3_offline_online.pdf")
