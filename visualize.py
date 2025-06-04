import logging
from visualize_utils import (
    get_data_quality_results,
    data_quality_lineplot,
    data_quality_barplot,
    data_quality_boxplot,
    get_data_size_results,
    data_size_lineplot,
    data_size_barplot,
    data_size_boxplot,
    get_baseline_results,
    get_baseline_validation_results,
    get_negative_sampling_results,
    get_negative_sampling_sensitivity_results,
    get_offline_online_results,
    get_bc_sr_results,
    baseline_plot,
    baseline_validation_plot,
    bc_sr_barplot,
    negative_sampling_barplot,
    print_t_test_results,
    print_man_whitney_u_test_results,
    variance_ranking_barplot,
    single_env_barplot,
    return_histogram_plot,
    negative_sampling_sensitivity_plot,
    offline_online_plot,
)

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

if __name__ == "__main__":
    logging.info("Visualizing the evaluation results")

    # results data
    data_quality_results_mujoco = get_data_quality_results(envs_group="mujoco")
    data_size_results_mujoco = get_data_size_results(envs_group="mujoco")
    data_size_results_adroit = get_data_size_results(envs_group="adroit")
    bc_sr_results = get_bc_sr_results()
    negative_sampling_results = get_negative_sampling_results()
    negative_sampling_sensitivity_results = get_negative_sampling_sensitivity_results()
    offline_online_results = get_offline_online_results()
    baseline_results = get_baseline_results()
    baseline_validation_results = get_baseline_validation_results()

    std_fraction = 1.0
    # bar plots
    data_quality_barplot(
        data_quality_results_mujoco, envs_group="mujoco", std_fraction=std_fraction
    )
    data_size_barplot(
        data_size_results_mujoco, envs_group="mujoco", std_fraction=std_fraction
    )
    data_size_barplot(
        data_size_results_adroit, envs_group="adroit", std_fraction=std_fraction
    )

    # line plots
    data_quality_lineplot(
        results=data_quality_results_mujoco,
        envs_group="mujoco",
        std_fraction=std_fraction,
    )
    data_size_lineplot(
        results=data_size_results_mujoco, envs_group="mujoco", std_fraction=std_fraction
    )
    data_size_lineplot(
        results=data_size_results_adroit, envs_group="adroit", std_fraction=std_fraction
    )

    # box plots
    data_quality_boxplot(results=data_quality_results_mujoco, envs_group="mujoco")
    data_size_boxplot(results=data_size_results_mujoco, envs_group="mujoco")
    data_size_boxplot(results=data_size_results_adroit, envs_group="adroit")

    # baseline plot
    baseline_plot(baseline_results)

    # bc_sr plot
    bc_sr_barplot(bc_sr_results)
    negative_sampling_barplot(negative_sampling_results)

    print_t_test_results(baseline_results)
    print_man_whitney_u_test_results(baseline_results)
    variance_ranking_barplot(baseline_results)

    single_env_barplot(baseline_results, env="navmap")

    return_histogram_plot(baseline_results)

    # negative sampling sensitivity plot
    negative_sampling_sensitivity_plot(negative_sampling_sensitivity_results)

    # offline online plot
    offline_online_plot(offline_online_results)

    # baseline validation plot
    baseline_validation_plot(baseline_validation_results)
