import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from experiments import labels
import numpy as np
from experiments.policy import can_convert_to_float


def intervention_vs_distance(experiment, ci_factor=1.96, save_to_file=False):

    for model_name, model_dict in experiment.distance_results.items():
        for algorithm, algorithm_dict in model_dict.items():

            shapley_methods = list(algorithm_dict.keys())
            distance_metrics = list(algorithm_dict[shapley_methods[0]].keys())

            fig, axes = plt.subplots(
                1, len(distance_metrics), figsize=(5 * len(distance_metrics), 4)
            )
            for i, distance_metric in enumerate(distance_metrics):

                for shapley_method in shapley_methods:
                    if (
                        shapley_method == "optimality"
                        and distance_metric != "mean_difference"
                    ):
                        continue
                    results = experiment.distance_results[model_name][algorithm][
                        shapley_method
                    ][distance_metric]
                    x_list = results[0]["x_list"]
                    data = np.array([results[i]["y_list"] for i in range(len(results))])
                    y_means = np.mean(data, axis=0)
                    y_sem = stats.sem(data, axis=0)
                    y_ci = y_sem * ci_factor

                    shapley_method_string_list = shapley_method.split("_")
                    if can_convert_to_float(shapley_method_string_list[-1]):
                        shapley_method = "_".join(shapley_method_string_list[:-1])
                        reg_str = "_" + shapley_method_string_list[-1]
                    else:
                        reg_str = ""

                    axes[i].plot(
                        x_list,
                        y_means,
                        label=labels.mapping[shapley_method] + reg_str,
                        # marker="o",
                    )
                    axes[i].fill_between(
                        x_list, y_means - y_ci, y_means + y_ci, alpha=0.2
                    )

                axes[i].set_xlabel("Interventions")
                axes[i].set_ylabel(labels.mapping[distance_metric])
                axes[i].legend()
                axes[i].grid(True)

            fig.subplots_adjust(wspace=0.4)
            fig.suptitle(f"{model_name}, {algorithm}")
            if save_to_file:
                fig.savefig(
                    f"pictures/{experiment.dataset.name}_distance_{model_name}_{algorithm}"
                )
            fig.show()
