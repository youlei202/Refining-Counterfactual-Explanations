from sklearn.metrics import accuracy_score, classification_report
from dataset import GermanCreditDataset
from models.wrapper import ClassifierWrapper
import ot
import shap
from explainers import pshap
import numpy as np
from utils.logger_config import setup_logger
from experiments import shapley
from experiments.distances import compute_distance
from tqdm import tqdm

logger = setup_logger()


class Benchmarking:

    def __init__(
        self, dataset: GermanCreditDataset, models, shapley_methods, distance_metrics
    ):
        self.unwrapped_models = models

        self.models = []
        for unwrapped_model, backend in self.unwrapped_models:
            self.models.append(
                ClassifierWrapper(classifier=unwrapped_model, backend=backend)
            )

        self.model_reports = {}
        self.dataset = dataset

        self.model_names = []
        for model in self.models:
            self.model_names.append(model.classifier.__class__.__name__)

        self.shapley_methods = shapley_methods
        self.distance_metrics = distance_metrics

    def train_and_evaluate_models(self, random_state=None):
        self.X_train, self.X_test, self.y_train, self.y_test = (
            self.dataset.get_standardized_train_test_split(random_state=random_state)
        )

        for model in self.models:
            model.fit(self.X_train, self.y_train)

        self._get_performance_report(self.X_test, self.y_test)

    def _get_performance_report(self, X_test, y_test):

        for model, model_name in zip(self.models, self.model_names):
            y_pred = model.predict(X_test)
            self.model_reports[model_name] = {
                "accuracy": accuracy_score(y_test, y_pred),
                "classification_report": classification_report(y_test, y_pred),
            }

    def models_performance(self):
        for model_name in self.model_names:
            logger.info(
                f'{model_name} accuracy: {self.model_reports[self.model_names[0]]["accuracy"]}'
            )

    def compute_shapley_values(
        self,
        X_factual,
        model_counterfactuals,
    ):
        self.shap_values = {}
        self.X_factual = X_factual
        self.model_counterfactuals = model_counterfactuals

        for model, model_name in zip(self.models, self.model_names):
            self.shap_values[model_name] = {}

            for algorithm, counterfactual_dict in model_counterfactuals[
                model_name
            ].items():
                self.shap_values[model_name][algorithm] = {}
                X_counterfactual = counterfactual_dict["X"]

                for shapley_method in self.shapley_methods:
                    logger.info(
                        f"Shapley values for {model_name} using {shapley_method} with counterfactual by {algorithm}"
                    )
                    self.shap_values[model_name][algorithm][shapley_method] = (
                        shapley.compute_shapley_values(
                            model=model,
                            X_train=self.X_train,
                            X_factual=X_factual,
                            X_counterfactual=X_counterfactual,
                            shapley_method=shapley_method,
                        )
                    )

        return self.shap_values

    def evaluate_distance_performance_under_interventions(
        self, intervention_num_list, trials_num, plotting=False
    ):

        self.distance_results = {}

        for (model_name, model_dict), model in zip(
            self.shap_values.items(), self.models
        ):
            self.distance_results[model_name] = {}

            for algorithm, algorithm_dict in model_dict.items():
                self.distance_results[model_name][algorithm] = {}
                X_counterfactual = self.model_counterfactuals[model_name][algorithm][
                    "X"
                ]
                y_counterfactual = self.model_counterfactuals[model_name][algorithm][
                    "y"
                ]

                for shapley_method, values in algorithm_dict.items():
                    self.distance_results[model_name][algorithm][shapley_method] = {}
                    P = shapley.compute_intervention_policy(shap_values=values)

                    for distance_metric in self.distance_metrics:

                        logger.info(
                            f"Computing {distance_metric} for ({model_name}, {algorithm}, {shapley_method})"
                        )
                        results_list = []
                        for t in tqdm(range(trials_num)):
                            trial_result = {
                                "x_list": intervention_num_list,
                                "y_list": [],
                            }

                            for intervention_num in intervention_num_list:
                                intervention_indices = np.random.choice(
                                    a=P.size,
                                    size=intervention_num,
                                    p=P.flatten(),
                                    replace=True,
                                )
                                intervention_indices = np.unique(intervention_indices)

                                # Convert flat indices back to 2D indices
                                i_indices, j_indices = np.unravel_index(
                                    intervention_indices, P.shape
                                )
                                X_intervention = self.X_factual.copy()

                                if intervention_num > 0:
                                    # Set values at selected 2D indices
                                    values_from_X_counterfactual = X_counterfactual[
                                        i_indices, j_indices
                                    ]
                                    X_intervention[i_indices, j_indices] = (
                                        values_from_X_counterfactual
                                    )
                                y_intervention = model.predict_proba(X_intervention)

                                result = compute_distance(
                                    y_intervention, y_counterfactual, distance_metric
                                )
                                trial_result["y_list"].append(result)

                            results_list.append(trial_result)

                        self.distance_results[model_name][algorithm][shapley_method][
                            distance_metric
                        ] = results_list
