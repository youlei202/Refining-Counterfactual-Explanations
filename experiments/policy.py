import shap
from explainers import pshap
import ot
import numpy as np

EPSILON = 1e-20
SHAP_SAMPLE_SIZE = 10000
# SHAP_SAMPLE_SIZE = "auto"


def can_convert_to_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def convert_matrix_to_policy(matrix):
    P = np.abs(matrix) / np.abs(matrix).sum()
    P += EPSILON
    P /= P.sum()
    return P


class CounterfactualUniformDistributionPolicy:
    def __init__(self, model, X_factual, X_counterfactual):
        self.model = model
        self.X_factual = X_factual
        self.X_counterfactual = X_counterfactual
        self.N, self.M = self.X_factual.shape[0], self.X_counterfactual.shape[0]

    def compute_policy(self):
        shap_values = shap.KernelExplainer(
            self.model.predict_proba, self.X_counterfactual
        ).shap_values(self.X_factual, nsamples=SHAP_SAMPLE_SIZE)

        return {
            "c_policy": convert_matrix_to_policy(shap_values),
            "z_policy": convert_matrix_to_policy(
                np.full((self.N, self.M), fill_value=1)
            ),
        }


class TrainUniformDistributionPolicy:
    def __init__(self, model, X_factual, X_train):
        self.model = model
        self.X_factual = X_factual
        self.X_train = X_train
        self.N, self.M = self.X_factual.shape[0], self.X_counterfactual.shape[0]

    def compute_policy(self):
        X_train_sampled = self.X_train.sample(self.N).values
        shap_values = shap.KernelExplainer(
            self.model.predict_proba, X_train_sampled
        ).shap_values(self.X_factual, nsamples=SHAP_SAMPLE_SIZE)

        return {
            "c_policy": convert_matrix_to_policy(shap_values),
            "z_policy": convert_matrix_to_policy(
                np.full((self.N, self.M), fill_value=1)
            ),
        }


class CounterfactualSingleMatchingPolicy:
    def __init__(self, model, X_factual, X_counterfactual):
        self.model = model
        self.X_factual = X_factual
        self.X_counterfactual = X_counterfactual
        self.N, self.M = self.X_factual.shape[0], self.X_counterfactual.shape[0]

    def compute_policy(self):
        self.probs = np.zeros((self.N, self.M))
        # Assign 1/N to a random position in each row
        for i in range(self.N):
            self.probs[i, np.random.randint(0, self.M)] = 1 / self.N
        shap_values = pshap.JointProbabilityExplainer(self.model).shap_values(
            self.X_factual,
            self.X_counterfactual,
            joint_probs=self.probs,
            shap_sample_size=SHAP_SAMPLE_SIZE,
        )

        return {
            "c_policy": convert_matrix_to_policy(shap_values),
            "z_policy": convert_matrix_to_policy(self.probs),
        }


class CounterfactualOptimalTransportPolicy:
    def __init__(self, model, X_factual, X_counterfactual, reg=0):
        self.model = model
        self.X_factual = X_factual
        self.X_counterfactual = X_counterfactual
        self.N, self.M = self.X_factual.shape[0], self.X_counterfactual.shape[0]
        self.reg = reg

        self.ot_cost = ot.dist(self.X_factual, self.X_counterfactual, p=2)

    def compute_policy(self):
        if self.reg <= 0:
            self.ot_plan = ot.emd(
                np.ones(self.N) / self.N, np.ones(self.M) / self.M, self.ot_cost
            )
        else:
            self.ot_plan = ot.bregman.sinkhorn(
                np.ones(self.N) / self.N,
                np.ones(self.M) / self.M,
                self.ot_cost,
                reg=self.reg,
                numItermax=5000,
            )
        shap_values = pshap.JointProbabilityExplainer(self.model).shap_values(
            self.X_factual,
            self.X_counterfactual,
            joint_probs=self.ot_plan,
            shap_sample_size=SHAP_SAMPLE_SIZE,
        )

        return {
            "c_policy": convert_matrix_to_policy(shap_values),
            "z_policy": convert_matrix_to_policy(self.ot_plan),
        }


def compute_intervention_policy(
    model,
    X_train,
    X_factual,
    X_counterfactual,
    shapley_method,
):
    if shapley_method == "CF_UniformMatch":
        return CounterfactualUniformDistributionPolicy(
            model=model, X_factual=X_factual, X_counterfactual=X_counterfactual
        ).compute_policy()
    elif shapley_method == "Train_Distri":
        return TrainUniformDistributionPolicy(
            model=model, X_factual=X_factual, X_train=X_train
        ).compute_policy()
    elif shapley_method == "CF_SingleMatch":
        return CounterfactualSingleMatchingPolicy(
            model=model, X_factual=X_factual, X_counterfactual=X_counterfactual
        ).compute_policy()
    else:
        shapley_method_string_list = shapley_method.split("_")
        entropic_ot = can_convert_to_float(shapley_method_string_list[-1])
        if entropic_ot:
            reg = float(shapley_method_string_list[-1])
            shapley_method = "_".join(shapley_method_string_list[:-1])
        else:
            reg = 0
        if shapley_method == "CF_OTMatch":
            return CounterfactualOptimalTransportPolicy(
                model=model,
                X_factual=X_factual,
                X_counterfactual=X_counterfactual,
                reg=reg,
            ).compute_policy()
        else:
            raise NotImplementedError
