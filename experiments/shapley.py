import shap
from explainers import pshap
import ot
import numpy as np


def compute_shapley_values(
    model,
    X_train,
    X_factual,
    X_counterfactual,
    shapley_method,
):
    assert X_factual.shape[0] == X_counterfactual.shape[0]
    N = X_factual.shape[0]

    if shapley_method == "independent_X_counterfactual":
        return shap.KernelExplainer(model.predict_proba, X_counterfactual).shap_values(
            X_factual
        )
    elif shapley_method == "independent_X_train":
        X_train_sampled = X_train.sample(N).values
        return shap.KernelExplainer(model.predict_proba, X_train_sampled).shap_values(
            X_factual
        )
    elif shapley_method == "joint_probability_X_counterfactual":
        ot_cost = ot.dist(X_factual, X_counterfactual)
        ot_plan = ot.emd(np.ones(N) / N, np.ones(N) / N, ot_cost)
        return pshap.JointProbabilityExplainer(model).shap_values(
            X_factual, X_counterfactual, joint_probs=ot_plan
        )
    elif shapley_method == "joint_probability_X_train":
        X_train_sampled = X_train.sample(N).values
        ot_cost = ot.dist(X_factual, X_train_sampled)
        ot_plan = ot.emd(np.ones(N) / N, np.ones(N) / N, ot_cost)
        return pshap.JointProbabilityExplainer(model).shap_values(
            X_factual, X_train_sampled, joint_probs=ot_plan
        )
    else:
        raise NotImplementedError


def compute_intervention_policy(shap_values):
    return np.abs(shap_values) / np.abs(shap_values).sum()
