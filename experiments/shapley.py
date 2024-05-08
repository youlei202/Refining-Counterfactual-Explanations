import shap
from explainers import pshap
import ot
import numpy as np


def can_convert_to_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def compute_shapley_values(
    model,
    X_train,
    X_factual,
    X_counterfactual,
    shapley_method,
):
    assert X_factual.shape[0] == X_counterfactual.shape[0]
    N = X_factual.shape[0]

    unshuffled_X_counterfactual = X_counterfactual.copy()
    np.random.shuffle(X_counterfactual)

    if shapley_method == "CF_UniformMatch":
        return shap.KernelExplainer(model.predict_proba, X_counterfactual).shap_values(
            X_factual
        )
    elif shapley_method == "Train_Distri":
        X_train_sampled = X_train.sample(N).values
        return shap.KernelExplainer(model.predict_proba, X_train_sampled).shap_values(
            X_factual
        )
    elif shapley_method == "CF_SingleMatch":
        probs = np.diag(np.ones(N)) / N
        return pshap.JointProbabilityExplainer(model).shap_values(
            X_factual, X_counterfactual, joint_probs=probs
        )
    elif shapley_method == "CF_ExactMatch":
        probs = np.diag(np.ones(N)) / N
        return pshap.JointProbabilityExplainer(model).shap_values(
            X_factual, unshuffled_X_counterfactual, joint_probs=probs
        )
    else:
        shapley_method_string_list = shapley_method.split("_")
        entropic_ot = can_convert_to_float(shapley_method_string_list[-1])
        if entropic_ot:
            reg = float(shapley_method_string_list[-1])
            shapley_method = "_".join(shapley_method_string_list[:-1])

        if shapley_method == "CF_OTMatch":
            ot_cost = ot.dist(X_factual, X_counterfactual)
            if entropic_ot:
                ot_plan = ot.bregman.sinkhorn(
                    np.ones(N) / N, np.ones(N) / N, ot_cost, reg=reg, numItermax=5000
                )
            else:
                ot_plan = ot.emd(np.ones(N) / N, np.ones(N) / N, ot_cost)
            return pshap.JointProbabilityExplainer(model).shap_values(
                X_factual, X_counterfactual, joint_probs=ot_plan
            )
        else:
            raise NotImplementedError


def compute_intervention_policy(shap_values):
    return np.abs(shap_values) / np.abs(shap_values).sum()
