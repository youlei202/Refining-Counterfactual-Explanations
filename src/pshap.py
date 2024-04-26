import numpy as np
import shap
from copy import deepcopy

class WeightedExplainer:
    def __init__(self, model):
        self.model = model

    def explain_instance(self, x, X_baseline, weights, num_samples=100):
        weights = weights / (weights.sum() + 1e-15)

        # Generate samples weighted by joint probabilities
        indices = np.random.choice(X_baseline.shape[0], size=num_samples, p=weights, replace=True)
        sampled_X_baseline = X_baseline[indices]

        # Use the sampled_X_baseline as the background data for this specific explanation
        explainer_temp = shap.KernelExplainer(self.model.predict_proba, sampled_X_baseline)
        shap_values = explainer_temp.shap_values(x)

        return shap_values


class JointProbabilityExplainer:
    def __init__(self, model):
        self.model = model
        self.weighted_explainer = WeightedExplainer(model)

    def shap_values(self, X, X_baseline, joint_probs, num_samples=100):
        return np.array([
            self.weighted_explainer.explain_instance(x, X_baseline, weights, num_samples)
            for x, weights in zip(X, joint_probs)
        ])