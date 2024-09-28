import shap
from explainers import pshap
import ot
import numpy as np
from scipy.special import rel_entr
from explainers.infoot import FusedInfoOT
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.linear_model import LogisticRegression
from cem import CEM
import pandas as pd
from sklearn.neighbors import NearestNeighbors

EPSILON = 1e-20
# SHAP_SAMPLE_SIZE = 10000
SHAP_SAMPLE_SIZE = "auto"


def difference_mask(arr1, arr2, atol=1e-8, rtol=1e-5):
    # Ensure both inputs are NumPy arrays
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)

    # Check if shapes are the same
    if arr1.shape != arr2.shape:
        raise ValueError(
            f"Shape mismatch: arr1.shape = {arr1.shape}, arr2.shape = {arr2.shape}"
        )

    # Use numpy.isclose to compare arrays element-wise
    close = np.isclose(arr1, arr2, atol=atol, rtol=rtol)

    # Invert the boolean array: True where elements differ beyond tolerance
    differences = ~close

    # Convert boolean array to integer (0 and 1)
    binary_mask = differences.astype(int)

    return binary_mask


def COLA(X_factual, varphi, q, C, replace):
    action_indice = np.random.choice(
        a=varphi.size,
        size=C,
        p=varphi.flatten(),
        replace=replace,
    )
    action_indice = np.unique(action_indice)

    # Convert flat indices back to 2D indices
    i_indice, k_indice = np.unravel_index(action_indice, varphi.shape)

    Z_counterfactual = X_factual.copy()
    if C > 0:
        # Set values at selected 2D indices
        values_from_q = q[i_indice, k_indice]
        Z_counterfactual[i_indice, k_indice] = values_from_q

    return Z_counterfactual, action_indice


def A_values(W, R, method):
    N, M = W.shape
    _, P = R.shape
    Q = np.zeros((N, P))

    if method == "avg":
        for i in range(N):
            weights = W[i, :]
            # Normalize weights to ensure they sum to 1
            normalized_weights = weights / np.sum(weights)
            # Reshape to match R's rows for broadcasting
            normalized_weights = normalized_weights.reshape(-1, 1)
            # Compute the weighted sum
            Q[i, :] = np.sum(normalized_weights * R, axis=0)
    elif method == "max":
        for i in range(N):
            max_weight_index = np.argmax(W[i, :])
            Q[i, :] = R[max_weight_index, :]
    else:
        raise NotImplementedError
    return Q


class Policy:
    def __init__(self, model, X_factual, X_counterfactual):
        self.model = model
        self.X_factual = X_factual
        self.X_counterfactual = X_counterfactual
        self.N, self.M = self.X_factual.shape[0], self.X_counterfactual.shape[0]
        self.diff_mask = difference_mask(X_factual, X_counterfactual)


class CounterfactualPolicy(Policy):
    def __init__(self, model, X_factual, X_counterfactual):
        super().__init__(model, X_factual, X_counterfactual)


class TrainsetPolicy(Policy):
    def __init__(self, model, X_factual, X_train, X_counterfactual):
        super().__init__(model, X_factual, X_counterfactual)
        self.X_train_sampled = X_train.sample(self.M).values


class TrainUniformDistributionPolicy(TrainsetPolicy):
    def __init__(self, model, X_factual, X_train, X_counterfactual, method="avg"):
        super().__init__(
            model=model,
            X_factual=X_factual,
            X_train=X_train,
            X_counterfactual=X_counterfactual,
        )
        self.method = method

    def compute_policy(self):
        p = get_uniform_distribution_matrix(self.N, self.M)

        shap_values = shap.KernelExplainer(
            self.model.predict_proba, self.X_train_sampled
        ).shap_values(self.X_factual, nsamples=SHAP_SAMPLE_SIZE)

        varphi = convert_matrix_to_policy_with_mask(shap_values, self.diff_mask)
        q = A_values(W=p, R=self.X_counterfactual, method=self.method)

        return {"varphi": varphi, "p": p, "q": q}


class TrainOptimalTransportPolicy(TrainsetPolicy):
    def __init__(
        self, model, X_factual, X_train, X_counterfactual, reg=0, method="avg"
    ):
        super().__init__(
            model=model,
            X_factual=X_factual,
            X_train=X_train,
            X_counterfactual=X_counterfactual,
        )
        self.reg = reg
        self.method = method
        self.ot_cost = ot.dist(self.X_factual, self.X_counterfactual, p=2)

    def compute_policy(self):
        if self.reg <= 0:
            probs_matrix = ot.emd(
                np.ones(self.N) / self.N, np.ones(self.M) / self.M, self.ot_cost
            )
        else:
            probs_matrix = ot.bregman.sinkhorn(
                np.ones(self.N) / self.N,
                np.ones(self.M) / self.M,
                self.ot_cost,
                reg=self.reg,
                numItermax=5000,
            )
        p = convert_matrix_to_policy(probs_matrix)

        shap_values = shap.KernelExplainer(
            self.model.predict_proba, self.X_train_sampled
        ).shap_values(self.X_factual, nsamples=SHAP_SAMPLE_SIZE)

        varphi = convert_matrix_to_policy_with_mask(shap_values, self.diff_mask)
        q = A_values(W=p, R=self.X_counterfactual, method=self.method)

        return {"varphi": varphi, "p": p, "q": q}


class CounterfactualUniformDistributionPolicy(CounterfactualPolicy):
    def __init__(self, model, X_factual, X_counterfactual, method="avg"):
        super().__init__(model, X_factual, X_counterfactual)
        self.method = method

    def compute_policy(self):
        p = get_uniform_distribution_matrix(self.N, self.M)

        shap_values = pshap.JointProbabilityExplainer(self.model).shap_values(
            self.X_factual,
            self.X_counterfactual,
            joint_probs=p,
            shap_sample_size=SHAP_SAMPLE_SIZE,
        )

        varphi = convert_matrix_to_policy_with_mask(shap_values, self.diff_mask)
        q = A_values(W=p, R=self.X_counterfactual, method=self.method)

        return {"varphi": varphi, "p": p, "q": q}


class CounterfactualSingleMatchingPolicy(CounterfactualPolicy):
    def __init__(self, model, X_factual, X_counterfactual, method="avg"):
        super().__init__(model, X_factual, X_counterfactual)
        self.method = method

    def compute_policy(self):
        p = get_one_one_distribution_matrix(self.N, self.M)

        shap_values = pshap.JointProbabilityExplainer(self.model).shap_values(
            self.X_factual,
            self.X_counterfactual,
            joint_probs=p,
            shap_sample_size=SHAP_SAMPLE_SIZE,
        )

        varphi = convert_matrix_to_policy_with_mask(shap_values, self.diff_mask)
        q = A_values(W=p, R=self.X_counterfactual, method=self.method)

        return {"varphi": varphi, "p": p, "q": q}


class CounterfactualRandomMatchingPolicy(CounterfactualPolicy):
    def __init__(self, model, X_factual, X_counterfactual, method="avg"):
        super().__init__(model, X_factual, X_counterfactual)
        self.method = method

    def compute_policy(self):
        p = get_random_distribution_matrix(self.N, self.M)

        shap_values = pshap.JointProbabilityExplainer(self.model).shap_values(
            self.X_factual,
            self.X_counterfactual,
            joint_probs=p,
            shap_sample_size=SHAP_SAMPLE_SIZE,
        )

        varphi = convert_matrix_to_policy_with_mask(shap_values, self.diff_mask)
        q = A_values(W=p, R=self.X_counterfactual, method=self.method)

        return {"varphi": varphi, "p": p, "q": q}


class CounterfactualRandomSingleMatchingPolicy(CounterfactualPolicy):
    def __init__(self, model, X_factual, X_counterfactual, method="avg"):
        super().__init__(model, X_factual, X_counterfactual)
        self.method = method

    def compute_policy(self):
        p = get_one_one_distribution_matrix(self.N, self.M)

        shap_values = pshap.JointProbabilityExplainer(self.model).shap_values(
            self.X_factual,
            self.X_counterfactual,
            joint_probs=p,
            shap_sample_size=SHAP_SAMPLE_SIZE,
        )

        varphi = convert_matrix_to_policy_with_mask(shap_values, self.diff_mask)
        q = A_values(W=p, R=self.X_counterfactual, method=self.method)

        return {"varphi": varphi, "p": p, "q": q}


class CounterfactualExactMatchingPolicy(CounterfactualPolicy):
    def __init__(self, model, X_factual, X_counterfactual, method="avg"):
        super().__init__(model, X_factual, X_counterfactual)
        self.method = method
        assert self.N == self.M

    def compute_policy(self):
        p = get_exact_one_one_matrix(self.N)

        shap_values = pshap.JointProbabilityExplainer(self.model).shap_values(
            self.X_factual,
            self.X_counterfactual,
            joint_probs=p,
            shap_sample_size=SHAP_SAMPLE_SIZE,
        )

        varphi = convert_matrix_to_policy_with_mask(shap_values, self.diff_mask)
        q = A_values(W=p, R=self.X_counterfactual, method=self.method)

        return {"varphi": varphi, "p": p, "q": q}


class CounterfactualOptimalTransportPolicy(CounterfactualPolicy):
    def __init__(self, model, X_factual, X_counterfactual, reg=0, method="avg"):
        super().__init__(model, X_factual, X_counterfactual)
        self.reg = reg
        self.method = method
        self.ot_cost = ot.dist(self.X_factual, self.X_counterfactual, p=2)

    def compute_policy(self):
        if self.reg <= 0:
            probs_matrix = ot.emd(
                np.ones(self.N) / self.N, np.ones(self.M) / self.M, self.ot_cost
            )
        else:
            probs_matrix = ot.bregman.sinkhorn(
                np.ones(self.N) / self.N,
                np.ones(self.M) / self.M,
                self.ot_cost,
                reg=self.reg,
                numItermax=5000,
            )
        p = convert_matrix_to_policy(probs_matrix)

        shap_values = pshap.JointProbabilityExplainer(self.model).shap_values(
            self.X_factual,
            self.X_counterfactual,
            joint_probs=p,
            shap_sample_size=SHAP_SAMPLE_SIZE,
        )

        varphi = convert_matrix_to_policy_with_mask(shap_values, self.diff_mask)
        q = A_values(W=p, R=self.X_counterfactual, method=self.method)

        return {"varphi": varphi, "p": p, "q": q}


class CounterfactualUnbalancedOptimalTransportPolicy(CounterfactualPolicy):
    def __init__(self, model, X_factual, X_counterfactual, reg=0, method="avg"):
        super().__init__(model, X_factual, X_counterfactual)
        self.reg = reg
        self.method = method
        self.ot_cost = ot.dist(self.X_factual, self.X_counterfactual, p=2)

    def compute_policy(self):
        probs_matrix = ot.unbalanced.mm_unbalanced(
            np.ones(self.N) / self.N,
            np.ones(self.M) / self.M,
            self.ot_cost,
            reg_m=5,
        )
        p = convert_matrix_to_policy(probs_matrix)

        shap_values = pshap.JointProbabilityExplainer(self.model).shap_values(
            self.X_factual,
            self.X_counterfactual,
            joint_probs=p,
            shap_sample_size=SHAP_SAMPLE_SIZE,
        )

        varphi = convert_matrix_to_policy_with_mask(shap_values, self.diff_mask)
        q = A_values(W=p, R=self.X_counterfactual, method=self.method)

        return {"varphi": varphi, "p": p, "q": q}


class CounterfactualCausalOptimalTransportPolicy(CounterfactualPolicy):
    def __init__(self, model, X_factual, X_counterfactual, method="avg"):
        super().__init__(model, X_factual, X_counterfactual)
        self.method = method

    def _compute_propensity_scores(self, x, r):
        # Combine the factual and counterfactual data
        combined_data = np.vstack((x, r))
        treatment_indicator = np.hstack((np.zeros(x.shape[0]), np.ones(r.shape[0])))

        # Fit a logistic regression model to estimate propensity scores
        model = LogisticRegression()
        model.fit(combined_data, treatment_indicator)
        propensity_scores = model.predict_proba(combined_data)[:, 1]

        return propensity_scores[: x.shape[0]], propensity_scores[x.shape[0] :]

    def _compute_causal_ot_prob_matrix(self, x, r):
        # Compute the cost matrix (e.g., Euclidean distance plus propensity score difference)
        propensity_scores_x, propensity_scores_r = self._compute_propensity_scores(x, r)
        cost_matrix = ot.dist(x, r, metric="euclidean")

        # Add causal component to the cost matrix
        for i in range(x.shape[0]):
            for j in range(r.shape[0]):
                cost_matrix[i, j] += np.abs(
                    propensity_scores_x[i] - propensity_scores_r[j]
                )

        # Uniform distribution over the rows of x and r
        a = np.ones(x.shape[0]) / x.shape[0]
        b = np.ones(r.shape[0]) / r.shape[0]

        # Compute the optimal transport plan
        transport_plan = ot.emd(a, b, cost_matrix)

        # The transport plan is the probability matrix
        prob_matrix = transport_plan

        return prob_matrix

    def compute_policy(self):
        p = self._compute_causal_ot_prob_matrix(self.X_factual, self.X_counterfactual)

        # Remove rows in p that sum to zero (unmatched rows in x)
        row_sums = p.sum(axis=1)
        matched_p = p[row_sums > 0, :]

        shap_values = pshap.JointProbabilityExplainer(self.model).shap_values(
            self.X_factual[row_sums > 0, :],
            self.X_counterfactual,
            joint_probs=matched_p,
            shap_sample_size=SHAP_SAMPLE_SIZE,
        )

        varphi = convert_matrix_to_policy_with_mask(shap_values, self.diff_mask)
        q = A_values(W=matched_p, R=self.X_counterfactual, method=self.method)

        return {"varphi": varphi, "p": matched_p, "q": q}


class CounterfactualInfomationOptimalTransportPolicy(CounterfactualPolicy):
    def __init__(self, model, X_factual, X_counterfactual, reg=0, method="avg"):
        super().__init__(model, X_factual, X_counterfactual)
        self.reg = reg
        self.method = method

    def compute_policy(self):
        p = FusedInfoOT(
            Xs=self.X_factual, Xt=self.X_counterfactual, h=0, reg=self.reg
        ).solve()

        shap_values = pshap.JointProbabilityExplainer(self.model).shap_values(
            self.X_factual,
            self.X_counterfactual,
            joint_probs=p,
            shap_sample_size=SHAP_SAMPLE_SIZE,
        )

        varphi = convert_matrix_to_policy_with_mask(shap_values, self.diff_mask)
        q = A_values(W=p, R=self.X_counterfactual, method=self.method)

        return {"varphi": varphi, "p": p, "q": q}


class CounterfactualMaximumMutualInformationPolicy(CounterfactualPolicy):
    def __init__(self, model, X_factual, X_counterfactual, method="avg"):
        super().__init__(model, X_factual, X_counterfactual)
        self.method = method

    def compute_joint_prob_matrix_with_max_mi(self, max_iter=1000, tol=1e-6):
        N = self.X_factual.shape[0]
        M = self.X_counterfactual.shape[0]

        # Initialize uniform marginal probabilities
        p_factual = np.ones(N) / N
        p_counterfactual = np.ones(M) / M

        # Initialize the joint probability matrix
        joint_prob_matrix = np.outer(p_factual, p_counterfactual)

        for _ in range(max_iter):
            # Compute the conditional probability p(X_counterfactual | X_factual)
            cond_prob_matrix = joint_prob_matrix / np.sum(
                joint_prob_matrix, axis=1, keepdims=True
            )
            # Compute new joint probability matrix
            new_joint_prob_matrix = np.outer(p_factual, p_counterfactual) * np.exp(
                cond_prob_matrix
            )
            new_joint_prob_matrix /= np.sum(new_joint_prob_matrix)

            # Check for convergence
            if np.linalg.norm(new_joint_prob_matrix - joint_prob_matrix) < tol:
                break

            joint_prob_matrix = new_joint_prob_matrix

        return joint_prob_matrix

    def compute_policy(self):
        joint_prob_matrix = self.compute_joint_prob_matrix_with_max_mi()

        p = convert_matrix_to_policy(joint_prob_matrix)

        shap_values = pshap.JointProbabilityExplainer(self.model).shap_values(
            self.X_factual,
            self.X_counterfactual,
            joint_probs=p,
            shap_sample_size=SHAP_SAMPLE_SIZE,
        )

        varphi = convert_matrix_to_policy_with_mask(shap_values, self.diff_mask)
        q = A_values(W=p, R=self.X_counterfactual, method=self.method)

        return {"varphi": varphi, "p": p, "q": q}


class CounterfactualMinimumMutualInformationPolicy(CounterfactualPolicy):
    def __init__(self, model, X_factual, X_counterfactual, method="avg"):
        super().__init__(model, X_factual, X_counterfactual)
        self.method = method

    def compute_joint_prob_matrix_with_min_mi(self):
        N = self.X_factual.shape[0]
        M = self.X_counterfactual.shape[0]

        # Initialize uniform marginal probabilities
        p_factual = np.ones(N) / N
        p_counterfactual = np.ones(M) / M

        # Compute the joint probability matrix that minimizes mutual information
        joint_prob_matrix = np.outer(p_factual, p_counterfactual)

        return joint_prob_matrix

    def compute_policy(self):
        joint_prob_matrix = self.compute_joint_prob_matrix_with_min_mi()

        p = convert_matrix_to_policy(joint_prob_matrix)

        shap_values = pshap.JointProbabilityExplainer(self.model).shap_values(
            self.X_factual,
            self.X_counterfactual,
            joint_probs=p,
            shap_sample_size=SHAP_SAMPLE_SIZE,
        )

        varphi = convert_matrix_to_policy_with_mask(shap_values, self.diff_mask)
        q = A_values(W=p, R=self.X_counterfactual, method=self.method)

        return {"varphi": varphi, "p": p, "q": q}


class CounterfactualCoarsenedExactMatchingPolicy(CounterfactualPolicy):
    def __init__(self, model, X_factual, X_counterfactual, n_bins=20, method="avg"):
        super().__init__(model, X_factual, X_counterfactual)
        self.n_bins = n_bins
        self.method = method

    def _compute_cem_prob_matrix(self, x, r):
        # Coarsen the data into bins
        discretizer = KBinsDiscretizer(
            n_bins=self.n_bins, encode="ordinal", strategy="uniform"
        )
        x_binned = discretizer.fit_transform(x)
        r_binned = discretizer.transform(r)

        # Initialize the probability matrix
        prob_matrix = np.zeros((x.shape[0], r.shape[0]))

        # Compute the matching probabilities
        for i, r_bin in enumerate(r_binned):
            matches = np.all(x_binned == r_bin, axis=1)
            matched_indices = np.where(matches)[0]
            if len(matched_indices) > 0:
                prob_matrix[matched_indices, i] = 1.0 / len(matched_indices)

        return prob_matrix

    def compute_policy(self):
        p = self._compute_cem_prob_matrix(self.X_factual, self.X_counterfactual)

        shap_values = pshap.JointProbabilityExplainer(self.model).shap_values(
            self.X_factual,
            self.X_counterfactual,
            joint_probs=p,
            shap_sample_size=SHAP_SAMPLE_SIZE,
        )

        varphi = convert_matrix_to_policy_with_mask(shap_values, self.diff_mask)
        q = A_values(W=p, R=self.X_counterfactual, method=self.method)

        return {"varphi": varphi, "p": p, "q": q}


class CounterfactualCoarsenedExactMatchingOTPolicy(CounterfactualPolicy):
    def __init__(self, model, X_factual, X_counterfactual, n_bins=5, method="avg"):
        super().__init__(model, X_factual, X_counterfactual)
        self.n_bins = n_bins
        self.method = method

    def _compute_cem_prob_matrix(self, x, r):
        # Combine the factual and counterfactual data
        combined_data = np.vstack((x, r))
        treatment_indicator = np.hstack((np.zeros(x.shape[0]), np.ones(r.shape[0])))

        # Perform Coarsened Exact Matching
        df = pd.DataFrame(combined_data)
        df["treatment"] = treatment_indicator
        cem_result = CEM(df, "treatment", drop="drop", cut=self.n_bins)
        matched_groups = cem_result["matched"]

        # Initialize the probability matrix
        prob_matrix = np.zeros((x.shape[0], r.shape[0]))

        # Fill the probability matrix based on matching results
        for group in matched_groups:
            x_indices = [idx for idx in group if idx < x.shape[0]]
            r_indices = [idx - x.shape[0] for idx in group if idx >= x.shape[0]]
            if x_indices and r_indices:
                prob = 1.0 / len(x_indices)
                for x_idx in x_indices:
                    for r_idx in r_indices:
                        prob_matrix[x_idx, r_idx] = prob

        return prob_matrix

    def _compute_ot_for_unmatched(self, x, r, prob_matrix):
        # Identify unmatched rows in x
        row_sums = prob_matrix.sum(axis=1)
        unmatched_x_indices = np.where(row_sums == 0)[0]

        if len(unmatched_x_indices) == 0:
            return prob_matrix

        # Compute the cost matrix for unmatched rows
        cost_matrix = ot.dist(x[unmatched_x_indices], r, metric="euclidean")

        # Uniform distribution over the unmatched rows of x and all rows of r
        a = np.ones(len(unmatched_x_indices)) / len(unmatched_x_indices)
        b = np.ones(r.shape[0]) / r.shape[0]

        # Compute the optimal transport plan
        transport_plan = ot.emd(a, b, cost_matrix)

        # Update the probability matrix with the OT results for unmatched rows
        for i, x_idx in enumerate(unmatched_x_indices):
            for j, r_idx in enumerate(range(r.shape[0])):
                prob_matrix[x_idx, r_idx] = transport_plan[i, j]

        return prob_matrix

    def compute_policy(self):
        # Compute initial probability matrix using CEM
        p = self._compute_cem_prob_matrix(self.X_factual, self.X_counterfactual)

        # Ensure all rows in x are matched using OT for unmatched rows
        p = self._compute_ot_for_unmatched(self.X_factual, self.X_counterfactual, p)

        # Remove rows in p that sum to zero (shouldn't be any after OT adjustment)
        row_sums = p.sum(axis=1)
        matched_p = p[row_sums > 0, :]

        shap_values = pshap.JointProbabilityExplainer(self.model).shap_values(
            self.X_factual[row_sums > 0, :],
            self.X_counterfactual,
            joint_probs=matched_p,
            shap_sample_size=SHAP_SAMPLE_SIZE,
        )

        varphi = convert_matrix_to_policy_with_mask(shap_values, self.diff_mask)
        q = A_values(W=matched_p, R=self.X_counterfactual, method=self.method)

        return {"varphi": varphi, "p": matched_p, "q": q}


class CounterfactualNearestNeighborMatchingPolicy(CounterfactualPolicy):
    def __init__(self, model, X_factual, X_counterfactual, method="avg"):
        super().__init__(model, X_factual, X_counterfactual)
        self.method = method

    def _compute_nn_prob_matrix(self, x, r):
        # Fit nearest neighbors model on the counterfactual data
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(r)

        # Find the nearest neighbors in r for each row in x
        distances, indices = nn.kneighbors(x)

        # Initialize the probability matrix
        prob_matrix = np.zeros((x.shape[0], r.shape[0]))

        # Fill the probability matrix based on nearest neighbors
        for i, neighbor_index in enumerate(indices.flatten()):
            prob_matrix[i, neighbor_index] = 1.0

        return prob_matrix

    def compute_policy(self):
        # Compute initial probability matrix using nearest neighbors
        p = self._compute_nn_prob_matrix(self.X_factual, self.X_counterfactual)

        # Optionally, you can also use OT to refine this further, but this example sticks to NN only
        # Remove rows in p that sum to zero (shouldn't be any if NN is used correctly)
        row_sums = p.sum(axis=1)
        matched_p = p[row_sums > 0, :]

        shap_values = pshap.JointProbabilityExplainer(self.model).shap_values(
            self.X_factual[row_sums > 0, :],
            self.X_counterfactual,
            joint_probs=matched_p,
            shap_sample_size=SHAP_SAMPLE_SIZE,
        )

        varphi = convert_matrix_to_policy_with_mask(shap_values, self.diff_mask)
        q = A_values(W=matched_p, R=self.X_counterfactual, method=self.method)

        return {"varphi": varphi, "p": matched_p, "q": q}


def compute_intervention_policy(
    model,
    X_train,
    X_factual,
    X_counterfactual,
    shapley_method,
    Avalues_method,
):
    if shapley_method == "CF_UniformMatch":
        return CounterfactualUniformDistributionPolicy(
            model=model,
            X_factual=X_factual,
            X_counterfactual=X_counterfactual,
            method=Avalues_method,
        ).compute_policy()
    elif shapley_method == "Train_Distri":
        return TrainUniformDistributionPolicy(
            model=model,
            X_factual=X_factual,
            X_train=X_train,
            X_counterfactual=X_counterfactual,
            method=Avalues_method,
        ).compute_policy()
    elif shapley_method == "Train_OTMatch":
        return TrainOptimalTransportPolicy(
            model=model,
            X_factual=X_factual,
            X_train=X_train,
            X_counterfactual=X_counterfactual,
            method=Avalues_method,
        ).compute_policy()
    elif shapley_method == "CF_ExactMatch":
        return CounterfactualExactMatchingPolicy(
            model=model,
            X_factual=X_factual,
            X_counterfactual=X_counterfactual,
            method=Avalues_method,
        ).compute_policy()
    elif shapley_method == "CF_RandomMatch":
        return CounterfactualRandomMatchingPolicy(
            model=model,
            X_factual=X_factual,
            X_counterfactual=X_counterfactual,
            method=Avalues_method,
        ).compute_policy()
    elif shapley_method == "CF_RandomSingleMatch":
        return CounterfactualRandomSingleMatchingPolicy(
            model=model,
            X_factual=X_factual,
            X_counterfactual=X_counterfactual,
            method=Avalues_method,
        ).compute_policy()
    elif shapley_method == "CF_MaximumMutualInformation":
        return CounterfactualMaximumMutualInformationPolicy(
            model=model,
            X_factual=X_factual,
            X_counterfactual=X_counterfactual,
            method=Avalues_method,
        ).compute_policy()
    elif shapley_method == "CF_MinimumMutualInformation":
        return CounterfactualMinimumMutualInformationPolicy(
            model=model,
            X_factual=X_factual,
            X_counterfactual=X_counterfactual,
            method=Avalues_method,
        ).compute_policy()
    elif shapley_method == "CF_CEMMatch":
        return CounterfactualCoarsenedExactMatchingPolicy(
            model=model,
            X_factual=X_factual,
            X_counterfactual=X_counterfactual,
            method=Avalues_method,
        ).compute_policy()
    elif shapley_method == "CF_CEMOTMatch":
        return CounterfactualCoarsenedExactMatchingOTPolicy(
            model=model,
            X_factual=X_factual,
            X_counterfactual=X_counterfactual,
            method=Avalues_method,
        ).compute_policy()
    elif shapley_method == "CF_NNMatch":
        return CounterfactualNearestNeighborMatchingPolicy(
            model=model,
            X_factual=X_factual,
            X_counterfactual=X_counterfactual,
            method=Avalues_method,
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
                method=Avalues_method,
            ).compute_policy()
        elif shapley_method == "CF_UBOTMatch":
            return CounterfactualUnbalancedOptimalTransportPolicy(
                model=model,
                X_factual=X_factual,
                X_counterfactual=X_counterfactual,
                reg=reg,
                method=Avalues_method,
            ).compute_policy()
        elif shapley_method == "CF_InfoOTMatch":
            return CounterfactualOptimalTransportPolicy(
                model=model,
                X_factual=X_factual,
                X_counterfactual=X_counterfactual,
                reg=reg,
                method=Avalues_method,
            ).compute_policy()
        elif shapley_method == "CF_CausalOTMatch":
            return CounterfactualCausalOptimalTransportPolicy(
                model=model,
                X_factual=X_factual,
                X_counterfactual=X_counterfactual,
                method=Avalues_method,
            ).compute_policy()
        else:
            raise NotImplementedError


def can_convert_to_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def get_uniform_distribution_matrix(N, M):
    uniform_matrix = np.full((N, M), fill_value=1)
    return convert_matrix_to_policy(uniform_matrix)


def get_exact_one_one_matrix(N):
    eye_matrix = np.eye(N) / N
    return convert_matrix_to_policy(eye_matrix)


def get_one_one_distribution_matrix(N, M):
    probs_matrix = np.zeros((N, M))
    for i in range(N):
        probs_matrix[i, np.random.randint(0, M)] = 1 / N
    return convert_matrix_to_policy(probs_matrix)


def get_random_distribution_matrix(N, M):
    probs_matrix = np.random.rand(N, M)
    return convert_matrix_to_policy(probs_matrix)


def convert_matrix_to_policy(matrix):
    P = np.abs(matrix) / np.abs(matrix).sum()
    P += EPSILON
    P /= P.sum()
    return P


def convert_matrix_to_policy_with_mask(matrix, diff_mask):
    P = np.abs(matrix) / np.abs(matrix).sum()
    P *= diff_mask
    P += EPSILON
    P /= P.sum()
    return P
