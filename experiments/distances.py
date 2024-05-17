import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, entropy
from scipy.special import kl_div
import ot
from sklearn.metrics.pairwise import rbf_kernel


def compute_mmd(y_s, y_t, gamma=1.0):
    """MMD using RBF (Gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))

    Arguments:
        y_s {array-like} -- Source 1D array
        y_t {array-like} -- Target 1D array

    Keyword Arguments:
        gamma {float} -- Kernel parameter (default: {1.0})

    Returns:
        float -- MMD value
    """
    # Reshape 1D arrays to 2D arrays with one feature
    y_s = np.array(y_s).reshape(-1, 1)
    y_t = np.array(y_t).reshape(-1, 1)

    XX = rbf_kernel(y_s, y_s, gamma)
    YY = rbf_kernel(y_t, y_t, gamma)
    XY = rbf_kernel(y_s, y_t, gamma)

    return XX.mean() + YY.mean() - 2 * XY.mean()


def compute_distance(y_s: np.array, y_t: np.array, distance_metric: str):
    N, M = y_s.shape[0], y_t.shape[0]
    if distance_metric == "optimal_transport":
        ot_cost = ot.dist(y_s.reshape(-1, 1), y_t.reshape(-1, 1))
        ot_plan = ot.emd(np.ones(N) / N, np.ones(M) / M, ot_cost)
        return np.sum(ot_plan * ot_cost)
    elif distance_metric == "kl_divergence":
        return kl_div(y_s, y_t).mean()
    elif distance_metric == "mean_difference":
        return abs(y_s.mean() - y_t.mean())
    elif distance_metric == "median_difference":
        return abs(np.median(y_s) - np.median(y_t))
    elif distance_metric == "max_mean_discrepancy":
        return compute_mmd(y_s, y_t)
    else:
        raise NotImplementedError
