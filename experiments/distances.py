import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, entropy
from scipy.special import kl_div
import ot


def gaussian_kernel(x, y, sigma=1.0):
    """Compute the Gaussian kernel between x and y"""
    return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * sigma**2))


def compute_mmd(y_s, y_t, kernel=gaussian_kernel):
    n = y_s.shape[0]
    m = y_t.shape[0]

    ss = np.sum([kernel(y_s[i], y_s[j]) for i in range(n) for j in range(n)])
    tt = np.sum([kernel(y_t[i], y_t[j]) for i in range(m) for j in range(m)])
    st = np.sum([kernel(y_s[i], y_t[j]) for i in range(n) for j in range(m)])

    return ss / (n**2) + tt / (m**2) - 2 * st / (n * m)


def compute_distance(y_s: np.array, y_t: np.array, distance_metric: str):
    assert y_s.shape[0] == y_t.shape[0]
    N = y_s.shape[0]
    if distance_metric == "optimal_transport":
        ot_cost = ot.dist(y_s.reshape(-1, 1), y_t.reshape(-1, 1))
        ot_plan = ot.emd(np.ones(N) / N, np.ones(N) / N, ot_cost)
        return np.sum(ot_plan * ot_cost)
    elif distance_metric == "kl_divergence":
        return kl_div(y_s, y_t).mean()
    elif distance_metric == "mean_difference":
        return abs(y_s.mean() - y_t.mean())
    elif distance_metric == "max_mean_discrepancy":
        return compute_mmd(y_s, y_t)
    else:
        raise NotImplementedError
