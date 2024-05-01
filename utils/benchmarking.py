import torch
import numpy as np
import pandas as pd
from explainers.distances import WassersteinDivergence
from scipy.stats import gaussian_kde, entropy
from numpy.linalg import LinAlgError



def get_ot_plan(mu_list):
    mu_sum = torch.zeros_like(mu_list[0])
    for mu in mu_list:
        mu_sum += mu


    # Initialize a tensor to store the maximum values, with the same shape and zeros
    mu_max = torch.zeros_like(mu_sum)

    # Find the indices of the maximum values in each row
    _, max_indices = torch.max(mu_sum, dim=1, keepdim=True)

    # Use the indices to place the maximum values in the 'max_only' tensor
    mu_max.scatter_(dim=1, index=max_indices, src=mu_sum.gather(dim=1, index=max_indices))


    total_sum = mu_max.sum()

    matrix_mu = mu_max / total_sum
    return matrix_mu

def get_ranked_features(shap_values, columns):
    # Get the mean absolute SHAP values for each feature
    shap_sum = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame([columns, shap_sum.tolist()]).T
    importance_df.columns = ['feature', 'shap_importance']
    importance_df = importance_df.sort_values('shap_importance', ascending=False)

    sorted_features = importance_df['feature'].to_list()
    return sorted_features

def accuracy_performance_benchmark(
        model,
        X_train,
        X_test, 
        y_test,
        baseline_drop_columns,
        jp_drop_columns
    ):
    X_test_baseline = X_test.copy()
    X_test_jp = X_test.copy()
    min_val = X_train.min()
    max_val = X_train.max()
    for col in baseline_drop_columns:
        X_test_baseline[col] = np.random.uniform(min_val[col], max_val[col], size=len(X_test_baseline))
    for col in jp_drop_columns:
        X_test_jp[col] = np.random.uniform(min_val[col], max_val[col], size=len(X_test_jp))

    X_test_baseline_tensor = torch.FloatTensor(X_test_baseline.values)
    X_test_jp_tensor = torch.FloatTensor(X_test_jp.values)
    y_test_tensor = torch.FloatTensor(y_test.values).view(-1, 1)

    results = {}
    accuracies = []

    for X_test_tensor in [X_test_baseline_tensor, X_test_jp_tensor]:
        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)

            # Convert outputs to binary using 0.5 as threshold
            y_pred_tensor = (test_outputs > 0.5).float()
            correct_predictions = (y_pred_tensor == y_test_tensor).float().sum()
            accuracy = correct_predictions / y_test_tensor.shape[0]

        accuracies.append(accuracy.item())

    results['Baseline acc'] = accuracies[0]
    results['JointProb acc'] = accuracies[1]

    return results


def compute_kl_divergence(y_s, y_t):
    try:
        kde_s = gaussian_kde(y_s)
        kde_t = gaussian_kde(y_t)
        y_min = min(y_s.min(), y_t.min())
        y_max = max(y_s.max(), y_t.max())
        y = np.linspace(y_min, y_max, 1000)

        kl_div = entropy(kde_s(y), kde_t(y))
    except:
        kl_div = np.inf

    return kl_div

def gaussian_kernel(x, y, sigma=1.0):
    """Compute the Gaussian kernel between x and y"""
    return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * sigma ** 2))

def compute_mmd(y_s, y_t, kernel=gaussian_kernel):
    n = y_s.shape[0]
    m = y_t.shape[0]

    ss = np.sum([kernel(y_s[i], y_s[j]) for i in range(n) for j in range(n)])
    tt = np.sum([kernel(y_t[i], y_t[j]) for i in range(m) for j in range(m)])
    st = np.sum([kernel(y_s[i], y_t[j]) for i in range(n) for j in range(m)])

    return ss / (n**2) + tt / (m**2) - 2 * st / (n*m)


def counterfactual_ability_performance_benchmarking(
        model,
        df_baseline,
        df_explain,
        y_baseline,
        baseline_change_columns,
        jp_change_columns,
        delta=0.05,
):
    X_explain_bs = df_explain.copy()
    X_explain_jp = df_explain.copy()

    for col in baseline_change_columns:
        X_explain_bs[col] = df_baseline[col]
    for col in jp_change_columns:
        X_explain_jp[col] = df_baseline[col]

    X_explain_bs_tensor = torch.FloatTensor(X_explain_bs.values)
    X_explain_jp_tensor = torch.FloatTensor(X_explain_jp.values)

    y_baseline_tensor = torch.FloatTensor(y_baseline)

    ot_list = []
    kl_list = []
    mmd_list = []
    results = {}

    for X_explain_tensor in [X_explain_bs_tensor, X_explain_jp_tensor]:
        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            y_explain_tensor = torch.FloatTensor(model.predict_proba(X_explain_tensor))
            ot, _ = WassersteinDivergence().distance(y_explain_tensor, y_baseline_tensor, delta=delta)
            kl = compute_kl_divergence(
                y_explain_tensor.detach().numpy(), 
                y_baseline_tensor.detach().numpy(),
            )
            mmd = compute_mmd(
                y_explain_tensor.detach().numpy(), 
                y_baseline_tensor.detach().numpy(),
            )

            ot_list.append(ot)
            kl_list.append(kl)            
            mmd_list.append(mmd)

    results['OT_bs'] = ot_list[0]
    results['OT_jp'] = ot_list[1]

    results['KL_bs'] = kl_list[0]
    results['KL_jp'] = kl_list[1]

    results['MMD_bs'] = mmd_list[0]
    results['MMD_jp'] = mmd_list[1]

    return results