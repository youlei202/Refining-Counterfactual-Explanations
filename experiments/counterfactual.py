import pandas as pd
import torch
import numpy as np

from explainers import dce
from explainers.globe_ce import GLOBE_CE
from explainers.ares import AReS
from explainers.knn import KNNCounterfactuals
from explainers.knn import EfficientQuantileTransformer
import dice_ml


FACTUAL_CLASS = 1


def get_factual_indices(X_test, model, target_name, sample_num):
    X_test_ext = X_test.copy()
    X_test_ext[target_name] = model.predict(X_test.values)

    sampling_weights = np.exp(X_test_ext[target_name].values.clip(min=0) * 4)
    indices = (X_test_ext.sample(sample_num, weights=sampling_weights)).index

    return indices


def compute_DiCE_counterfactuals(
    X_test, model, target_name, sample_num, experiment=None
):

    indices = get_factual_indices(X_test, model, target_name, sample_num)
    df_factual = X_test.loc[indices]

    df_factual_ext = df_factual.copy()
    df_factual_ext[target_name] = model.predict(df_factual.values)

    # Prepare for DiCE
    dice_model = dice_ml.Model(model=model, backend=model.backend)
    dice_features = df_factual.columns.to_list()
    dice_data = dice_ml.Data(
        dataframe=df_factual_ext,
        continuous_features=dice_features,
        outcome_name=target_name,
    )
    dice_explainer = dice_ml.Dice(dice_data, dice_model)
    dice_results = dice_explainer.generate_counterfactuals(
        query_instances=df_factual,
        features_to_vary=dice_features,
        desired_class=1 - FACTUAL_CLASS,
        total_CFs=1,
    )

    # Iterate through each result and append to the DataFrame
    dice_df_list = []
    for cf in dice_results.cf_examples_list:
        # Convert to DataFrame and append
        cf_df = cf.final_cfs_df
        dice_df_list.append(cf_df)

    df_counterfactual = (
        pd.concat(dice_df_list).reset_index(drop=True).drop(target_name, axis=1)
    )
    X_counterfactual = df_counterfactual.values

    y_counterfactual = model.predict(X_counterfactual)
    return {
        "X_factual": df_factual.values,
        "y_factual": df_factual_ext[target_name].values,
        "X": X_counterfactual,
        "y": y_counterfactual,
    }


def compute_DisCount_counterfactuals(
    X_test,
    model,
    target_name,
    sample_num,
    experiment=None,
    lr=1e-1,
    n_proj=10,
    delta=0.05,
    U_1=0.4,
    U_2=0.3,
    l=0.2,
    r=1,
    max_iter=50,
    tau=1e3,
):
    indices = get_factual_indices(X_test, model, target_name, sample_num)
    df_factual = X_test.loc[indices]
    df_factual_ext = df_factual.copy()
    df_factual_ext[target_name] = model.predict(df_factual.values)
    y_target = torch.FloatTensor(
        [1 - FACTUAL_CLASS for _ in range(df_factual.shape[0])]
    )

    discount_explainer = dce.DistributionalCounterfactualExplainer(
        model=model,
        df_X=df_factual,
        explain_columns=df_factual.columns,
        y_target=y_target,
        lr=lr,
        n_proj=n_proj,
        delta=delta,
    )

    discount_explainer.optimize(U_1=U_1, U_2=U_2, l=l, r=r, max_iter=max_iter, tau=tau)
    df_counterfactual = pd.DataFrame(
        discount_explainer.best_X.detach().numpy(),
        columns=discount_explainer.explain_columns,
        index=df_factual.index,
    )
    X_counterfactual = df_counterfactual.values
    y_counterfactual = model.predict(X_counterfactual)

    return {
        "X_factual": df_factual.values,
        "y_factual": df_factual_ext[target_name].values,
        "X": X_counterfactual,
        "y": y_counterfactual,
    }


def compute_GlobeCE_counterfactuals(
    X_test,
    model,
    target_name,
    sample_num,
    experiment,
):
    indices = get_factual_indices(X_test, model, target_name, sample_num)
    df_factual = X_test.loc[indices]

    ares = AReS(
        model=model,
        dataset=experiment.dataset.dataset_ares,
        X=df_factual,
        n_bins=10,
        normalise=None,
    )  # 1MB
    bin_widths = ares.bin_widths

    globe_ce = GLOBE_CE(
        model=model,
        dataset=experiment.dataset.dataset_ares,
        X=df_factual,
        affected_subgroup=None,
        dropped_features=[],
        ordinal_features=[],
        delta_init="zeros",
        normalise=None,
        bin_widths=bin_widths,
        monotonicity=None,
        p=1,
    )
    globe_ce.sample(
        n_sample=sample_num,
        magnitude=2,
        sparsity_power=1,  # magnitude is the fixed cost sampled at
        idxs=None,
        n_features=df_factual.shape[1],
        disable_tqdm=False,  # 2 random features chosen at each sample, no sparsity smoothing (p=1)
        plot=False,
        seed=None,
        scheme="random",
        dropped_features=[],
    )
    globe_ce.select_n_deltas(n_div=3)

    X_counterfactual = (
        globe_ce.round_categorical(globe_ce.x_aff + globe_ce.best_delta)
        if globe_ce.n_categorical
        else globe_ce.x_aff + globe_ce.best_delta
    )
    df_counterfactual = pd.DataFrame(X_counterfactual, columns=X_test.columns)

    final_sample_num = min(df_factual.shape[0], df_counterfactual.shape[0])

    X_factual = df_factual.sample(final_sample_num).values
    X_counterfactual = df_counterfactual.sample(final_sample_num).values

    y_factual = model.predict(X_factual)
    y_counterfactual = model.predict(X_counterfactual)

    return {
        "X_factual": X_factual,
        "y_factual": y_factual,
        "X": X_counterfactual,
        "y": y_counterfactual,
    }


def compute_AReS_counterfactuals(
    X_test,
    model,
    target_name,
    sample_num,
    experiment,
):
    indices = get_factual_indices(X_test, model, target_name, sample_num)
    df_factual = X_test.loc[indices]

    ares = AReS(
        model=model,
        dataset=experiment.dataset.dataset_ares,
        X=df_factual,
        n_bins=10,
        normalise=None,
    )
    ares.generate_itemsets(
        apriori_threshold=0.2,
        max_width=None,  # defaults to e2-1
        affected_subgroup=None,
        save_copy=False,
    )
    # Note: progress bar initial time estimate about 10 times too large
    ares.generate_groundset(
        max_width=None, RL_reduction=True, then_generation=None, save_copy=False
    )
    lams = [1, 0]  # can play around with these lambda values
    ares.evaluate_groundset(
        lams=lams, r=200, save_mode=1, disable_tqdm=False, plot_accuracy=False
    )
    ares.select_groundset(s=200)
    ares.optimise_groundset(lams=lams, factor=1, print_updates=False, print_terms=False)

    df_counterfactual = pd.DataFrame(ares.R.cfx_matrix[0], columns=X_test.columns)
    X_counterfactual = df_counterfactual.values
    y_counterfactual = model.predict(X_counterfactual)

    final_sample_num = min(df_factual.shape[0], df_counterfactual.shape[0])

    X_factual = df_factual.sample(final_sample_num).values
    y_factual = model.predict(X_factual)

    return {
        "X_factual": X_factual,
        "y_factual": y_factual,
        "X": X_counterfactual,
        "y": y_counterfactual,
    }


def compute_KNN_counterfactuals(
    X_test, model, target_name, sample_num, experiment, n_neighbors=50
):
    indices = get_factual_indices(X_test, model, target_name, sample_num)
    df_factual = X_test.loc[indices]

    X_train, X_test = experiment.X_train, experiment.X_test

    scaler = EfficientQuantileTransformer()
    scaler.fit(X_train)

    knn_explainer = KNNCounterfactuals(
        model=model,
        X=X_train.values,
        n_neighbors=n_neighbors,
        distance="cityblock",
        scaler=scaler,
        max_samples=10000,
    )

    estimated = knn_explainer.get_multiple_counterfactuals(df_factual.values)

    df_counterfactual = pd.DataFrame(
        np.array(estimated).reshape(sample_num * n_neighbors, X_train.shape[1]),
        columns=X_train.columns,
    )

    final_sample_num = min(df_factual.shape[0], df_counterfactual.shape[0])
    X_factual = df_factual.sample(final_sample_num).values
    X_counterfactual = df_counterfactual.sample(final_sample_num).values

    y_factual = model.predict(X_factual)
    y_counterfactual = model.predict(X_counterfactual)

    return {
        "X_factual": X_factual,
        "y_factual": y_factual,
        "X": X_counterfactual,
        "y": y_counterfactual,
    }
