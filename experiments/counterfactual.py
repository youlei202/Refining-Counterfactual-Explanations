import dice_ml
from explainers import dce
import pandas as pd
import torch
import numpy as np

FACTUAL_CLASS = 1


def get_factual_indices(X_test, model, target_name, sample_num):
    X_test_ext = X_test.copy()
    X_test_ext[target_name] = model.predict_proba(X_test.values)

    sampling_weights = np.exp(X_test_ext[target_name].values.clip(min=0) * 10)
    indices = (X_test_ext.sample(sample_num, weights=sampling_weights)).index

    return indices


def compute_DiCE_counterfactuals(X_test, model, target_name, sample_num):

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
    y_counterfactual = (
        pd.concat(dice_df_list).reset_index(drop=True)[target_name].values
    )

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
    y_counterfactual = discount_explainer.best_y.detach().numpy().flatten()

    return {
        "X_factual": df_factual.values,
        "y_factual": df_factual_ext[target_name].values,
        "X": X_counterfactual,
        "y": y_counterfactual,
    }
