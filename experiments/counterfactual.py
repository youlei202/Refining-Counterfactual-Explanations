import dice_ml
from explainers import dce
import pandas as pd


def compute_counterfactuals(X_test, model, target_name, algorithm):
    if algorithm == "DiCE":
        return compute_DiCE_counterfactuals(X_test, model, target_name)
    elif algorithm == "DisCount":
        return compute_DisCount_counterfactuals(X_test, model, target_name)
    else:
        raise NotImplementedError


def compute_DiCE_counterfactuals(df_factual, model, target_name):

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
        desired_class="opposite",
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

    return {"X": X_counterfactual, "y": y_counterfactual}


def compute_DisCount_counterfactuals(X_test, model, target_name):
    raise NotImplementedError
