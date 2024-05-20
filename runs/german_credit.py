from dataset import GermanCreditDataset
from experiments import Benchmarking
from utils.logger_config import setup_logger
from models.wrapper import PYTORCH_MODELS
import warnings

from experiments.counterfactual import *
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    BaggingClassifier,
)
from sklearn.svm import SVC
from models import (
    PyTorchDNN,
    PyTorchLinearSVM,
    PyTorchRBFNet,
    PyTorchLogisticRegression,
)
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from sklearn.gaussian_process import GaussianProcessClassifier
from experiments import plotting
import pickle

warnings.filterwarnings("ignore")
logger = setup_logger()


def main():
    dataset = GermanCreditDataset()
    input_dim = dataset.get_dataframe().shape[1] - 1
    seed = 0
    torch.manual_seed(seed)
    Avalues_method = "avg"

    counterfactual_algorithms = [
        "DiCE",
        "DisCount",
        "KNN",
    ]

    experiment = Benchmarking(
        dataset=dataset,
        models=[
            # (BaggingClassifier(), "sklearn"),
            (XGBClassifier(), 'sklearn'),
            (GaussianProcessClassifier(), "sklearn"),
            (PyTorchLogisticRegression(input_dim=input_dim), "PYT"),
            (PyTorchDNN(input_dim=input_dim), "PYT"),
            (PyTorchRBFNet(input_dim=input_dim, hidden_dim=input_dim), "PYT"),
            (PyTorchLinearSVM(input_dim=input_dim), "PYT"),
            (RandomForestClassifier(), "sklearn"),
            (GradientBoostingClassifier(), "sklearn"),
            (AdaBoostClassifier(), "sklearn"),
        ],
        shapley_methods=[
            "Train_Distri",
            "Train_OTMatch",
            "CF_UniformMatch",
            "CF_RandomMatch",
            "CF_OTMatch",
        ],
        distance_metrics=[
            "optimal_transport",
            "mean_difference",
            "median_difference",
            "max_mean_discrepancy",
        ],
        md_baseline=False,
    )

    experiment.train_and_evaluate_models(random_state=seed)
    experiment.models_performance()

    logger.info("\n\n------Compute Counterfactuals------")
    sample_num = 50
    model_counterfactuals = {}
    for model, model_name in zip(experiment.models, experiment.model_names):
        model_counterfactuals[model_name] = {}

        for algorithm in counterfactual_algorithms:
            if algorithm == "DisCount" and model_name not in PYTORCH_MODELS:
                logger.info(
                    f"Skipping {algorithm} for {model_name} due to incompatability"
                )
                continue
            logger.info(f"Computing {model_name} counterfactuals with {algorithm}")
            function_name = f"compute_{algorithm}_counterfactuals"
            try:
                func = globals()[function_name]
                model_counterfactuals[model_name][algorithm] = func(
                    experiment.X_test,
                    model=model,
                    target_name=experiment.dataset.target_name,
                    sample_num=sample_num,
                    experiment=experiment,
                )
            except KeyError:
                logger.info(f"Function {function_name} is not working.")

    logger.info("\n\n------Compute Action Policies------")
    experiment.compute_intervention_policies(
        model_counterfactuals=model_counterfactuals,
        Avalues_method = Avalues_method,
    )

    logger.info("\n\n------Evaluating Distance Performance Under Interventions------")
    experiment.evaluate_distance_performance_under_interventions(
        intervention_num_list=range(0,201,5),
        trials_num=100,
        replace=False,
    )

    # plotting.intervention_vs_distance(experiment, save_to_file=False)

    with open(f"pickles/{dataset.name}_experiment.pickle", "wb") as output_file:
        pickle.dump(experiment, output_file)


if __name__ == "__main__":
    logger.info("German credit analysis started")
    main()
