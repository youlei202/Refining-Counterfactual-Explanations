from dataset import CompasDataset
from experiments import Benchmarking
from utils.logger_config import setup_logger
from models.wrapper import PYTORCH_MODELS
import warnings
from dataset import dataset_loader

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

    name = "compas"
    dataset_ares = dataset_loader(
        name, data_path="data/", dropped_features=[], n_bins=None
    )

    dataset = CompasDataset(dataset_ares=dataset_ares)
    input_dim = dataset.get_dataframe().shape[1] - 1
    seed = 1
    torch.manual_seed(seed)
    Avalues_method = "avg"

    counterfactual_algorithms = [
        'DiCE',
        'DisCount',
        # "GlobeCE",
        # 'AReS',
        'KNN',
    ]

    experiment = Benchmarking(
        dataset=dataset,
        models=[
            # (BaggingClassifier(), "sklearn"),
            # (GaussianProcessClassifier(),'sklearn'),
            (XGBClassifier(), 'sklearn'),
            (PyTorchLogisticRegression(input_dim=input_dim), 'PYT'),
            (PyTorchDNN(input_dim=input_dim), 'PYT'),
            (PyTorchRBFNet(input_dim=input_dim, hidden_dim=input_dim), 'PYT'),
            (PyTorchLinearSVM(input_dim=input_dim), 'PYT'),
            (RandomForestClassifier(), 'sklearn'),
            (GradientBoostingClassifier(), 'sklearn'),
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
    sample_num = 100
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
                print(f"Function {function_name} is not defined.")

    logger.info("\n\n------Compute Shapley Values------")
    experiment.compute_intervention_policies(
        model_counterfactuals=model_counterfactuals,
        Avalues_method=Avalues_method,
    )

    logger.info("\n\n------Evaluating Distance Performance Under Interventions------")
    experiment.evaluate_distance_performance_under_interventions(
        intervention_num_list=range(0, 101, 5), trials_num=20, replace=False
    )

    # plotting.intervention_vs_distance(experiment, save_to_file=False)

    with open(f"pickles/{dataset.name}_experiment.pickle", "wb") as output_file:
        pickle.dump(experiment, output_file)


if __name__ == "__main__":
    logger.info("Compas analysis started")
    main()
