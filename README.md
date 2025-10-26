# Contrastive Feature Attribution Experiments

This repository contains the experiment code that accompanies the preprint\
*L. You ✉, Y. Bian, and L. Cao, "Refining Counterfactual Explanations With Joint-Distribution-Informed Shapley Towards Actionable Minimality", preprint*.\
It operationalises joint-distribution-informed Shapley attributions to refine counterfactual explanations and evaluate actionable policies across several credit-risk benchmarks. Feel free to cite the paper if this work is useful in your research.

> Looking for a production-ready library? Check out the companion package **COLA** \
> (<https://github.com/understanding-ml/COLA>), which provides a maintained implementation of the intervention policy solver used in these experiments.

## Highlights
- Integrated pipeline for loading tabular risk assessment datasets (COMPAS, German Credit, HELOC, Hotel Bookings) and standardising features.
- Model zoo covering scikit-learn ensembles, gradient boosters, Gaussian processes, and PyTorch networks via a common `ClassifierWrapper`.
- Counterfactual generation with AReS, DiCE, DisCount, GLOBE-CE, and KNN-based methods (`experiments/counterfactual.py` & `explainers/`).
- Joint-distribution-aware Shapley computations and COLA-based policy optimisation to study the trade-off between feasibility and target satisfaction (`experiments/policy.py`).
- Automated benchmarking loop with distance metrics (optimal transport, mean/median differences, MMD) and intervention rollouts, with results stored in `pickles/` and plots in `pictures/`.
- Reproducible scripts for batch execution on LSF clusters (`scripts/*.sh`) alongside dataset-specific run files in `runs/`.

## Repository Structure
- `dataset/` – dataset loaders and preprocessing utilities derived from GLOBE-CE.
- `models/` – scikit-learn and PyTorch classifiers plus a lightweight wrapper API.
- `explainers/` – implementations and helpers for the counterfactual explainers used in the study.
- `experiments/` – orchestration code: benchmarking loop, policy construction, evaluation metrics, LaTeX/table helpers, plotting routines.
- `runs/` – entry points for each dataset experiment (e.g. `python -m runs.compas`).
- `scripts/` – LSF batch scripts mirroring the module set-up used in the paper’s compute environment.
- `data/` – cached copies of the public datasets to avoid repeated downloads.
- `pickles/` & `pictures/` – saved experiment artefacts (serialized pipelines, figures).
- Jupyter notebooks (`*.ipynb`) – exploratory analysis, ablation summaries, and figure generation.

## Getting Started
1. Create a Python 3.9 environment (matches the cluster modules used in `scripts/*.sh`).
2. Install the core dependencies. The experiments rely on
   `numpy`, `pandas`, `scikit-learn`, `torch`, `lightgbm`, `xgboost`, `shap`,
   `dice-ml`, `POT (ot)`, `tqdm`, `gurobipy`, `matplotlib`, and `scipy`.
   Install Gurobi and obtain a license if you plan to run the optimal subset baseline (`experiments/baseline.py`).
3. (Optional) Install `cem` if you wish to use the Contrastive Explanation Method backend invoked in `experiments/policy.py`.
4. Ensure the public datasets you plan to use are available under `data/` (or let `dataset/data_loader.py` fetch them automatically).

```bash
conda create -n cfa python=3.9
conda activate cfa
pip install numpy pandas scikit-learn torch lightgbm xgboost shap dice-ml pot tqdm gurobipy matplotlib scipy
# Add any backend-specific extras such as cem or GPU-enabled torch builds as needed.
```

## Running Experiments
- Local run: execute a dataset entry point, then inspect the returned `Benchmarking` object saved under `pickles/`.
  ```bash
  python -m runs.compas
  ```
- Cluster run: adapt the `scripts/*.sh` templates and submit via `bsub`. Logs will be written to `data/logs/`.
- Post-processing: use `experiments/plotting.py` or the dataset notebooks to reproduce the figures reported in the manuscript.

Saved artefacts (`pickles/*.pickle`) bundle trained models, counterfactuals, policies, and evaluation metrics to simplify downstream analysis without recomputation.

## Citation
If you build on this codebase, please cite the accompanying preprint:

```
@article{you2025jointshapley,
  title   = {Refining Counterfactual Explanations With Joint-Distribution-Informed Shapley Towards Actionable Minimality},
  author  = {You, Liyan and Bian, Yatao and Cao, Longbing},
  year    = {2025},
  note    = {Preprint},
  url     = {https://github.com/understanding-ml/Contrastive-Feature-Attribution}
}
```

## Related Software
- **COLA: Counterfactual Linear Assignment** – <https://github.com/understanding-ml/COLA>. This standalone package provides the production version of the COLA policy optimisation routine referenced throughout `experiments/policy.py`.
- The counterfactual baselines draw on implementations from AReS, GLOBE-CE, DiCE, DisCount, and related repositories; please consult their respective licences when extending this work.

## License & Acknowledgements
The original dataset loader is adapted from the GLOBE-CE project (see headers in `dataset/data_loader.py`).\
Third-party solvers, especially Gurobi, require separate licences.\
If you encounter issues or have suggestions for extending the benchmarking suite, feel free to open an issue or submit a pull request.

