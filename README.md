# CSE382: City Hotel (H2) Booking Demand ML System

Guide for the team to get the repo running and start contributing.

## Team Members

Fill in your details here so contributions can be tracked:

| Name | Student ID | GitHub | 
| :--- | :--- | :--- | :--- |
| Mostafa |24p0353 | @Top1mo | 
|  | | |
|  | | |  
| | | | 
|  | | |  

## Quick Setup

```bash
git clone https://github.com/Top1mo/hotel-pms-predictive-suite.git
cd hotel-pms-predictive-suite
git checkout dev

python -m venv .venv

# Activate (pick ONE for your shell)
source .venv/bin/activate        # Bash/Zsh (Linux/macOS)
source .venv/bin/activate.fish   # Fish
.venv\Scripts\activate           # Windows (cmd/PowerShell)

pip install -r requirements.txt
python src/data/download.py
```

If the last step prints `Verified H2.csv: ... rows, ... columns.`, the raw dataset is ready.

To generate the cleaned dataset (`data/cleaned/H2_cleaned.csv`), open and run `src/data/Data_cleaning.ipynb`.

## About the data

This snapshot contains raw and historical prepared CSV files. `src/data/download.py`
downloads `H2.csv` into the path set in `config.yaml` and checks the shape
against `expected_rows` / `expected_cols`. If the check fails, the file is
corrupted or wrong: delete it and run the script again.

Generated splits are ignored by Git. Keep the raw source unchanged; regenerate preparation outputs using the commands below.

## Config

All paths and data settings live in `config.yaml`. Don't hardcode paths in
your code, read them from the config.

## Project structure

```text
hotel-pms-predictive-suite/
├── README.md
├── Report.md
├── requirements.txt
├── config.yaml
├── main.py
├── app.py
├── data/
│   ├── raw/
│   └── cleaned/
├── src/
│   ├── data/
│   ├── preprocessing/
│   ├── features/
│   ├── regression/
│   ├── classification/
│   ├── clustering/
│   ├── tuning/
│   ├── evaluation/
│   └── inference/
├── artifacts/
└── outputs/
```

## Workflow

1. Pull the latest `dev`: `git checkout dev && git pull`
2. Make your own branch: `git checkout -b feature/<short-name>`
3. Commit small and often, with clear messages
4. Push and open a PR into `dev`: `git push -u origin feature/<short-name>`
5. Get one teammate to review before merging
6. Don't push directly to `dev`

## Adding dependencies

If you install a new package, add it to `requirements.txt` in the same PR
so everyone's environment stays in sync.

## Troubleshooting

- **`ModuleNotFoundError`**: venv isn't activated, or you forgot `pip install -r requirements.txt`
- **Shape mismatch on download**: delete the CSV and rerun `download.py`
- **Download fails**: check your internet and that the `url` in `config.yaml` is still valid

## Submission info

- **Repository URL:** `https://github.com/Top1mo/hotel-pms-predictive-suite/tree/dev`
- **Submitted Commit SHA:** `<INSERT_SHA_BEFORE_ZIP>`


## Reproducible preparation and model handoff

The authoritative modeling flow starts from raw H2, normalizes missing markers, removes known zero-guest rows and exact full-row duplicates, then creates grouped train/validation/test splits. Historical cleaned/processed CSVs and the cleaning notebook are not inputs to this flow.

```bash
python -m pip install -r requirements-person3-tested.txt
python main.py prepare
python main.py check-preprocessing
python main.py analyze
python -m pytest -q
```

The tested environment is Python 3.12; `requirements-person3.txt` provides broader version ranges. The original requirements file includes dependencies for later milestones.

Splits and original row IDs are written to `data/splits`. Fractions 60/20/20 apply to groups, so row counts are approximate. Final test data is reserved for final supervised evaluation. Training-only EDA is generated in `outputs/eda_training`; old full-dataset figures must not be mixed with those results. Preparation metadata and checks are in `outputs/person3`.

Model owners should load a partition, select task data, and attach their estimator to the full preprocessing pipeline:

```python
from src.data.prepare import load_partition
from src.data.splits import training_cv
from src.preprocessing.pipelines import task_data, make_model_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

train = load_partition('train')
X, y = task_data(train, 'classification')
model = make_model_pipeline('classification', LogisticRegression(max_iter=1000))
cv, groups = training_cv(train)
results = cross_validate(model, X, y, cv=cv, groups=groups,
                         scoring=['accuracy', 'f1', 'roc_auc'])
```

For regression use task `regression` and a regressor; for clustering use `clustering` and a clustering estimator (`y` is None). These are usage examples, not completed experiments. Do not prefit preprocessing before cross-validation. Tree models can use `scale=False`. Outputs are sparse; an estimator requiring dense arrays needs a deliberate adapter.

## Cancellation classification

Prepare the grouped partitions, then train and evaluate the logistic-regression classifier:

```bash
python main.py prepare
python -m src.classification.train
```

The model predicts `IsCanceled` (`1` means canceled) using the existing training-only preprocessing pipeline. Five-fold `GroupKFold` cross-validation is reported on the training partition. The final model is fitted on train plus validation, then evaluated once on the held-out test partition. The test set is not used for preprocessing, cross-validation, or model selection.

Metrics and the final-test confusion matrix are saved in `outputs/classification/metrics.json` and `outputs/classification/confusion_matrix.csv`; the fitted pipeline is saved as `outputs/classification/logistic_regression.joblib`.

Metric guide: **accuracy** is the fraction of all predictions correct; **precision** is the fraction of predicted cancellations that were cancellations; **recall** is the fraction of cancellations detected; **F1** balances precision and recall; the **confusion matrix** counts actual versus predicted classes; **ROC-AUC** summarizes ranking across classification thresholds; **PR-AUC** is reported as average precision and emphasizes performance on the cancellation class. Cross-validation reports each fold's score and the mean and standard deviation; final-test metrics estimate performance on unseen bookings.
