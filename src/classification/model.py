"""Logistic-regression training and evaluation for booking cancellation."""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_validate

from src.data.prepare import ROOT, load_partition
from src.data.splits import training_cv
from src.preprocessing.pipelines import make_model_pipeline, task_data

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}


def classification_metrics(y_true, y_pred, y_score):
    """Return binary classification metrics with cancellation as the positive class."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score = np.asarray(y_score)
    if not np.isin(y_true, [0, 1]).all() or set(np.unique(y_true)) != {0, 1}:
        raise ValueError("Evaluation data must contain both binary classes 0 and 1.")

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "confusion_matrix_labels": ["not_canceled", "canceled"],
    }


def train_and_evaluate(output_dir=None, n_splits=5):
    """Run grouped CV on train, fit train+validation, and evaluate once on test."""
    partitions = {name: load_partition(name) for name in ("train", "validation", "test")}
    train = partitions["train"]
    X_train, y_train = task_data(train, "classification")
    cv, groups = training_cv(train, n_splits=n_splits)
    estimator = LogisticRegression(max_iter=1000, random_state=42)
    pipeline = make_model_pipeline("classification", estimator)
    cv_results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        groups=groups,
        scoring=SCORING,
        error_score="raise",
        n_jobs=1,
    )
    cross_validation = {
        metric: {
            "fold_scores": [float(value) for value in cv_results[f"test_{metric}"]],
            "mean": float(np.mean(cv_results[f"test_{metric}"])),
            "std": float(np.std(cv_results[f"test_{metric}"], ddof=1)),
        }
        for metric in SCORING
    }

    fit_data = pd.concat([partitions["train"], partitions["validation"]])
    X_fit, y_fit = task_data(fit_data, "classification")
    X_test, y_test = task_data(partitions["test"], "classification")
    pipeline.fit(X_fit, y_fit)
    test_metrics = classification_metrics(
        y_test,
        pipeline.predict(X_test),
        pipeline.predict_proba(X_test)[:, 1],
    )

    report = {
        "task": "hotel booking cancellation prediction",
        "positive_class": {"label": 1, "meaning": "canceled"},
        "decision_threshold": 0.5,
        "cross_validation": {
            "strategy": f"{n_splits}-fold GroupKFold on training partition",
            "n_splits": n_splits,
            "training_rows": len(train),
            "training_groups": int(groups.nunique()),
            "metrics": cross_validation,
        },
        "final_fit": {
            "rows": len(fit_data),
            "partitions": ["train", "validation"],
            "estimator": "LogisticRegression",
            "max_iter": estimator.max_iter,
        },
        "final_test": {
            "rows": len(partitions["test"]),
            "metrics": test_metrics,
        },
        "metric_notes": {
            "precision": "Of bookings predicted canceled, the fraction actually canceled.",
            "recall": "Of bookings actually canceled, the fraction detected.",
            "f1": "Harmonic mean of precision and recall at the 0.5 decision threshold.",
            "roc_auc": "Threshold-independent area under the ROC curve.",
            "pr_auc": "Average precision, the area-under-precision-recall summary; useful when classes are imbalanced.",
            "confusion_matrix": "Rows are actual [not_canceled, canceled]; columns are predicted [not_canceled, canceled].",
        },
    }

    destination = Path(output_dir) if output_dir else ROOT / "outputs" / "classification"
    destination.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, destination / "logistic_regression.joblib")
    (destination / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        test_metrics["confusion_matrix"],
        index=test_metrics["confusion_matrix_labels"],
        columns=test_metrics["confusion_matrix_labels"],
    ).to_csv(destination / "confusion_matrix.csv")
    return report
