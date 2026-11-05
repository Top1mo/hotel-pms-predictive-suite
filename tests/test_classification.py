import pytest

from src.classification.model import classification_metrics


def test_classification_metrics_report_binary_metrics_and_confusion_matrix():
    metrics = classification_metrics(
        [0, 0, 1, 1],
        [0, 1, 1, 1],
        [0.1, 0.8, 0.7, 0.9],
    )

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(0.8)
    assert metrics["roc_auc"] == pytest.approx(0.75)
    assert metrics["pr_auc"] == pytest.approx(5 / 6)
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]


def test_classification_metrics_reject_single_class_evaluation_data():
    with pytest.raises(ValueError, match="both binary classes"):
        classification_metrics([0, 0], [0, 0], [0.1, 0.2])
