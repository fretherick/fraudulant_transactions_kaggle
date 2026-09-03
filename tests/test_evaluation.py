import numpy as np

from fraud_detection.evaluation import classification_metrics, select_threshold


def test_metrics_report_confusion_matrix_counts() -> None:
    target = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.7, 0.8, 0.9])

    result = classification_metrics(target, scores, threshold=0.75)

    assert result["true_negatives"] == 2
    assert result["false_positives"] == 0
    assert result["false_negatives"] == 0
    assert result["true_positives"] == 2


def test_threshold_selection_returns_observed_score() -> None:
    target = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.6, 0.9])

    threshold = select_threshold(target, scores)

    assert threshold in scores
