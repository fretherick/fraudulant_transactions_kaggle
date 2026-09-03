import pandas as pd
import pytest

from fraud_detection.data import MODEL_COLUMNS, temporal_split


def test_model_columns_exclude_post_transaction_balances() -> None:
    assert "newbalanceOrig" not in MODEL_COLUMNS
    assert "newbalanceDest" not in MODEL_COLUMNS


def test_temporal_split_preserves_time_order() -> None:
    frame = pd.DataFrame(
        {
            "step": [step for step in range(1, 11) for _ in range(2)],
            "isFraud": [label for _ in range(10) for label in (0, 1)],
        }
    )

    split = temporal_split(frame)

    assert split.train["step"].max() < split.validation["step"].min()
    assert split.validation["step"].max() < split.test["step"].min()
    assert len(split.train) + len(split.validation) + len(split.test) == len(frame)


def test_temporal_split_rejects_invalid_fractions() -> None:
    frame = pd.DataFrame({"step": [1, 2, 3], "isFraud": [0, 1, 0]})
    with pytest.raises(ValueError, match="sum to less than one"):
        temporal_split(frame, train_fraction=0.8, validation_fraction=0.2)
