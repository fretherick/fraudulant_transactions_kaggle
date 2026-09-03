import pandas as pd

from fraud_detection.features import build_features


def test_build_features_uses_only_pre_transaction_values_and_stable_types() -> None:
    frame = pd.DataFrame(
        {
            "step": [25],
            "type": ["TRANSFER"],
            "amount": [75.0],
            "oldbalanceOrg": [100.0],
            "oldbalanceDest": [10.0],
        }
    )

    result = build_features(frame)

    assert result.loc[0, "hour"] == 0
    assert result.loc[0, "day"] == 1
    assert result.loc[0, "type_transfer"] == 1
    assert result.loc[0, "type_payment"] == 0
    assert len(result.columns) == 11
    assert not any("newbalance" in column for column in result.columns)
