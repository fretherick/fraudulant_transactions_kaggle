import numpy as np
import pandas as pd

from fraud_detection.modeling import sample_training_data


def test_sampling_keeps_all_fraud_and_corrects_legitimate_weight() -> None:
    features = pd.DataFrame({"feature": np.arange(100)})
    target = pd.Series([1] * 5 + [0] * 95)

    sampled_features, sampled_target, weights = sample_training_data(
        features,
        target,
        max_rows=20,
        random_state=42,
    )

    assert len(sampled_features) == 20
    assert sampled_target.sum() == 5
    assert set(target.index[target == 1]).issubset(sampled_features.index)
    assert np.all(weights[sampled_target.to_numpy() == 1] == 1)
    assert np.allclose(weights[sampled_target.to_numpy() == 0], 95 / 15)
