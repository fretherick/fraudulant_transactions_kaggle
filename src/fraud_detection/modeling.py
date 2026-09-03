"""Model definitions and memory-aware sampling."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def sample_training_data(
    features: pd.DataFrame,
    target: pd.Series,
    max_rows: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Keep every fraud example and sample legitimate rows with correction weights."""
    if len(features) <= max_rows:
        return features, target, np.ones(len(features), dtype="float64")

    positive_index = target.index[target == 1]
    negative_index = target.index[target == 0]
    negative_sample_size = max_rows - len(positive_index)
    if negative_sample_size <= 0:
        raise ValueError("max_rows must exceed the number of fraudulent training rows.")

    rng = np.random.default_rng(random_state)
    sampled_negative_index = rng.choice(
        negative_index.to_numpy(), size=negative_sample_size, replace=False
    )
    sampled_index = np.concatenate([positive_index.to_numpy(), sampled_negative_index])
    rng.shuffle(sampled_index)

    sampled_features = features.loc[sampled_index]
    sampled_target = target.loc[sampled_index]
    weights = np.ones(len(sampled_target), dtype="float64")
    negative_weight = len(negative_index) / negative_sample_size
    weights[sampled_target.to_numpy() == 0] = negative_weight
    return sampled_features, sampled_target, weights


def build_models(random_state: int) -> dict[str, object]:
    """Return an interpretable baseline and the primary nonlinear model."""
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1_000, random_state=random_state),
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=250,
            max_leaf_nodes=31,
            min_samples_leaf=30,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=random_state,
        ),
    }
