"""Dataset loading, validation, and temporal splitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

TARGET = "isFraud"
FLAG_COLUMN = "isFlaggedFraud"
# New balances are deliberately excluded: they are outcomes of the transaction.
MODEL_COLUMNS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    TARGET,
    FLAG_COLUMN,
]

DTYPES = {
    "step": "int16",
    "type": "category",
    "amount": "float32",
    "oldbalanceOrg": "float32",
    "oldbalanceDest": "float32",
    TARGET: "uint8",
    FLAG_COLUMN: "uint8",
}


@dataclass(frozen=True)
class TemporalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    train_end_step: int
    validation_end_step: int


def load_transactions(path: str | Path, max_rows: int | None = None) -> pd.DataFrame:
    """Load only columns used by the project, with memory-efficient dtypes."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Run `fraud-download` first.")

    frame = pd.read_csv(
        csv_path,
        usecols=MODEL_COLUMNS,
        dtype=DTYPES,
        nrows=max_rows,
    )
    _validate(frame)
    return frame


def temporal_split(
    frame: pd.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> TemporalSplit:
    """Split on whole time steps so later transactions never inform earlier ones."""
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("Split fractions must be positive.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must sum to less than one.")

    steps = sorted(frame["step"].unique())
    if len(steps) < 3:
        raise ValueError("At least three distinct time steps are required.")

    train_idx = max(0, int(len(steps) * train_fraction) - 1)
    validation_idx = max(
        train_idx + 1,
        int(len(steps) * (train_fraction + validation_fraction)) - 1,
    )
    validation_idx = min(validation_idx, len(steps) - 2)
    train_end = int(steps[train_idx])
    validation_end = int(steps[validation_idx])

    train = frame.loc[frame["step"] <= train_end].copy()
    validation = frame.loc[(frame["step"] > train_end) & (frame["step"] <= validation_end)].copy()
    test = frame.loc[frame["step"] > validation_end].copy()

    for name, partition in {
        "train": train,
        "validation": validation,
        "test": test,
    }.items():
        if partition.empty or partition[TARGET].nunique() < 2:
            raise ValueError(f"The {name} split must contain both target classes.")

    return TemporalSplit(train, validation, test, train_end, validation_end)


def _validate(frame: pd.DataFrame) -> None:
    missing_columns = sorted(set(MODEL_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    if frame.empty:
        raise ValueError("Dataset is empty.")
    if frame[MODEL_COLUMNS].isna().any().any():
        raise ValueError("Required model columns contain missing values.")
    invalid_labels = set(frame[TARGET].unique()) - {0, 1}
    if invalid_labels:
        raise ValueError(f"Target must be binary; found {sorted(invalid_labels)}")
