"""Feature engineering for PaySim transactions."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRANSACTION_TYPES = ("CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER")


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create stable authorization-time features without account memorization."""
    result = pd.DataFrame(index=frame.index)

    step = frame["step"].astype("float32")
    amount = frame["amount"].astype("float32")
    old_origin = frame["oldbalanceOrg"].astype("float32")
    old_destination = frame["oldbalanceDest"].astype("float32")

    result["step"] = step
    result["hour"] = ((step - 1) % 24).astype("float32")
    result["day"] = ((step - 1) // 24).astype("float32")
    result["log_amount"] = np.log1p(amount).astype("float32")
    result["log_oldbalance_org"] = np.log1p(old_origin).astype("float32")
    result["log_oldbalance_dest"] = np.log1p(old_destination).astype("float32")

    transaction_type = frame["type"].astype("string")
    for category in TRANSACTION_TYPES:
        result[f"type_{category.lower()}"] = (transaction_type == category).astype("float32")

    return result
