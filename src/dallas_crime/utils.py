"""Shared numeric helpers used across pipeline and acquisition layers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _coerce_numeric(values: pd.Series) -> pd.Series:
    """Coerce a series to numeric, replacing unparseable values with NaN."""
    return pd.to_numeric(values, errors="coerce")


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Element-wise division that returns NaN where the denominator is non-positive."""
    numerator_num = _coerce_numeric(numerator)
    denominator_num = _coerce_numeric(denominator)
    result = np.where(denominator_num > 0, numerator_num / denominator_num, np.nan)
    return pd.Series(result, index=numerator_num.index, dtype="float64")
