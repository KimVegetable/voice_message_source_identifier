from __future__ import annotations

import numpy as np
import pandas as pd


def same_value_filter(frame: pd.DataFrame, threshold: float = 0.70) -> list[str]:
    kept: list[str] = []
    n_rows = len(frame)
    if n_rows == 0:
        return kept
    for column in frame.columns:
        values = frame[column].to_numpy()
        _, counts = np.unique(values, return_counts=True)
        if counts.max() / n_rows <= threshold:
            kept.append(column)
    return kept


def minmax_fit_transform(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train = np.asarray(train, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)
    minimum = np.nanmin(train, axis=0)
    maximum = np.nanmax(train, axis=0)
    span = maximum - minimum
    span[span == 0] = 1.0
    return (train - minimum) / span, (test - minimum) / span

