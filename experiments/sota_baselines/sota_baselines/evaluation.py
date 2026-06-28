from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold


@dataclass(frozen=True)
class FoldResult:
    fold: int
    group: str
    accuracy: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    test_size: int


def leave_one_group_splits(labels: np.ndarray, groups: np.ndarray) -> Iterable[tuple[str, np.ndarray, np.ndarray]]:
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    for group in sorted(np.unique(groups)):
        test_idx = np.flatnonzero(groups == group)
        train_idx = np.flatnonzero(groups != group)
        yield str(group), train_idx, test_idx


def stratified_kfold_splits(
    labels: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> Iterable[tuple[str, np.ndarray, np.ndarray]]:
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    dummy = np.zeros(len(labels))
    for fold, (train_idx, test_idx) in enumerate(splitter.split(dummy, labels), start=1):
        yield f"fold{fold}", train_idx, test_idx


def weighted_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
    }

