from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC, SVC

from .jin_features import minmax_fit_transform, same_value_filter


@dataclass(frozen=True)
class JinFoldResult:
    predictions: np.ndarray
    selected_columns: list[str]
    selected_portion: float
    selected_c: float = 1.0
    selected_gamma: str | float = "auto"
    validation_score: float = 0.0


def _rank_columns(
    x_train: np.ndarray,
    y_train: np.ndarray,
    columns: Sequence[str],
    random_state: int,
) -> list[str]:
    if len(columns) <= 1:
        return list(columns)
    estimator = LinearSVC(C=1.0, dual="auto", max_iter=20000, random_state=random_state)
    selector = RFE(estimator, n_features_to_select=1, step=0.1)
    selector.fit(x_train, y_train)
    ranked = sorted(zip(columns, selector.ranking_), key=lambda item: item[1])
    return [column for column, _ in ranked]


def _portion_columns(columns: Sequence[str], portion: float) -> list[str]:
    if not columns:
        raise ValueError("no columns available after feature filtering")
    n_selected = max(1, int(round(len(columns) * portion)))
    return list(columns[:n_selected])


def _score_ranked_portion(
    x_train_raw: np.ndarray,
    labels: np.ndarray,
    ranked_idx: Sequence[int],
    portion: float,
    random_state: int,
    inner_cv_splits: int,
) -> float:
    if inner_cv_splits <= 1:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    if counts.min() < inner_cv_splits:
        return 0.0
    selected_idx = list(ranked_idx[: max(1, int(round(len(ranked_idx) * portion)))])
    scores: list[float] = []
    splitter = StratifiedKFold(n_splits=inner_cv_splits, shuffle=True, random_state=random_state)
    for inner_train_idx, inner_val_idx in splitter.split(np.zeros(len(labels)), labels):
        x_inner_train, x_inner_val = minmax_fit_transform(
            x_train_raw[inner_train_idx][:, selected_idx],
            x_train_raw[inner_val_idx][:, selected_idx],
        )
        classifier = SVC(kernel="rbf", C=1.0, gamma="auto", decision_function_shape="ovr")
        classifier.fit(x_inner_train, labels[inner_train_idx])
        pred = classifier.predict(x_inner_val)
        scores.append(float(np.mean(pred == labels[inner_val_idx])))
    return float(np.mean(scores))


def _score_ranked_hyperparameters(
    x_train_raw: np.ndarray,
    labels: np.ndarray,
    ranked_idx: Sequence[int],
    portion: float,
    c_value: float,
    gamma: str | float,
    random_state: int,
    inner_cv_splits: int,
) -> float:
    if inner_cv_splits <= 1:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    if counts.min() < inner_cv_splits:
        return 0.0
    selected_idx = list(ranked_idx[: max(1, int(round(len(ranked_idx) * portion)))])
    scores: list[float] = []
    splitter = StratifiedKFold(n_splits=inner_cv_splits, shuffle=True, random_state=random_state)
    for inner_train_idx, inner_val_idx in splitter.split(np.zeros(len(labels)), labels):
        x_inner_train, x_inner_val = minmax_fit_transform(
            x_train_raw[inner_train_idx][:, selected_idx],
            x_train_raw[inner_val_idx][:, selected_idx],
        )
        classifier = SVC(kernel="rbf", C=float(c_value), gamma=gamma, decision_function_shape="ovr")
        classifier.fit(x_inner_train, labels[inner_train_idx])
        pred = classifier.predict(x_inner_val)
        scores.append(float(np.mean(pred == labels[inner_val_idx])))
    return float(np.mean(scores))


def fit_predict_jin_svm(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: Sequence[str],
    portions: Sequence[float] = tuple(i / 10.0 for i in range(1, 11)),
    threshold: float = 0.70,
    inner_cv_splits: int = 3,
    random_state: int = 42,
) -> JinFoldResult:
    kept = same_value_filter(train.loc[:, feature_columns], threshold=threshold)
    if not kept:
        raise ValueError("all features were removed by Jin same-value filtering")

    x_train_raw = train.loc[:, kept].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
    x_test = test.loc[:, kept].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
    x_train, x_test = minmax_fit_transform(x_train_raw, x_test)
    y_train = train["label"].to_numpy()

    ranked_columns = _rank_columns(x_train, y_train, kept, random_state=random_state)
    ranked_idx = [kept.index(column) for column in ranked_columns]
    if len(portions) == 1 or inner_cv_splits <= 1:
        selected_portion = float(portions[0])
    else:
        scores = [
            (
                _score_ranked_portion(
                    x_train_raw,
                    y_train,
                    ranked_idx,
                    portion,
                    random_state,
                    inner_cv_splits,
                ),
                float(portion),
            )
            for portion in portions
        ]
        selected_portion = max(scores, key=lambda item: (item[0], -item[1]))[1]

    selected_columns = _portion_columns(ranked_columns, selected_portion)
    selected_idx = [kept.index(column) for column in selected_columns]
    classifier = SVC(kernel="rbf", C=1.0, gamma="auto", decision_function_shape="ovr")
    classifier.fit(x_train[:, selected_idx], y_train)
    predictions = classifier.predict(x_test[:, selected_idx])
    return JinFoldResult(
        predictions=predictions,
        selected_columns=selected_columns,
        selected_portion=selected_portion,
    )


def fit_predict_jin_tuned_svm(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: Sequence[str],
    portions: Sequence[float] = tuple(i / 10.0 for i in range(1, 11)),
    c_values: Sequence[float] = (0.1, 1.0, 10.0),
    gamma_values: Sequence[str | float] = ("scale", "auto", 0.01, 0.1),
    threshold: float = 0.70,
    inner_cv_splits: int = 3,
    random_state: int = 42,
) -> JinFoldResult:
    kept = same_value_filter(train.loc[:, feature_columns], threshold=threshold)
    if not kept:
        raise ValueError("all features were removed by Jin same-value filtering")

    x_train_raw = train.loc[:, kept].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
    x_test_raw = test.loc[:, kept].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
    x_train, x_test = minmax_fit_transform(x_train_raw, x_test_raw)
    y_train = train["label"].to_numpy()

    ranked_columns = _rank_columns(x_train, y_train, kept, random_state=random_state)
    ranked_idx = [kept.index(column) for column in ranked_columns]
    candidates: list[tuple[float, float, str | float, float]] = []
    for portion in portions:
        for c_value in c_values:
            for gamma in gamma_values:
                score = _score_ranked_hyperparameters(
                    x_train_raw,
                    y_train,
                    ranked_idx,
                    float(portion),
                    float(c_value),
                    gamma,
                    random_state,
                    inner_cv_splits,
                )
                candidates.append((score, float(portion), gamma, float(c_value)))
    best_score, selected_portion, selected_gamma, selected_c = max(
        candidates,
        key=lambda item: (item[0], item[1], -item[3]),
    )

    selected_columns = _portion_columns(ranked_columns, selected_portion)
    selected_idx = [kept.index(column) for column in selected_columns]
    classifier = SVC(kernel="rbf", C=selected_c, gamma=selected_gamma, decision_function_shape="ovr")
    classifier.fit(x_train[:, selected_idx], y_train)
    predictions = classifier.predict(x_test[:, selected_idx])
    return JinFoldResult(
        predictions=predictions,
        selected_columns=selected_columns,
        selected_portion=selected_portion,
        selected_c=selected_c,
        selected_gamma=selected_gamma,
        validation_score=float(best_score),
    )
