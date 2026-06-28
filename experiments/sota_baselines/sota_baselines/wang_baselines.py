from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def _stack_frames(feature_sequences: Sequence[np.ndarray]) -> np.ndarray:
    frames = [np.asarray(item, dtype=np.float64) for item in feature_sequences if len(item) > 0]
    if not frames:
        raise ValueError("at least one non-empty feature sequence is required")
    return np.vstack(frames)


def train_ubm(
    feature_sequences: Sequence[np.ndarray],
    n_components: int = 64,
    random_state: int = 42,
    max_iter: int = 100,
) -> GaussianMixture:
    all_frames = _stack_frames(feature_sequences)
    n_components = min(n_components, len(all_frames))
    ubm = GaussianMixture(
        n_components=n_components,
        covariance_type="diag",
        reg_covar=1e-6,
        max_iter=max_iter,
        random_state=random_state,
    )
    ubm.fit(all_frames)
    return ubm


def map_adapted_means(
    ubm: GaussianMixture,
    frames: np.ndarray,
    relevance_factor: float = 16.0,
) -> np.ndarray:
    frames = np.asarray(frames, dtype=np.float64)
    if frames.ndim != 2:
        raise ValueError("frames must be a two dimensional array")
    responsibilities = ubm.predict_proba(frames)
    counts = responsibilities.sum(axis=0)
    weighted_sum = responsibilities.T @ frames
    observed_means = np.divide(
        weighted_sum,
        counts[:, None],
        out=np.array(ubm.means_, copy=True),
        where=counts[:, None] > 0,
    )
    alpha = counts / (counts + relevance_factor)
    return alpha[:, None] * observed_means + (1.0 - alpha[:, None]) * ubm.means_


def map_adapted_weights(
    ubm: GaussianMixture,
    frames: np.ndarray,
    relevance_factor: float = 16.0,
) -> np.ndarray:
    responsibilities = ubm.predict_proba(np.asarray(frames, dtype=np.float64))
    counts = responsibilities.sum(axis=0)
    adapted = counts + relevance_factor * ubm.weights_
    total = adapted.sum()
    if total == 0:
        return np.array(ubm.weights_, copy=True)
    return adapted / total


def map_adapted_gsv(
    ubm: GaussianMixture,
    frames: np.ndarray,
    relevance_factor: float = 16.0,
) -> np.ndarray:
    return map_adapted_means(ubm, frames, relevance_factor=relevance_factor).ravel()


def _score_with_adapted_means(
    ubm: GaussianMixture,
    adapted_means: np.ndarray,
    frames: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    frames = np.asarray(frames, dtype=np.float64)
    covariances = np.asarray(ubm.covariances_, dtype=np.float64)
    model_weights = ubm.weights_ if weights is None else weights
    log_weights = np.log(np.maximum(model_weights, 1e-300))
    dim = frames.shape[1]
    log_det = np.sum(np.log(covariances), axis=1)
    diff = frames[:, None, :] - adapted_means[None, :, :]
    mahal = np.sum((diff * diff) / covariances[None, :, :], axis=2)
    log_prob = -0.5 * (dim * np.log(2.0 * np.pi) + log_det[None, :] + mahal)
    return float(np.mean(logsumexp(log_prob + log_weights[None, :], axis=1)))


def fit_predict_gmm_ubm(
    train_features: Sequence[np.ndarray],
    train_labels: np.ndarray,
    test_features: Sequence[np.ndarray],
    n_components: int = 64,
    relevance_factor: float = 16.0,
    random_state: int = 42,
) -> np.ndarray:
    train_labels = np.asarray(train_labels)
    ubm = train_ubm(train_features, n_components=n_components, random_state=random_state)
    class_models: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for label in sorted(np.unique(train_labels)):
        class_frames = _stack_frames([frames for frames, y in zip(train_features, train_labels) if y == label])
        class_models[str(label)] = (
            map_adapted_means(ubm, class_frames, relevance_factor=relevance_factor),
            map_adapted_weights(ubm, class_frames, relevance_factor=relevance_factor),
        )

    predictions: list[str] = []
    for frames in test_features:
        scores = {
            label: _score_with_adapted_means(ubm, means, frames, weights=weights)
            for label, (means, weights) in class_models.items()
        }
        predictions.append(max(scores.items(), key=lambda item: item[1])[0])
    return np.asarray(predictions)


def fit_predict_gsv_svm(
    train_features: Sequence[np.ndarray],
    train_labels: np.ndarray,
    test_features: Sequence[np.ndarray],
    n_components: int = 64,
    relevance_factor: float = 16.0,
    random_state: int = 42,
) -> np.ndarray:
    train_labels = np.asarray(train_labels)
    ubm = train_ubm(train_features, n_components=n_components, random_state=random_state)
    x_train = np.vstack([map_adapted_gsv(ubm, frames, relevance_factor) for frames in train_features])
    x_test = np.vstack([map_adapted_gsv(ubm, frames, relevance_factor) for frames in test_features])
    classifier = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=1.0, gamma="auto", decision_function_shape="ovr"),
    )
    classifier.fit(x_train, train_labels)
    return classifier.predict(x_test)
