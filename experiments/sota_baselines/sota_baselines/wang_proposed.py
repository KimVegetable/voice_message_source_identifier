from __future__ import annotations

from dataclasses import asdict, dataclass
from collections.abc import Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class WangProposedConfig:
    sample_rate: int = 16000
    segment_samples: int = 16000
    max_segments_per_file: int = 4
    n_sinc_filters: int = 40
    sinc_kernel_size: int = 251
    sinc_stride: int = 1
    conv_channels: int = 32
    second_conv_channels: int | None = None
    first_conv_kernel_size: int = 7
    second_conv_kernel_size: int = 5
    hidden_dim: int = 128
    dnn_layers: int = 1
    dropout: float = 0.2
    learning_rate: float = 1e-3
    batch_size: int = 64
    tune_epochs: int = 6
    final_epochs: int = 14
    aggregation: str = "mean"
    optimizer: str = "adam"


@dataclass(frozen=True)
class WangProposedResult:
    predictions: np.ndarray
    selected_config: WangProposedConfig
    validation_score: float


class SincConv1d(nn.Module):
    def __init__(
        self,
        out_channels: int,
        kernel_size: int,
        sample_rate: int,
        stride: int = 1,
        min_low_hz: float = 30.0,
        min_band_hz: float = 50.0,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("SincConv1d kernel_size must be odd")
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.sample_rate = int(sample_rate)
        self.stride = int(stride)
        self.min_low_hz = float(min_low_hz)
        self.min_band_hz = float(min_band_hz)

        nyquist = self.sample_rate / 2.0
        low = np.linspace(self.min_low_hz, max(self.min_low_hz + 1.0, nyquist - 300.0), self.out_channels)
        band = np.full(self.out_channels, max(self.min_band_hz, (nyquist - self.min_low_hz) / self.out_channels))
        self.low_hz_ = nn.Parameter(torch.tensor(low, dtype=torch.float32))
        self.band_hz_ = nn.Parameter(torch.tensor(band, dtype=torch.float32))

        half = (self.kernel_size - 1) // 2
        n = torch.arange(-half, half + 1, dtype=torch.float32)
        window = 0.54 - 0.46 * torch.cos(2 * torch.pi * torch.arange(self.kernel_size, dtype=torch.float32) / self.kernel_size)
        self.register_buffer("n_", n)
        self.register_buffer("window_", window)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_), max=self.sample_rate / 2.0 - 1.0)
        low = torch.clamp(low, max=self.sample_rate / 2.0 - self.min_band_hz - 1.0)
        low_norm = low / self.sample_rate
        high_norm = high / self.sample_rate

        n = self.n_[None, :]
        low_pass1 = 2 * low_norm[:, None] * torch.sinc(2 * low_norm[:, None] * n)
        low_pass2 = 2 * high_norm[:, None] * torch.sinc(2 * high_norm[:, None] * n)
        band_pass = (low_pass2 - low_pass1) * self.window_[None, :]
        band_pass = band_pass / (band_pass.abs().sum(dim=1, keepdim=True) + 1e-8)
        filters = band_pass[:, None, :]
        return torch.nn.functional.conv1d(x, filters, stride=self.stride)


class WangSincNet(nn.Module):
    def __init__(self, n_classes: int, config: WangProposedConfig):
        super().__init__()
        second_conv_channels = config.second_conv_channels or config.conv_channels * 2
        self.sinc = SincConv1d(
            out_channels=config.n_sinc_filters,
            kernel_size=config.sinc_kernel_size,
            sample_rate=config.sample_rate,
            stride=config.sinc_stride,
        )
        self.features = nn.Sequential(
            nn.BatchNorm1d(config.n_sinc_filters),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(4),
            nn.Conv1d(
                config.n_sinc_filters,
                config.conv_channels,
                kernel_size=config.first_conv_kernel_size,
                padding=config.first_conv_kernel_size // 2,
            ),
            nn.BatchNorm1d(config.conv_channels),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(4),
            nn.Conv1d(
                config.conv_channels,
                second_conv_channels,
                kernel_size=config.second_conv_kernel_size,
                padding=config.second_conv_kernel_size // 2,
            ),
            nn.BatchNorm1d(second_conv_channels),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(1),
        )
        classifier_layers: list[nn.Module] = [nn.Flatten()]
        in_features = second_conv_channels
        for _ in range(config.dnn_layers):
            classifier_layers.extend(
                [
                    nn.Dropout(config.dropout),
                    nn.Linear(in_features, config.hidden_dim),
                    nn.LeakyReLU(0.2),
                ]
            )
            in_features = config.hidden_dim
        classifier_layers.extend([nn.Dropout(config.dropout), nn.Linear(in_features, n_classes)])
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sinc(x)
        x = torch.abs(x)
        x = self.features(x)
        return self.classifier(x)


def normalize_waveform(waveform: np.ndarray) -> np.ndarray:
    signal = np.asarray(waveform, dtype=np.float32)
    if signal.size == 0:
        return np.zeros(1, dtype=np.float32)
    signal = signal - float(signal.mean())
    std = float(signal.std())
    if std > 1e-8:
        signal = signal / std
    return signal.astype(np.float32)


def make_fixed_segments(waveform: np.ndarray, segment_samples: int, max_segments: int) -> np.ndarray:
    signal = normalize_waveform(waveform)
    if signal.size <= segment_samples:
        padded = np.zeros(segment_samples, dtype=np.float32)
        padded[: signal.size] = signal
        return padded[None, :]
    n_segments = min(int(max_segments), int(np.ceil(signal.size / segment_samples)))
    starts = np.linspace(0, signal.size - segment_samples, n_segments).round().astype(int)
    return np.stack([signal[start : start + segment_samples] for start in starts]).astype(np.float32)


def aggregate_segment_probabilities(probabilities: np.ndarray, classes: np.ndarray, mode: str = "mean") -> str:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[0] == 0:
        raise ValueError("probabilities must have shape (segments, classes)")
    if mode == "vote":
        votes = np.bincount(probabilities.argmax(axis=1), minlength=probabilities.shape[1])
        winners = np.flatnonzero(votes == votes.max())
        if len(winners) == 1:
            return str(classes[int(winners[0])])
    mean_prob = probabilities.mean(axis=0)
    return str(classes[int(mean_prob.argmax())])


def config_to_dict(config: WangProposedConfig) -> dict[str, int | float | str]:
    return asdict(config)


def wang_config_grid(profile: str, sample_rate: int) -> tuple[WangProposedConfig, ...]:
    if profile == "smoke":
        return (
            WangProposedConfig(
                sample_rate=sample_rate,
                segment_samples=sample_rate,
                max_segments_per_file=2,
                n_sinc_filters=16,
                conv_channels=16,
                hidden_dim=64,
                tune_epochs=2,
                final_epochs=4,
            ),
        )
    if profile == "broad":
        return (
            WangProposedConfig(sample_rate=sample_rate, segment_samples=sample_rate, max_segments_per_file=4, n_sinc_filters=40, conv_channels=32, hidden_dim=128, dropout=0.2, learning_rate=1e-3, tune_epochs=8, final_epochs=20),
            WangProposedConfig(sample_rate=sample_rate, segment_samples=sample_rate * 2, max_segments_per_file=4, n_sinc_filters=40, conv_channels=32, hidden_dim=128, dropout=0.2, learning_rate=1e-3, tune_epochs=8, final_epochs=20),
            WangProposedConfig(sample_rate=sample_rate, segment_samples=sample_rate, max_segments_per_file=6, n_sinc_filters=80, conv_channels=48, hidden_dim=192, dropout=0.3, learning_rate=5e-4, tune_epochs=8, final_epochs=20),
            WangProposedConfig(sample_rate=sample_rate, segment_samples=sample_rate * 2, max_segments_per_file=6, n_sinc_filters=80, conv_channels=48, hidden_dim=192, dropout=0.3, learning_rate=5e-4, tune_epochs=8, final_epochs=20),
        )
    if profile == "paper":
        return (
            WangProposedConfig(
                sample_rate=sample_rate,
                segment_samples=sample_rate * 3,
                max_segments_per_file=6,
                n_sinc_filters=80,
                sinc_kernel_size=251,
                sinc_stride=16,
                conv_channels=60,
                second_conv_channels=60,
                first_conv_kernel_size=5,
                second_conv_kernel_size=5,
                hidden_dim=2048,
                dnn_layers=3,
                dropout=0.2,
                learning_rate=1e-3,
                batch_size=64,
                tune_epochs=30,
                final_epochs=90,
                aggregation="vote",
                optimizer="sgd",
            ),
        )
    return (
        WangProposedConfig(sample_rate=sample_rate, segment_samples=sample_rate, max_segments_per_file=2, n_sinc_filters=24, conv_channels=24, hidden_dim=96, dropout=0.2, learning_rate=1e-3, batch_size=128, tune_epochs=4, final_epochs=12),
        WangProposedConfig(sample_rate=sample_rate, segment_samples=sample_rate, max_segments_per_file=2, n_sinc_filters=32, conv_channels=32, hidden_dim=128, dropout=0.3, learning_rate=5e-4, batch_size=128, tune_epochs=4, final_epochs=12),
    )


def _segments_and_index(
    waveforms: Sequence[np.ndarray],
    labels: np.ndarray | None,
    label_encoder: LabelEncoder | None,
    config: WangProposedConfig,
) -> tuple[np.ndarray, np.ndarray | None, list[slice]]:
    segment_arrays: list[np.ndarray] = []
    encoded_labels: list[int] = []
    slices: list[slice] = []
    cursor = 0
    for idx, waveform in enumerate(waveforms):
        segments = make_fixed_segments(waveform, config.segment_samples, config.max_segments_per_file)
        segment_arrays.append(segments)
        next_cursor = cursor + len(segments)
        slices.append(slice(cursor, next_cursor))
        cursor = next_cursor
        if labels is not None and label_encoder is not None:
            encoded = int(label_encoder.transform([labels[idx]])[0])
            encoded_labels.extend([encoded] * len(segments))
    x = np.concatenate(segment_arrays, axis=0)[:, None, :].astype(np.float32)
    y = np.asarray(encoded_labels, dtype=np.int64) if labels is not None else None
    return x, y, slices


def _train_model(
    train_waveforms: Sequence[np.ndarray],
    train_labels: np.ndarray,
    config: WangProposedConfig,
    label_encoder: LabelEncoder,
    random_state: int,
    epochs: int,
) -> WangSincNet:
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    x_train, y_train, _ = _segments_and_index(train_waveforms, train_labels, label_encoder, config)
    dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    generator = torch.Generator().manual_seed(random_state)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, generator=generator)
    model = WangSincNet(n_classes=len(label_encoder.classes_), config=config)
    if config.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        scheduler.step()
    return model


def _predict_probabilities(
    model: WangSincNet,
    waveforms: Sequence[np.ndarray],
    config: WangProposedConfig,
) -> list[np.ndarray]:
    x_test, _, slices = _segments_and_index(waveforms, None, None, config)
    model.eval()
    probs: list[np.ndarray] = []
    with torch.no_grad():
        outputs: list[np.ndarray] = []
        for start in range(0, len(x_test), config.batch_size):
            batch = torch.tensor(x_test[start : start + config.batch_size], dtype=torch.float32)
            logits = model(batch)
            outputs.append(torch.softmax(logits, dim=1).cpu().numpy())
    all_probs = np.concatenate(outputs, axis=0)
    for segment_slice in slices:
        probs.append(all_probs[segment_slice])
    return probs


def _predict_labels(
    model: WangSincNet,
    waveforms: Sequence[np.ndarray],
    config: WangProposedConfig,
    classes: np.ndarray,
) -> np.ndarray:
    file_probs = _predict_probabilities(model, waveforms, config)
    return np.asarray(
        [aggregate_segment_probabilities(probs, classes=classes, mode=config.aggregation) for probs in file_probs],
        dtype=object,
    )


def _train_validation_split(labels: np.ndarray, random_state: int) -> tuple[np.ndarray, np.ndarray] | None:
    _, counts = np.unique(labels, return_counts=True)
    if counts.min() < 2:
        return None
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    train_idx, val_idx = next(splitter.split(np.zeros(len(labels)), labels))
    return train_idx, val_idx


def fit_predict_wang_proposed(
    train_waveforms: Sequence[np.ndarray],
    train_labels: np.ndarray,
    test_waveforms: Sequence[np.ndarray],
    configs: Sequence[WangProposedConfig],
    random_state: int = 42,
) -> WangProposedResult:
    if not configs:
        raise ValueError("at least one WangProposedConfig is required")
    label_encoder = LabelEncoder()
    label_encoder.fit(train_labels)
    split = None if len(configs) == 1 else _train_validation_split(train_labels, random_state=random_state)
    if split is None:
        selected_config = configs[0]
        validation_score = 0.0
    else:
        inner_train_idx, val_idx = split
        inner_train_waveforms = [train_waveforms[i] for i in inner_train_idx]
        val_waveforms = [train_waveforms[i] for i in val_idx]
        inner_labels = train_labels[inner_train_idx]
        val_labels = train_labels[val_idx]
        scores: list[tuple[float, int, WangProposedConfig]] = []
        for order, config in enumerate(configs):
            model = _train_model(
                inner_train_waveforms,
                inner_labels,
                config,
                label_encoder,
                random_state=random_state + order,
                epochs=config.tune_epochs,
            )
            pred = _predict_labels(model, val_waveforms, config, label_encoder.classes_)
            scores.append((float(accuracy_score(val_labels, pred)), -order, config))
        validation_score, _, selected_config = max(scores, key=lambda item: (item[0], item[1]))
    final_model = _train_model(
        train_waveforms,
        train_labels,
        selected_config,
        label_encoder,
        random_state=random_state + 10_000,
        epochs=selected_config.final_epochs,
    )
    predictions = _predict_labels(final_model, test_waveforms, selected_config, label_encoder.classes_)
    return WangProposedResult(
        predictions=predictions,
        selected_config=selected_config,
        validation_score=float(validation_score),
    )
