from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def sequence_to_image(sequence: np.ndarray, max_frames: int = 256) -> np.ndarray:
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim != 2 or sequence.shape[1] != 39:
        raise ValueError("MFCC sequence must have shape (frames, 39)")
    image = np.zeros((max_frames, 39), dtype=np.float32)
    n = min(max_frames, sequence.shape[0])
    image[:n, :] = sequence[:n, :]
    return image[None, :, :]


class MfccCnn(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Linear(64 * 4 * 4, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)


def _images_from_sequences(sequences: Sequence[np.ndarray], max_frames: int) -> np.ndarray:
    return np.stack([sequence_to_image(sequence, max_frames=max_frames) for sequence in sequences])


def fit_predict_mfcc_cnn(
    train_features: Sequence[np.ndarray],
    train_labels: np.ndarray,
    test_features: Sequence[np.ndarray],
    max_frames: int = 256,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    random_state: int = 42,
) -> np.ndarray:
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_labels)

    x_train = _images_from_sequences(train_features, max_frames=max_frames)
    x_test = _images_from_sequences(test_features, max_frames=max_frames)
    mean = x_train.mean(axis=(0, 2, 3), keepdims=True)
    std = x_train.std(axis=(0, 2, 3), keepdims=True)
    std[std == 0] = 1.0
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    train_dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    generator = torch.Generator().manual_seed(random_state)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    model = MfccCnn(n_classes=len(label_encoder.classes_))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x_test), batch_size):
            batch = torch.tensor(x_test[start : start + batch_size], dtype=torch.float32)
            predictions.append(model(batch).argmax(dim=1).cpu().numpy())
    encoded = np.concatenate(predictions)
    return label_encoder.inverse_transform(encoded)

