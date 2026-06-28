from __future__ import annotations

import math

import numpy as np
from scipy.fftpack import dct


def hz_to_mel(freq: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(freq) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def framing(signal: np.ndarray, sample_rate: int, frame_ms: float = 25.0, hop_ms: float = 10.0) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError("signal must be one dimensional")
    frame_len = max(1, int(round(sample_rate * frame_ms / 1000.0)))
    hop_len = max(1, int(round(sample_rate * hop_ms / 1000.0)))
    if signal.size < frame_len:
        signal = np.pad(signal, (0, frame_len - signal.size))
    n_frames = 1 + int(math.ceil((signal.size - frame_len) / hop_len))
    pad_len = (n_frames - 1) * hop_len + frame_len
    if pad_len > signal.size:
        signal = np.pad(signal, (0, pad_len - signal.size))
    strides = (signal.strides[0] * hop_len, signal.strides[0])
    return np.lib.stride_tricks.as_strided(signal, shape=(n_frames, frame_len), strides=strides).copy()


def mel_filterbank(sample_rate: int, n_fft: int, n_filters: int = 26) -> np.ndarray:
    low_mel = hz_to_mel(0.0)
    high_mel = hz_to_mel(sample_rate / 2.0)
    mel_points = np.linspace(low_mel, high_mel, n_filters + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    filters = np.zeros((n_filters, n_fft // 2 + 1), dtype=np.float64)
    for m in range(1, n_filters + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        if center == left:
            center += 1
        if right == center:
            right += 1
        for k in range(left, min(center, filters.shape[1])):
            filters[m - 1, k] = (k - left) / (center - left)
        for k in range(center, min(right, filters.shape[1])):
            filters[m - 1, k] = (right - k) / (right - center)
    return filters


def delta(features: np.ndarray, width: int = 2) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    denom = 2.0 * sum(i * i for i in range(1, width + 1))
    padded = np.pad(features, ((width, width), (0, 0)), mode="edge")
    out = np.zeros_like(features)
    for t in range(features.shape[0]):
        for n in range(1, width + 1):
            out[t] += n * (padded[t + width + n] - padded[t + width - n])
    return out / denom


def mfcc(signal: np.ndarray, sample_rate: int, n_mfcc: int = 13) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64)
    if signal.size == 0:
        signal = np.zeros(1, dtype=np.float64)
    emphasized = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])
    frames = framing(emphasized, sample_rate)
    frames *= np.hamming(frames.shape[1])
    n_fft = 1
    while n_fft < frames.shape[1]:
        n_fft *= 2
    n_fft = max(n_fft, 512)
    spectrum = np.fft.rfft(frames, n=n_fft)
    power = (np.abs(spectrum) ** 2) / n_fft
    energies = np.dot(power, mel_filterbank(sample_rate, n_fft).T)
    log_energies = np.log(np.maximum(energies, 1e-12))
    coeffs = dct(log_energies, type=2, axis=1, norm="ortho")[:, :n_mfcc]
    return coeffs


def mfcc_with_deltas(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    base = mfcc(signal, sample_rate, n_mfcc=13)
    first = delta(base)
    second = delta(first)
    return np.hstack([base, first, second])

