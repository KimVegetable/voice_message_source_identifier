from __future__ import annotations

import subprocess
from zipfile import BadZipFile
from pathlib import Path

import numpy as np

from .mfcc import mfcc_with_deltas


def decode_audio_ffmpeg(path: Path, sample_rate: int = 16000) -> np.ndarray:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "-",
    ]
    completed = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    audio = np.frombuffer(completed.stdout, dtype=np.float32).astype(np.float64)
    if audio.size == 0:
        return np.zeros(sample_rate // 10, dtype=np.float64)
    return audio


def load_or_extract_mfcc(
    audio_path: Path,
    cache_path: Path,
    sample_rate: int = 16000,
) -> np.ndarray:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        try:
            with np.load(cache_path) as data:
                return data["mfcc"]
        except (EOFError, BadZipFile, OSError, ValueError, KeyError):
            cache_path.unlink(missing_ok=True)
    signal = decode_audio_ffmpeg(audio_path, sample_rate=sample_rate)
    features = mfcc_with_deltas(signal, sample_rate)
    np.savez_compressed(cache_path, mfcc=features)
    return features


def load_or_extract_waveform(
    audio_path: Path,
    cache_path: Path,
    sample_rate: int = 16000,
) -> np.ndarray:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        try:
            with np.load(cache_path) as data:
                return data["waveform"]
        except (EOFError, BadZipFile, OSError, ValueError, KeyError):
            cache_path.unlink(missing_ok=True)
    signal = decode_audio_ffmpeg(audio_path, sample_rate=sample_rate).astype(np.float32)
    np.savez_compressed(cache_path, waveform=signal)
    return signal
