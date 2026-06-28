from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import pandas as pd


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]


def _env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, default))


DEFAULT_METADATA = _env_path("ASI_METADATA", EXPERIMENT_ROOT / "inputs" / "sample_metadata_from_filename.csv")
DEFAULT_JIN_FEATURE_DIR = _env_path("ASI_JIN_FEATURE_DIR", EXPERIMENT_ROOT / "inputs" / "preprocessing_20250227_225033")
DEFAULT_PROPOSED_MATRIX = _env_path("ASI_PROPOSED_MATRIX", EXPERIMENT_ROOT / "inputs" / "feature_matrix_used_columns.csv")
DEFAULT_AUDIO_ROOT = _env_path("ASI_AUDIO_ROOT", Path("G:/audio_dataset"))


JIN_FEATURE_FILES = [
    "cb.csv",
    "cbMOTPintra.csv",
    "dpcm_sf_probabilities.csv",
    "fa_sfmotp_inter.csv",
    "fa_sfmotp_intra.csv",
    "num_sec_probabilities.csv",
    "sect_len_motp.csv",
    "sect_len_probabilities.csv",
]


def _read_feature_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    renamed = {"file": "file"}
    for column in frame.columns:
        if column != "file":
            renamed[column] = f"{path.stem}::{column}"
    return frame.rename(columns=renamed)


def merge_feature_tables(paths: Iterable[Path]) -> pd.DataFrame:
    tables = [_read_feature_table(Path(path)) for path in paths]
    if not tables:
        raise ValueError("at least one feature table is required")
    merged = tables[0]
    for table in tables[1:]:
        merged = merged.merge(table, on="file", how="inner", validate="one_to_one")
    return merged


def attach_metadata(features: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    required = {"file", "label", "device"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"metadata missing columns: {sorted(missing)}")
    return features.merge(metadata, on="file", how="inner", validate="one_to_one")


def load_metadata(path: Path = DEFAULT_METADATA) -> pd.DataFrame:
    return pd.read_csv(path)


def load_jin_feature_frame(
    feature_dir: Path = DEFAULT_JIN_FEATURE_DIR,
    metadata_path: Path = DEFAULT_METADATA,
) -> pd.DataFrame:
    paths = [Path(feature_dir) / name for name in JIN_FEATURE_FILES]
    features = merge_feature_tables(paths)
    return attach_metadata(features, load_metadata(metadata_path))


def load_proposed_feature_frame(
    matrix_path: Path = DEFAULT_PROPOSED_MATRIX,
    metadata_path: Path = DEFAULT_METADATA,
) -> pd.DataFrame:
    features = pd.read_csv(matrix_path)
    return attach_metadata(features, load_metadata(metadata_path))


def resolve_audio_paths(audio_root: Path, filenames: Iterable[str]) -> dict[str, Path]:
    wanted = set(filenames)
    resolved: dict[str, Path] = {}
    for path in Path(audio_root).rglob("*"):
        if path.is_file() and path.name in wanted and path.name not in resolved:
            resolved[path.name] = path
            if len(resolved) == len(wanted):
                break
    missing = sorted(wanted.difference(resolved))
    if missing:
        raise FileNotFoundError(f"missing audio files: {missing[:10]}")
    return resolved
