from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from sota_baselines.audio import load_or_extract_mfcc, load_or_extract_waveform
from sota_baselines.dataset import (
    DEFAULT_AUDIO_ROOT,
    DEFAULT_JIN_FEATURE_DIR,
    DEFAULT_METADATA,
    DEFAULT_PROPOSED_MATRIX,
    load_jin_feature_frame,
    load_proposed_feature_frame,
    resolve_audio_paths,
)
from sota_baselines.evaluation import leave_one_group_splits, stratified_kfold_splits, weighted_metrics
from sota_baselines.jin_baseline import fit_predict_jin_svm, fit_predict_jin_tuned_svm
from sota_baselines.wang_baselines import fit_predict_gmm_ubm, fit_predict_gsv_svm


META_COLUMNS = {"file", "label", "device", "platform", "app", "distance", "direction", "duration"}


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in META_COLUMNS]


def make_proposed_classifier(random_state: int) -> VotingClassifier:
    lr = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, C=1.0, solver="lbfgs"),
    )
    rf = RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1)
    svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", gamma="auto", C=1.0, probability=False))
    return VotingClassifier(
        estimators=[("lr", lr), ("rf", rf), ("svm", svm)],
        voting="hard",
    )


def get_splits(protocol: str, labels: np.ndarray, groups: np.ndarray, random_state: int):
    if protocol == "logo":
        return list(leave_one_group_splits(labels, groups))
    if protocol == "random5":
        return list(stratified_kfold_splits(labels, n_splits=5, random_state=random_state))
    raise ValueError(f"unknown protocol: {protocol}")


def run_jin(
    frame: pd.DataFrame,
    protocol: str,
    random_state: int,
    tuned: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = frame["label"].to_numpy()
    groups = frame["device"].to_numpy()
    columns = feature_columns(frame)
    rows = []
    predictions = []
    for fold_id, (fold_name, train_idx, test_idx) in enumerate(get_splits(protocol, labels, groups, random_state), start=1):
        if tuned:
            result = fit_predict_jin_tuned_svm(
                frame.iloc[train_idx],
                frame.iloc[test_idx],
                columns,
                random_state=random_state + fold_id,
            )
        else:
            result = fit_predict_jin_svm(
                frame.iloc[train_idx],
                frame.iloc[test_idx],
                columns,
                random_state=random_state + fold_id,
            )
        y_true = labels[test_idx]
        metrics = weighted_metrics(y_true, result.predictions)
        rows.append(
            {
                "fold": fold_id,
                "held_group": fold_name,
                "test_size": len(test_idx),
                "selected_features": len(result.selected_columns),
                "selected_portion": result.selected_portion,
                "selected_c": result.selected_c,
                "selected_gamma": result.selected_gamma,
                "validation_score": result.validation_score,
                **metrics,
            }
        )
        for file_name, truth, pred in zip(frame.iloc[test_idx]["file"], y_true, result.predictions):
            predictions.append({"fold": fold_id, "file": file_name, "true": truth, "pred": pred})
    return pd.DataFrame(rows), pd.DataFrame(predictions)


def run_proposed(frame: pd.DataFrame, protocol: str, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = frame["label"].to_numpy()
    groups = frame["device"].to_numpy()
    columns = feature_columns(frame)
    x = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
    rows = []
    predictions = []
    for fold_id, (fold_name, train_idx, test_idx) in enumerate(get_splits(protocol, labels, groups, random_state), start=1):
        classifier = make_proposed_classifier(random_state + fold_id)
        classifier.fit(x[train_idx], labels[train_idx])
        y_pred = classifier.predict(x[test_idx])
        metrics = weighted_metrics(labels[test_idx], y_pred)
        rows.append({"fold": fold_id, "held_group": fold_name, "test_size": len(test_idx), **metrics})
        for file_name, truth, pred in zip(frame.iloc[test_idx]["file"], labels[test_idx], y_pred):
            predictions.append({"fold": fold_id, "file": file_name, "true": truth, "pred": pred})
    return pd.DataFrame(rows), pd.DataFrame(predictions)


def build_mfcc_sequences(
    frame: pd.DataFrame,
    audio_root: Path,
    cache_dir: Path,
    sample_rate: int,
) -> list[np.ndarray]:
    paths = resolve_audio_paths(audio_root, frame["file"])
    sequences = []
    for idx, file_name in enumerate(frame["file"], start=1):
        cache_path = cache_dir / f"{Path(file_name).stem}.npz"
        sequences.append(load_or_extract_mfcc(paths[file_name], cache_path, sample_rate=sample_rate))
        if idx % 100 == 0:
            print(f"MFCC cached {idx}/{len(frame)}", flush=True)
    return sequences


def build_waveforms(
    frame: pd.DataFrame,
    audio_root: Path,
    cache_dir: Path,
    sample_rate: int,
) -> list[np.ndarray]:
    paths = resolve_audio_paths(audio_root, frame["file"])
    waveforms = []
    for idx, file_name in enumerate(frame["file"], start=1):
        cache_path = cache_dir / f"{Path(file_name).stem}.npz"
        waveforms.append(load_or_extract_waveform(paths[file_name], cache_path, sample_rate=sample_rate))
        if idx % 100 == 0:
            print(f"Waveform cached {idx}/{len(frame)}", flush=True)
    return waveforms


def run_wang(
    frame: pd.DataFrame,
    sequences: list[np.ndarray],
    method: str,
    protocol: str,
    random_state: int,
    n_components: int,
    cnn_epochs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = frame["label"].to_numpy()
    groups = frame["device"].to_numpy()
    rows = []
    predictions = []
    for fold_id, (fold_name, train_idx, test_idx) in enumerate(get_splits(protocol, labels, groups, random_state), start=1):
        train_features = [sequences[i] for i in train_idx]
        test_features = [sequences[i] for i in test_idx]
        if method == "gmm_ubm":
            y_pred = fit_predict_gmm_ubm(
                train_features,
                labels[train_idx],
                test_features,
                n_components=n_components,
                random_state=random_state + fold_id,
            )
        elif method == "gsv_svm":
            y_pred = fit_predict_gsv_svm(
                train_features,
                labels[train_idx],
                test_features,
                n_components=n_components,
                random_state=random_state + fold_id,
            )
        elif method == "mfcc_cnn":
            from sota_baselines.torch_baselines import fit_predict_mfcc_cnn

            y_pred = fit_predict_mfcc_cnn(
                train_features,
                labels[train_idx],
                test_features,
                epochs=cnn_epochs,
                random_state=random_state + fold_id,
            )
        else:
            raise ValueError(method)
        metrics = weighted_metrics(labels[test_idx], y_pred)
        rows.append({"fold": fold_id, "held_group": fold_name, "test_size": len(test_idx), **metrics})
        for file_name, truth, pred in zip(frame.iloc[test_idx]["file"], labels[test_idx], y_pred):
            predictions.append({"fold": fold_id, "file": file_name, "true": truth, "pred": pred})
    return pd.DataFrame(rows), pd.DataFrame(predictions)


def run_wang_proposed(
    frame: pd.DataFrame,
    waveforms: list[np.ndarray],
    protocol: str,
    random_state: int,
    sample_rate: int,
    tune_profile: str,
    checkpoint_dir: Path | None = None,
    checkpoint_stem: str | None = None,
    only_held_group: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = frame["label"].to_numpy()
    groups = frame["device"].to_numpy()
    rows = []
    predictions = []
    from sota_baselines.wang_proposed import config_to_dict, fit_predict_wang_proposed, wang_config_grid

    configs = wang_config_grid(tune_profile, sample_rate=sample_rate)
    for fold_id, (fold_name, train_idx, test_idx) in enumerate(get_splits(protocol, labels, groups, random_state), start=1):
        if only_held_group is not None and str(fold_name) != only_held_group:
            continue
        print(f"Wang proposed fold {fold_id}: {fold_name}", flush=True)
        train_waveforms = [waveforms[i] for i in train_idx]
        test_waveforms = [waveforms[i] for i in test_idx]
        result = fit_predict_wang_proposed(
            train_waveforms,
            labels[train_idx],
            test_waveforms,
            configs=configs,
            random_state=random_state + fold_id,
        )
        metrics = weighted_metrics(labels[test_idx], result.predictions)
        config_info = {f"config_{key}": value for key, value in config_to_dict(result.selected_config).items()}
        rows.append(
            {
                "fold": fold_id,
                "held_group": fold_name,
                "test_size": len(test_idx),
                "validation_score": result.validation_score,
                **config_info,
                **metrics,
            }
        )
        for file_name, truth, pred in zip(frame.iloc[test_idx]["file"], labels[test_idx], result.predictions):
            predictions.append({"fold": fold_id, "file": file_name, "true": truth, "pred": pred})
        if checkpoint_dir is not None and checkpoint_stem is not None:
            pd.DataFrame(rows).to_csv(checkpoint_dir / f"{checkpoint_stem}_partial_folds.csv", index=False)
            pd.DataFrame(predictions).to_csv(checkpoint_dir / f"{checkpoint_stem}_partial_predictions.csv", index=False)
    return pd.DataFrame(rows), pd.DataFrame(predictions)


def summarize(method: str, folds: pd.DataFrame) -> dict[str, float | str | int]:
    return {
        "method": method,
        "folds": int(len(folds)),
        "accuracy_mean": float(folds["accuracy"].mean()),
        "accuracy_std": float(folds["accuracy"].std(ddof=1)) if len(folds) > 1 else 0.0,
        "precision_weighted_mean": float(folds["precision_weighted"].mean()),
        "recall_weighted_mean": float(folds["recall_weighted"].mean()),
        "f1_weighted_mean": float(folds["f1_weighted"].mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["jin", "jin_proposed", "proposed", "gmm_ubm", "gsv_svm", "mfcc_cnn", "wang_proposed"], required=True)
    parser.add_argument("--protocol", choices=["random5", "logo"], default="random5")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--jin-feature-dir", type=Path, default=DEFAULT_JIN_FEATURE_DIR)
    parser.add_argument("--proposed-matrix", type=Path, default=DEFAULT_PROPOSED_MATRIX)
    parser.add_argument("--audio-root", type=Path, default=DEFAULT_AUDIO_ROOT)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--gmm-components", type=int, default=64)
    parser.add_argument("--cnn-epochs", type=int, default=20)
    parser.add_argument("--wang-tune-profile", choices=["smoke", "fast", "broad", "paper"], default="fast")
    parser.add_argument("--only-held-group", default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.method == "jin":
        frame = load_jin_feature_frame(args.jin_feature_dir, args.metadata)
        folds, preds = run_jin(frame, args.protocol, args.random_state)
    elif args.method == "jin_proposed":
        frame = load_jin_feature_frame(args.jin_feature_dir, args.metadata)
        folds, preds = run_jin(frame, args.protocol, args.random_state, tuned=True)
    elif args.method == "proposed":
        frame = load_proposed_feature_frame(args.proposed_matrix, args.metadata)
        folds, preds = run_proposed(frame, args.protocol, args.random_state)
    elif args.method == "wang_proposed":
        frame = load_jin_feature_frame(args.jin_feature_dir, args.metadata)[["file", "label", "device"]]
        cache_dir = ROOT / "cache" / f"waveform_{args.sample_rate}"
        waveforms = build_waveforms(frame, args.audio_root, cache_dir, args.sample_rate)
        stem = f"{args.method}_{args.protocol}"
        folds, preds = run_wang_proposed(
            frame,
            waveforms,
            args.protocol,
            args.random_state,
            args.sample_rate,
            args.wang_tune_profile,
            checkpoint_dir=args.output_dir,
            checkpoint_stem=stem,
            only_held_group=args.only_held_group,
        )
    else:
        frame = load_jin_feature_frame(args.jin_feature_dir, args.metadata)[["file", "label", "device"]]
        cache_dir = ROOT / "cache" / f"mfcc_{args.sample_rate}"
        sequences = build_mfcc_sequences(frame, args.audio_root, cache_dir, args.sample_rate)
        folds, preds = run_wang(
            frame,
            sequences,
            args.method,
            args.protocol,
            args.random_state,
            args.gmm_components,
            args.cnn_epochs,
        )

    stem = f"{args.method}_{args.protocol}"
    folds.to_csv(args.output_dir / f"{stem}_folds.csv", index=False)
    preds.to_csv(args.output_dir / f"{stem}_predictions.csv", index=False)
    summary = summarize(args.method, folds)
    summary.update({"protocol": args.protocol})
    (args.output_dir / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
