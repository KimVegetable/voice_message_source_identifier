import csv
import os
import re
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = Path(os.environ.get("ASI_CURRENT_OUTPUT_DIR", EXPERIMENT_ROOT / "results"))
SOURCE_ROOT = Path(os.environ.get("ASI_SOURCE_ROOT", REPO_ROOT))
DATASET_ROOT = Path(
    os.environ.get(
        "ASI_CURRENT_DATASET_ROOT",
        str(EXPERIMENT_ROOT / "inputs" / "current_version_voice_message_dataset"),
    )
)
MANIFEST_PATH = Path(
    os.environ.get(
        "ASI_CURRENT_MANIFEST",
        EXPERIMENT_ROOT / "inputs" / "current_version_dataset_manifest_final_renamed.csv",
    )
)
WORK_DIR = ARTIFACT_DIR / "_tmp_current_human_noise_usability"
TEMP_AUDIO_DIR = WORK_DIR / "aac_audio_flat"

FILENAME_RE = re.compile(
    r"^(and5|ios5)_(.+?)_30cm_front_s([1-5])_(tts_clean|human_clean|human_noise)$"
)
APP_TO_LABEL = {
    "band": "band",
    "messenger": "messenger",
    "kakaotalk": "kakaotalk",
    "line": "line",
    "naverworks": "naverworks",
    "session": "session",
    "signal": "signal",
    "slack": "slack",
    "viber": "viber",
    "webex": "webex",
    "wire": "wire",
}
LDA_TARGETS = [
    "cbMOTPintra",
    "dpcm_sf_probabilities",
    "fa_sfmotp_inter",
    "fa_sfmotp_intra",
    "sect_len_probabilities",
    "sect_len_motp",
]


@contextmanager
def pushd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def parse_name(file_name: str) -> dict:
    match = FILENAME_RE.match(Path(file_name).stem)
    if not match:
        raise ValueError(f"Unexpected file name pattern: {file_name}")
    platform_token, app, sentence, condition = match.groups()
    platform = "and" if platform_token == "and5" else "ios"
    return {
        "platform": platform,
        "app": app,
        "sentence": f"s{sentence}",
        "condition": condition,
        "true_label": f"{platform}_{APP_TO_LABEL[app]}",
    }


def reset_work_dir() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    if WORK_DIR.exists():
        resolved_work = WORK_DIR.resolve()
        if ARTIFACT_DIR.resolve() not in resolved_work.parents:
            raise RuntimeError(f"Refusing to delete unexpected path: {resolved_work}")
        shutil.rmtree(WORK_DIR)
    TEMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)


def copy_aac_scope_files(manifest: pd.DataFrame) -> pd.DataFrame:
    aac = manifest[manifest["codec_name"].str.lower().eq("aac")].copy()
    for _, row in aac.iterrows():
        source = DATASET_ROOT / row["relative_path"]
        destination = TEMP_AUDIO_DIR / row["name"]
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, destination)
    return aac


def extract_current_features() -> Path:
    sys.path.insert(0, str(SOURCE_ROOT))
    from preprocessing.extract_feature import run_extract
    from preprocessing.preprocessing_feature import run_preprocessing

    with pushd(WORK_DIR):
        extracted_csv = Path(run_extract(str(TEMP_AUDIO_DIR)))
        feature_folder = Path(run_preprocessing(str(extracted_csv)))
    return WORK_DIR / feature_folder


def read_feature_frames(feature_folder: Path) -> dict[str, pd.DataFrame]:
    frames = {}
    for csv_path in sorted(feature_folder.glob("*.csv")):
        frames[csv_path.stem] = pd.read_csv(csv_path, float_precision="round_trip")
    return frames


def numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    file_col = df["file"]
    features = df.drop(columns=["file"]).copy()
    for col in features.columns:
        features[col] = pd.to_numeric(features[col], errors="coerce")
    return pd.concat([file_col, features], axis=1)


def fit_lda_frames(
    frames: dict[str, pd.DataFrame],
    metadata: pd.DataFrame,
    train_files: set[str],
) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    out_frames = {name: numeric_frame(df) for name, df in frames.items()}
    train_meta = metadata[metadata["file"].isin(train_files)].copy()
    y_train = train_meta.set_index("file").loc[list(train_files), "true_label"].to_numpy()
    diagnostics = []

    for name in LDA_TARGETS:
        if name not in out_frames:
            continue
        df = out_frames[name]
        feature_cols = [c for c in df.columns if c != "file"]
        train_df = df[df["file"].isin(train_files)].set_index("file").loc[list(train_files)]
        train_values = train_df[feature_cols].replace([np.inf, -np.inf], np.nan)
        means = train_values.mean(axis=0).fillna(0.0)
        train_values = train_values.fillna(means)
        variances = train_values.var(axis=0)
        selected_cols = variances[variances > 1e-12].index.tolist()
        if not selected_cols:
            continue

        all_values = df[selected_cols].replace([np.inf, -np.inf], np.nan).fillna(means[selected_cols])
        n_components = min(len(np.unique(y_train)) - 1, len(selected_cols))
        lda = LinearDiscriminantAnalysis(n_components=n_components, solver="svd")
        lda.fit(train_values[selected_cols].to_numpy(dtype=np.float64), y_train)
        transformed = lda.transform(all_values.to_numpy(dtype=np.float64))
        lda_cols = [f"LDA{i + 1}" for i in range(transformed.shape[1])]
        lda_df = pd.DataFrame(transformed, columns=lda_cols)
        lda_df.insert(0, "file", df["file"])
        out_frames[f"{name}_lda"] = lda_df
        diagnostics.append(
            {
                "feature_group": name,
                "train_files": len(train_files),
                "raw_columns": len(feature_cols),
                "train_nonconstant_columns": len(selected_cols),
                "lda_components": transformed.shape[1],
            }
        )
    return out_frames, diagnostics


def merge_frames(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    merged = None
    for name in sorted(frames):
        if name in LDA_TARGETS and f"{name}_lda" in frames:
            continue
        df = frames[name]
        if "file" not in df.columns:
            continue
        if merged is None:
            merged = df.copy()
        else:
            merged = pd.merge(merged, df, on="file", suffixes=("", f"_{name}"))
            merged = merged.loc[:, ~merged.columns.duplicated()]
    if merged is None:
        raise RuntimeError("No features were merged.")
    return merged


def make_protocol_matrix(
    frames: dict[str, pd.DataFrame],
    metadata: pd.DataFrame,
    train_files: set[str],
    test_files: set[str],
) -> tuple[pd.DataFrame, list[str], list[dict]]:
    transformed_frames, lda_diagnostics = fit_lda_frames(frames, metadata, train_files)
    merged = merge_frames(transformed_frames)
    data = metadata[metadata["file"].isin(train_files | test_files)].merge(merged, on="file", how="left")
    feature_cols = [
        c for c in data.columns
        if c not in {"file", "platform", "app", "sentence", "condition", "true_label"}
    ]
    for col in feature_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    train_mask = data["file"].isin(train_files)
    train_values = data.loc[train_mask, feature_cols].replace([np.inf, -np.inf], np.nan)
    means = train_values.mean(axis=0).fillna(0.0)
    data[feature_cols] = data[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(means)
    variances = data.loc[train_mask, feature_cols].var(axis=0)
    selected_cols = variances[variances > 1e-12].index.tolist()
    return data, selected_cols, lda_diagnostics


def model_factories() -> dict[str, Pipeline]:
    return {
        "voting_rf_lr_svc": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    VotingClassifier(
                        estimators=[
                            ("rf", RandomForestClassifier(n_estimators=100, max_depth=16, random_state=42)),
                            ("lr", LogisticRegression(max_iter=2000, random_state=42)),
                            ("svc", SVC(kernel="linear", probability=True, random_state=42)),
                        ],
                        voting="soft",
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", RandomForestClassifier(n_estimators=100, max_depth=16, random_state=42)),
            ]
        ),
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
            ]
        ),
    }


def protocol_specs(metadata: pd.DataFrame) -> list[dict]:
    specs = []
    sentences = sorted(metadata["sentence"].unique())

    for sentence in sentences:
        specs.append(
            {
                "protocol": "human_clean_leave_one_sentence_out",
                "fold": sentence,
                "train_files": set(metadata[(metadata["condition"].eq("human_clean")) & (~metadata["sentence"].eq(sentence))]["file"]),
                "test_files": set(metadata[(metadata["condition"].eq("human_clean")) & (metadata["sentence"].eq(sentence))]["file"]),
            }
        )
        specs.append(
            {
                "protocol": "human_noise_leave_one_sentence_out",
                "fold": sentence,
                "train_files": set(metadata[(metadata["condition"].eq("human_noise")) & (~metadata["sentence"].eq(sentence))]["file"]),
                "test_files": set(metadata[(metadata["condition"].eq("human_noise")) & (metadata["sentence"].eq(sentence))]["file"]),
            }
        )
        specs.append(
            {
                "protocol": "human_clean_and_noise_leave_one_sentence_out",
                "fold": sentence,
                "train_files": set(
                    metadata[
                        (metadata["condition"].isin(["human_clean", "human_noise"]))
                        & (~metadata["sentence"].eq(sentence))
                    ]["file"]
                ),
                "test_files": set(
                    metadata[
                        (metadata["condition"].isin(["human_clean", "human_noise"]))
                        & (metadata["sentence"].eq(sentence))
                    ]["file"]
                ),
            }
        )
        specs.append(
            {
                "protocol": "all_conditions_leave_one_sentence_out",
                "fold": sentence,
                "train_files": set(metadata[~metadata["sentence"].eq(sentence)]["file"]),
                "test_files": set(metadata[metadata["sentence"].eq(sentence)]["file"]),
            }
        )

    specs.extend(
        [
            {
                "protocol": "train_human_clean_test_human_noise",
                "fold": "all",
                "train_files": set(metadata[metadata["condition"].eq("human_clean")]["file"]),
                "test_files": set(metadata[metadata["condition"].eq("human_noise")]["file"]),
            },
            {
                "protocol": "train_tts_and_human_clean_test_human_noise",
                "fold": "all",
                "train_files": set(metadata[metadata["condition"].isin(["tts_clean", "human_clean"])]["file"]),
                "test_files": set(metadata[metadata["condition"].eq("human_noise")]["file"]),
            },
            {
                "protocol": "train_human_clean_and_noise_test_tts",
                "fold": "all",
                "train_files": set(metadata[metadata["condition"].isin(["human_clean", "human_noise"])]["file"]),
                "test_files": set(metadata[metadata["condition"].eq("tts_clean")]["file"]),
            },
        ]
    )
    return specs


def evaluate_protocols(frames: dict[str, pd.DataFrame], metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    result_rows = []
    prediction_rows = []
    lda_rows = []

    for spec in protocol_specs(metadata):
        data, feature_cols, lda_diagnostics = make_protocol_matrix(
            frames,
            metadata,
            spec["train_files"],
            spec["test_files"],
        )
        train = data[data["file"].isin(spec["train_files"])].copy()
        test = data[data["file"].isin(spec["test_files"])].copy()
        X_train = train[feature_cols].to_numpy(dtype=np.float64)
        y_train = train["true_label"].to_numpy()
        X_test = test[feature_cols].to_numpy(dtype=np.float64)
        y_test = test["true_label"].to_numpy()

        for row in lda_diagnostics:
            lda_rows.append({"protocol": spec["protocol"], "fold": spec["fold"], **row})

        for model_name, model in model_factories().items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            result_rows.append(
                {
                    "protocol": spec["protocol"],
                    "fold": spec["fold"],
                    "model": model_name,
                    "n_train": len(train),
                    "n_test": len(test),
                    "selected_features": len(feature_cols),
                    "correct": int(np.sum(y_test == y_pred)),
                    "accuracy": accuracy,
                }
            )
            probs = model.predict_proba(X_test)
            max_probs = probs.max(axis=1)
            for (_, row), pred, prob in zip(test.iterrows(), y_pred, max_probs):
                prediction_rows.append(
                    {
                        "protocol": spec["protocol"],
                        "fold": spec["fold"],
                        "model": model_name,
                        "file": row["file"],
                        "condition": row["condition"],
                        "sentence": row["sentence"],
                        "true_label": row["true_label"],
                        "predicted_label": pred,
                        "match": bool(row["true_label"] == pred),
                        "max_probability": float(prob),
                    }
                )
        print(f"finished {spec['protocol']} fold={spec['fold']}")

    return pd.DataFrame(result_rows), pd.DataFrame(prediction_rows), pd.DataFrame(lda_rows)


def aggregate_results(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby(["protocol", "model"], as_index=False)
        .agg(
            folds=("fold", "count"),
            n_train_min=("n_train", "min"),
            n_train_max=("n_train", "max"),
            n_test_total=("n_test", "sum"),
            correct_total=("correct", "sum"),
            mean_fold_accuracy=("accuracy", "mean"),
            selected_features_mean=("selected_features", "mean"),
        )
        .assign(accuracy=lambda df: df["correct_total"] / df["n_test_total"])
    )


def main() -> None:
    reset_work_dir()
    manifest = pd.read_csv(MANIFEST_PATH)
    aac_manifest = copy_aac_scope_files(manifest)
    metadata = pd.DataFrame([{"file": row["name"], **parse_name(row["name"])} for _, row in aac_manifest.iterrows()])
    feature_folder = extract_current_features()
    frames = read_feature_frames(feature_folder)

    results, predictions, lda_diagnostics = evaluate_protocols(frames, metadata)
    aggregate = aggregate_results(results)

    results_path = ARTIFACT_DIR / "current_human_noise_usability_fold_results.csv"
    aggregate_path = ARTIFACT_DIR / "current_human_noise_usability_summary.csv"
    prediction_path = ARTIFACT_DIR / "current_human_noise_usability_predictions.csv"
    lda_path = ARTIFACT_DIR / "current_human_noise_usability_lda_diagnostics.csv"
    results.to_csv(results_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    aggregate.to_csv(aggregate_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    predictions.to_csv(prediction_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    lda_diagnostics.to_csv(lda_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)

    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)

    print(f"aac_files={len(aac_manifest)}")
    print(f"classes={metadata['true_label'].nunique()}")
    print(f"fold_results={results_path}")
    print(f"summary={aggregate_path}")
    print(f"predictions={prediction_path}")
    print(f"lda_diagnostics={lda_path}")
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
