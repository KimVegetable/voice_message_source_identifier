import argparse
import csv
import os
import re
import shutil
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = Path(os.environ.get("ASI_CURRENT_OUTPUT_DIR", EXPERIMENT_ROOT / "results"))
SOURCE_ROOT = Path(os.environ.get("ASI_SOURCE_ROOT", REPO_ROOT))
MANIFEST_PATH = Path(
    os.environ.get(
        "ASI_CURRENT_MANIFEST",
        EXPERIMENT_ROOT / "inputs" / "current_version_dataset_manifest_final_renamed.csv",
    )
)
DATASET_ROOT = Path(
    os.environ.get(
        "ASI_CURRENT_DATASET_ROOT",
        str(EXPERIMENT_ROOT / "inputs" / "current_version_voice_message_dataset"),
    )
)
TEMP_AUDIO_DIR = ARTIFACT_DIR / "_tmp_current_version_lr_aac_audio_flat"
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


def parse_name(file_name: str) -> dict:
    stem = Path(file_name).stem
    match = FILENAME_RE.match(stem)
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


def reset_temp_audio_dir() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    resolved_artifacts = ARTIFACT_DIR.resolve()
    if TEMP_AUDIO_DIR.exists():
        resolved_temp = TEMP_AUDIO_DIR.resolve()
        if resolved_artifacts not in resolved_temp.parents:
            raise RuntimeError(f"Refusing to delete unexpected path: {resolved_temp}")
        shutil.rmtree(TEMP_AUDIO_DIR)
    TEMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)


def copy_scope_files(rows: pd.DataFrame) -> None:
    reset_temp_audio_dir()
    for _, row in rows.iterrows():
        source = DATASET_ROOT / row["relative_path"]
        destination = TEMP_AUDIO_DIR / row["name"]
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, destination)


def load_classifier_module():
    sys.path.insert(0, str(SOURCE_ROOT))
    from cli import classify_audio_files

    classify_audio_files.SESSION_TIME = datetime.now().strftime(
        "current_version_original_lr_component_%Y%m%d_%H%M%S"
    )
    return classify_audio_files


def build_feature_matrix(audio_path: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    classifier = load_classifier_module()
    extracted_csv = classifier.run_extract(str(audio_path))
    feature_folder = classifier.run_preprocessing(extracted_csv)

    filtering_targets = [
        "cbMOTPintra",
        "dpcm_sf_probabilities",
        "fa_sfmotp_inter",
        "fa_sfmotp_intra",
        "num_sec_probabilities",
        "sect_len_probabilities",
        "sect_len_motp",
    ]

    for name in filtering_targets:
        csv_path = Path(feature_folder) / f"{name}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path, float_precision="round_trip")
            classifier.apply_column_filtering(df, name).to_csv(csv_path, index=False)

    classifier.apply_lda_transform(feature_folder)
    merged_data = classifier.merge_csvs(feature_folder)
    if merged_data is None or merged_data.empty:
        raise RuntimeError("No features to classify. Check input path and preprocessing pipeline.")

    file_list = merged_data["file"].values
    feature_frame = merged_data.drop(columns=["file"], errors="ignore")

    used_columns = joblib.load(SOURCE_ROOT / "models" / "used_columns.pkl")
    for col in [c for c in used_columns if c not in feature_frame.columns]:
        feature_frame[col] = 0.0
    feature_frame = feature_frame[used_columns]
    feature_frame = feature_frame.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    feature_frame.insert(0, "file", file_list)
    feature_frame.to_csv(
        ARTIFACT_DIR / "current_version_original_lr_component_feature_matrix.csv",
        index=False,
        encoding="utf-8-sig",
    )

    scaler = joblib.load(SOURCE_ROOT / "models" / "scaler.pkl")
    x_scaled = scaler.transform(feature_frame.drop(columns=["file"]).values)
    return file_list, x_scaled, feature_frame


def predict_with_lr_component(file_list: np.ndarray, x_scaled: np.ndarray) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr_path = SOURCE_ROOT / "models" / "trained_lr.pkl"
        if lr_path.exists():
            lr_model = joblib.load(lr_path)
        else:
            voting_model = joblib.load(SOURCE_ROOT / "models" / "trained_voting.pkl")
            lr_model = voting_model.named_estimators_["lr"]
        labels = joblib.load(SOURCE_ROOT / "models" / "labels.pkl")

    probs = lr_model.predict_proba(x_scaled)

    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.asarray(labels)
    model_classes = np.asarray(lr_model.classes_)
    if np.issubdtype(model_classes.dtype, np.integer):
        class_names = label_encoder.inverse_transform(model_classes.astype(int))
    else:
        class_names = np.array([str(c) for c in model_classes])

    probs_df = pd.DataFrame(probs, columns=class_names)
    probs_df.insert(0, "file", file_list)
    probs_df.to_csv(
        ARTIFACT_DIR / "current_version_original_lr_component_probabilities.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.12f",
    )
    return probs_df


def group_accuracy(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    grouped = (
        df.groupby(group_cols, dropna=False)
        .agg(n=("match", "size"), correct=("match", "sum"), accuracy=("match", "mean"))
        .reset_index()
    )
    grouped["correct"] = grouped["correct"].astype(int)
    return grouped


def summarize_predictions(probs: pd.DataFrame, manifest_scope: pd.DataFrame) -> dict:
    label_cols = [col for col in probs.columns if col != "file"]
    probs = probs.copy()
    probs["max_prob"] = probs[label_cols].max(axis=1)
    probs["predicted_label"] = probs[label_cols].idxmax(axis=1)

    parsed = probs["file"].apply(parse_name).apply(pd.Series)
    predictions = pd.concat([probs[["file", "predicted_label", "max_prob"]], parsed], axis=1)
    predictions["match"] = predictions["predicted_label"] == predictions["true_label"]

    manifest_small = manifest_scope[
        ["name", "extension", "duration_sec", "sample_rate", "channels", "bit_rate", "sha256"]
    ].rename(columns={"name": "file"})
    predictions = predictions.merge(manifest_small, on="file", how="left")

    prediction_path = ARTIFACT_DIR / "current_version_original_lr_component_predictions.csv"
    predictions.to_csv(prediction_path, index=False, encoding="utf-8-sig")

    overall = {
        "scope_files": int(len(predictions)),
        "correct": int(predictions["match"].sum()),
        "accuracy": float(predictions["match"].mean()),
    }

    summaries = {
        "overall": pd.DataFrame([overall]),
        "by_condition": group_accuracy(predictions, ["condition"]),
        "by_platform": group_accuracy(predictions, ["platform"]),
        "by_app": group_accuracy(predictions, ["app"]),
        "by_true_label": group_accuracy(predictions, ["true_label"]),
        "by_condition_platform": group_accuracy(predictions, ["condition", "platform"]),
    }

    for name, df in summaries.items():
        df.to_csv(
            ARTIFACT_DIR / f"current_version_original_lr_component_summary_{name}.csv",
            index=False,
            encoding="utf-8-sig",
            quoting=csv.QUOTE_MINIMAL,
        )

    return {
        "prediction_path": str(prediction_path),
        "overall": overall,
        "summaries": {
            key: str(ARTIFACT_DIR / f"current_version_original_lr_component_summary_{key}.csv")
            for key in summaries
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke test limit.")
    args = parser.parse_args()

    manifest = pd.read_csv(MANIFEST_PATH)
    in_scope = manifest[manifest["codec_name"].str.lower().eq("aac")].copy()
    out_of_scope = manifest[~manifest["codec_name"].str.lower().eq("aac")].copy()

    if args.limit is not None:
        in_scope = in_scope.head(args.limit).copy()

    copy_scope_files(in_scope)
    file_list, x_scaled, _ = build_feature_matrix(TEMP_AUDIO_DIR)
    probabilities = predict_with_lr_component(file_list, x_scaled)
    result = summarize_predictions(probabilities, in_scope)

    if TEMP_AUDIO_DIR.exists():
        shutil.rmtree(TEMP_AUDIO_DIR)

    print(
        "overall="
        f"{result['overall']['correct']}/{result['overall']['scope_files']} "
        f"accuracy={result['overall']['accuracy']:.6f}"
    )
    print(f"predictions={result['prediction_path']}")
    print(f"out_of_aac_scope={len(out_of_scope)}")


if __name__ == "__main__":
    main()
