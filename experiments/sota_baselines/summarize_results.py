from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PAPER_SOURCE = {
    "jin": "Jin et al. 2019 style",
    "jin_proposed": "Jin et al. 2019 proposed, adapted",
    "gmm_ubm": "Wang et al. 2023 baseline",
    "gsv_svm": "Wang et al. 2023 baseline",
    "mfcc_cnn": "Wang et al. 2023 baseline",
    "wang_proposed": "Wang et al. 2023 proposed, paper based reimplementation",
    "proposed": "Ours",
}

INPUT_FEATURE = {
    "jin": "AAC codebook, scalefactor, and section statistics",
    "jin_proposed": "AAC codebook, scalefactor, and section statistics",
    "gmm_ubm": "39 dimensional MFCC sequences",
    "gsv_svm": "64 component GMM supervectors from MFCC",
    "mfcc_cnn": "39 dimensional MFCC maps",
    "wang_proposed": "Raw waveform segments",
    "proposed": "ISOBMFF metadata plus AAC PMF and TPM features",
}

MODEL_NAME = {
    "jin": "VT plus SVM ranking, RBF SVM",
    "jin_proposed": "VT plus SVM ranking, tuned RBF SVM",
    "gmm_ubm": "64 component GMM-UBM score",
    "gsv_svm": "GSV with RBF SVM",
    "mfcc_cnn": "compact MFCC-CNN",
    "wang_proposed": "80 Sinc windows, two 60 kernel convolution layers, three layer 2048 node DNN, voting aggregation",
    "proposed": "Voting classifier, LR RF SVM",
}


def markdown_table(table: pd.DataFrame) -> str:
    columns = list(table.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in table.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.6f}".rstrip("0").rstrip("."))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    result_dir = Path(__file__).resolve().parent / "results"
    rows = []
    for path in sorted(result_dir.glob("*_summary.json")):
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        method = data["method"]
        rows.append(
            {
                "Method": method,
                "Inspired by": PAPER_SOURCE.get(method, method),
                "Input feature": INPUT_FEATURE.get(method, ""),
                "Model": MODEL_NAME.get(method, ""),
                "Protocol": data["protocol"],
                "Folds": data["folds"],
                "Accuracy mean": data["accuracy_mean"],
                "Accuracy std": data["accuracy_std"],
                "Weighted F1 mean": data["f1_weighted_mean"],
            }
        )
    table = pd.DataFrame(rows)
    out_csv = result_dir / "sota_baseline_comparison_summary.csv"
    out_md = result_dir / "sota_baseline_comparison_summary.md"
    table.to_csv(out_csv, index=False)
    markdown = markdown_table(table)
    out_md.write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
