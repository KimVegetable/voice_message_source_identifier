# SOTA Baseline Comparison

This folder contains independent comparison code for the manuscript revision. It does not change the public classifier model or the GUI and CLI behavior.

Implemented comparison methods:

- `jin`: Jin et al. style AAC encoding characteristic baseline.
- `jin_proposed`: Jin et al. proposed method adapted to this dataset, with feature portion, `C`, and `gamma` selected inside each training fold.
- `wang_proposed`: Wang et al. proposed end to end transfer learning framework reimplemented from the paper description. The `paper` profile uses the disclosed architecture values used in the manuscript comparison.
- `gmm_ubm`, `gsv_svm`, `mfcc_cnn`: auxiliary Wang style methods retained for diagnostics.
- `proposed`: the project feature representation with the original voting classifier, retained for diagnostic reruns. The manuscript comparison row uses the revised logistic regression result reported in `paper_results/jin_wang_ours_logo_comparison.csv`.

Curated result snapshots:

- `paper_results/jin_wang_ours_logo_comparison.csv`
- `paper_results/jin_wang_ours_logo_comparison.md`
- `paper_results/SOTA_BASELINE_RESULT_REPORT.md`

The manuscript three row comparison is:

| Method | Accuracy (%) | Weighted F1 (%) |
| --- | ---: | ---: |
| Jin et al. proposed, adapted | 80.425 | 79.671 |
| Wang et al. proposed, paper profile reimplementation | 4.123 | 2.102 |
| Proposed representation with LR | 97.917 | 97.895 |

To rerun the experiments, provide the local dataset and feature inputs. These files are not committed to the public source tree.

```powershell
$env:ASI_METADATA = "path\to\sample_metadata_from_filename.csv"
$env:ASI_JIN_FEATURE_DIR = "path\to\preprocessing_20250227_225033"
$env:ASI_PROPOSED_MATRIX = "path\to\feature_matrix_used_columns.csv"
$env:ASI_AUDIO_ROOT = "path\to\audio_dataset"

python experiments\sota_baselines\run_baselines.py --method jin_proposed --protocol logo
python experiments\sota_baselines\run_baselines.py --method wang_proposed --protocol logo --sample-rate 32000 --wang-tune-profile paper
```

`wang_proposed` and `mfcc_cnn` require PyTorch. The Jin and classical Wang baselines use the main project scientific Python stack.

