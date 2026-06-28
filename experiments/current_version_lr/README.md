# Current Version LR Validation

This folder contains the current app version validation scripts used for the manuscript revision after switching the reported representative classifier to logistic regression.

Scripts:

- `current_version_original_lr_component_eval.py`: loads the deployable LR classifier from `models/trained_lr.pkl` and evaluates the current version AAC files without retraining.
- `current_human_noise_usability_eval.py`: trains condition specific diagnostic models on the current version subset to check speech and noise effects.

Curated result snapshots are in `paper_results`.

Key LR component results from `current_version_original_lr_component_eval.py`:

| Scope | Correct / Total | Accuracy (%) |
| --- | ---: | ---: |
| All current AAC files | 227 / 315 | 72.063 |
| TTS clean | 69 / 105 | 65.714 |
| Human clean | 89 / 105 | 84.762 |
| Human noise | 69 / 105 | 65.714 |

Useful diagnostic LR results from `current_human_noise_usability_eval.py`:

| Protocol | Correct / Total | Accuracy (%) |
| --- | ---: | ---: |
| Human clean leave one sentence out | 87 / 105 | 82.857 |
| Human noise leave one sentence out | 99 / 105 | 94.286 |
| Human clean and noise leave one sentence out | 185 / 210 | 88.095 |

To rerun locally, keep the repository as `ASI_SOURCE_ROOT` and point `ASI_CURRENT_DATASET_ROOT` to the current version audio dataset folder.

```powershell
$env:ASI_SOURCE_ROOT = "E:\06. WorkSpace\Pycharm\voice_message_source_identifier"
$env:ASI_CURRENT_DATASET_ROOT = "path\to\current_version_voice_message_dataset"

python experiments\current_version_lr\current_version_original_lr_component_eval.py
python experiments\current_version_lr\current_human_noise_usability_eval.py
```

Outputs from reruns are written to `experiments/current_version_lr/results` and ignored by Git.
