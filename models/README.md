# Model Files

The revised source tree provides the logistic regression classifier as the default deployable model because it is the representative best performing classifier in the manuscript revision.

| File | Role |
| --- | --- |
| `trained_lr.pkl` | Default classifier used by the CLI and GUI. It was extracted from `trained_voting.pkl` as `named_estimators_["lr"]`. |
| `trained_voting.pkl` | Original voting ensemble retained for backward compatibility. |
| `scaler.pkl`, `used_columns.pkl`, `labels.pkl` | Shared preprocessing and label mapping files required by both classifiers. |

Use the legacy voting model with:

```powershell
python -m cli.classify_audio_files --model voting path\to\audio_or_folder
```

The SOTA comparison baselines are evaluation scripts under `experiments/sota_baselines`, not deployable model files.

