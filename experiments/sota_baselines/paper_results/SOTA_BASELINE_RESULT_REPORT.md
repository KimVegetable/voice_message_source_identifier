# SOTA Baseline Result Report

Date: 2026-06-24

Purpose: provide independent reviewer response artifacts for comparing the proposed app source classifier against prior audio source identification methods adapted to the current dataset.

## Compared Methods For Manuscript

| Method | Paper basis | Implemented comparison role |
| --- | --- | --- |
| `jin_proposed` | Jin et al. 2019 proposed method | AAC encoding characteristic method adapted to the available AAC codebook, scalefactor, and section statistics. Same value filtering, SVM feature ranking, and RBF SVM classification are used, with feature portion, `C`, and `gamma` tuned inside each training fold. |
| `wang_proposed` | Wang et al. 2023 proposed method | Paper based reimplementation of the end to end transfer learning method. It uses raw waveform segments, amplitude normalization, downsampling, 80 Sinc windows with length 251, two convolutional layers with 60 kernels of length 5, a three layer DNN with 2048 nodes per layer, and voting based segment aggregation. |
| `proposed_lr` | Ours | Current finalized ISOBMFF metadata plus AAC PMF and TPM representation with logistic regression, matching the representative classifier used in the revised manuscript comparison. |

Official source code or pretrained weights were not available for Jin et al. or Wang et al. Therefore, these rows should be described as paper based adaptations or reimplementations, not exact reproductions of the original authors' private pipelines.

Hyperparameters for `jin_proposed` and `wang_proposed` were selected only from the outer training fold. The held out device fold was not used for tuning.

## Manuscript Result Table

| Method | Input feature | Model | Protocol | Folds | Accuracy (%) | Weighted F1 (%) |
| --- | --- | --- | --- | --- | --- | --- |
| Jin et al. proposed, adapted | AAC codebook, scalefactor, and section statistics | Same value filtering, SVM feature ranking, tuned RBF SVM | Leave one device out | 8 | 80.425 | 79.671 |
| Wang et al. proposed, paper profile reimplementation | Raw waveform segments | Amplitude normalization, downsampling, 80 Sinc windows, two 60 kernel convolution layers, three layer 2048 node DNN, voting aggregation | Leave one device out | 8 | 4.123 | 2.102 |
| Proposed representation with LR | ISOBMFF metadata plus AAC PMF and TPM features | Logistic regression | Leave one device out | 8 | 97.917 | 97.895 |

The proposed row uses the best classifier in the finalized main evaluation table, LR, with 97.917% accuracy and 97.895% weighted F1. The voting ensemble remains reported in the main classifier comparison table, but it is not used as the representative proposed result in the adapted prior method comparison.

## Auxiliary Result Table

The broader artifact still contains earlier auxiliary Wang baseline rows (`gmm_ubm`, `gsv_svm`, and `mfcc_cnn`) and the initial `jin` row. These rows are retained for traceability but are not used in the final manuscript comparison table after the decision to compare Wang et al.'s proposed method directly.

| Method | Inspired by | Input feature | Model | Protocol | Folds | Accuracy mean | Accuracy std | Weighted F1 mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gmm_ubm | Wang et al. 2023 baseline | 39 dimensional MFCC sequences | 64 component GMM-UBM score | logo | 8 | 0.578993 | 0.085893 | 0.564318 |
| gsv_svm | Wang et al. 2023 baseline | 64 component GMM supervectors from MFCC | GSV with RBF SVM | logo | 8 | 0.677517 | 0.055693 | 0.661827 |
| mfcc_cnn | Wang et al. 2023 baseline | 39 dimensional MFCC maps | compact MFCC-CNN | logo | 8 | 0.596354 | 0.201967 | 0.565839 |
| jin | Jin et al. 2019 style | AAC codebook, scalefactor, and section statistics | VT plus SVM ranking, RBF SVM | logo | 8 | 0.774306 | 0.041251 | 0.757142 |

## Interpretation For Manuscript Revision

These results support a table that compares adapted prior methods on the same dataset, rather than comparing percentages reported on different external datasets. This is the stronger response to a reviewer request for SOTA and baseline comparison.

Recommended wording should be cautious:

- The Jin and Wang rows are paper based adaptations using available data and the algorithmic details reported in the papers.
- The comparison evaluates whether prior device oriented audio source identification methods transfer to application source identification in this dataset.
- Under leave one device out validation, Jin et al.'s AAC encoding characteristic approach transferred better than Wang et al.'s raw waveform Sinc and DNN approach, but both were substantially below the proposed container and AAC syntax representation.
- The Wang paper profile result supersedes the earlier CPU feasible fast profile result. The fast profile reached 18.793% accuracy and 16.939% weighted F1, while the larger paper profile reached 4.123% accuracy and 2.102% weighted F1.
- The result should be described as evidence under the tested dataset and protocol, not as a universal ranking of the original papers.

## Artifact Paths

- Code: `experiments/sota_baselines`
- Three row comparison CSV: `experiments/sota_baselines/paper_results/jin_wang_ours_logo_comparison.csv`
- Three row comparison Markdown: `experiments/sota_baselines/paper_results/jin_wang_ours_logo_comparison.md`
- Summary CSV: `experiments/sota_baselines/paper_results/sota_baseline_comparison_summary.csv`
- Summary Markdown: `experiments/sota_baselines/paper_results/sota_baseline_comparison_summary.md`
- Jin fold metrics: `experiments/sota_baselines/paper_results/jin_proposed_logo_folds.csv`
- Wang fold metrics: `experiments/sota_baselines/paper_results/wang_proposed_logo_folds.csv`
