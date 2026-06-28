| Method | Input feature | Model | Protocol | Folds | Accuracy (%) | Weighted F1 (%) |
| --- | --- | --- | --- | --- | --- | --- |
| Jin et al. proposed, adapted | AAC codebook, scalefactor, and section statistics | Same value filtering, SVM feature ranking, tuned RBF SVM | Leave one device out | 8 | 80.425 | 79.671 |
| Wang et al. proposed, paper profile reimplementation | Raw waveform segments | Amplitude normalization, downsampling, 80 Sinc windows, two 60-kernel convolution layers, 3-layer 2048-node DNN, voting aggregation | Leave one device out | 8 | 4.123 | 2.102 |
| Proposed representation with LR | ISOBMFF metadata plus AAC PMF and TPM features | Logistic regression | Leave one device out | 8 | 97.917 | 97.895 |
