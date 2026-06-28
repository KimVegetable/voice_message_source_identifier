| Method | Inspired by | Input feature | Model | Protocol | Folds | Accuracy mean | Accuracy std | Weighted F1 mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gmm_ubm | Wang et al. 2023 baseline | 39 dimensional MFCC sequences | 64 component GMM-UBM score | logo | 8 | 0.578993 | 0.085893 | 0.564318 |
| gmm_ubm | Wang et al. 2023 baseline | 39 dimensional MFCC sequences | 64 component GMM-UBM score | random5 | 5 | 0.608491 | 0.038155 | 0.603034 |
| gsv_svm | Wang et al. 2023 baseline | 64 component GMM supervectors from MFCC | GSV with RBF SVM | logo | 8 | 0.677517 | 0.055693 | 0.661827 |
| gsv_svm | Wang et al. 2023 baseline | 64 component GMM supervectors from MFCC | GSV with RBF SVM | random5 | 5 | 0.738716 | 0.023477 | 0.736147 |
| jin | Jin et al. 2019 style | AAC codebook, scalefactor, and section statistics | VT plus SVM ranking, RBF SVM | logo | 8 | 0.774306 | 0.041251 | 0.757142 |
| jin_proposed | Jin et al. 2019 proposed, adapted | AAC codebook, scalefactor, and section statistics | VT plus SVM ranking, tuned RBF SVM | logo | 8 | 0.804253 | 0.040829 | 0.796706 |
| jin | Jin et al. 2019 style | AAC codebook, scalefactor, and section statistics | VT plus SVM ranking, RBF SVM | random5 | 5 | 0.773889 | 0.027482 | 0.746157 |
| mfcc_cnn | Wang et al. 2023 baseline | 39 dimensional MFCC maps | compact MFCC-CNN | logo | 8 | 0.596354 | 0.201967 | 0.565839 |
| mfcc_cnn | Wang et al. 2023 baseline | 39 dimensional MFCC maps | compact MFCC-CNN | random5 | 5 | 0.748286 | 0.078739 | 0.728616 |
| proposed | Ours | ISOBMFF metadata plus AAC PMF and TPM features | Voting classifier, LR RF SVM | logo | 8 | 0.982639 | 0.008901 | 0.982483 |
| proposed | Ours | ISOBMFF metadata plus AAC PMF and TPM features | Voting classifier, LR RF SVM | random5 | 5 | 0.986546 | 0.00356 | 0.986529 |
| wang_proposed | Wang et al. 2023 proposed, paper based reimplementation | Raw waveform segments | 80 Sinc windows, two 60 kernel convolution layers, three layer 2048 node DNN, voting aggregation | logo | 8 | 0.041233 | 0.044328 | 0.021023 |
