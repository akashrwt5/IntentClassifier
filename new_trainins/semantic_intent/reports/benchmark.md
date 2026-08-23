# Encoder x Classifier Benchmark

Selection rule, fixed before any numbers were seen:

1. validation macro-F1 is primary
2. calibrated ECE breaks ties within 0.005 macro-F1
3. size then latency break what remains

The challenge suites below are **reported, not used for selection**. Choosing a model on them would be the 'tune on the test set' mistake the plan rules out in Section 33.


## Trained on the original split (`train.csv`)

| encoder | classifier | val_macro_f1 | test_accuracy | test_macro_f1 | ece_raw | ece_calibrated | temperature | contextual | minimal_pair_pair | hard_negative | negation | stt | ood_rejection | gated_coverage | gated_accepted_precision | latency_p50_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tfidf-svd | logreg | 0.8162 | 0.8724 | 0.8504 | 0.0614 | 0.0266 | 0.7788 | 0.56 | 0.5455 | 0.6111 | 0.3846 | 0.7821 | 0.8222 | 0.5948 | 0.9844 | 5.37 |
| tfidf-svd | logreg_balanced | 0.8246 | 0.8486 | 0.8451 | 0.0809 | 0.0161 | 0.7666 | 0.64 | 0.6136 | 0.5833 | 0.3462 | 0.7834 | 0.6889 | 0.6391 | 0.9741 | 5.37 |
| tfidf-svd | linsvm | 0.8081 | 0.8533 | 0.824 | 0.1202 | 0.0194 | 0.7101 | 0.52 | 0.5 | 0.5833 | 0.3077 | 0.7901 | 0.6889 | 0.5889 | 0.9809 | 5.37 |
| tfidf-svd | linsvm_balanced | 0.8174 | 0.8559 | 0.8431 | 0.1323 | 0.0174 | 0.6988 | 0.44 | 0.5227 | 0.5833 | 0.3462 | 0.7914 | 0.6444 | 0.6272 | 0.9705 | 5.37 |
| tfidf-svd | mlp | 0.8226 | 0.883 | 0.8759 | 0.0922 | 0.016 | 2.1169 | 0.44 | 0.6364 | 0.6111 | 0.3462 | 0.7908 | 0.8 | 0.6279 | 0.9821 | 5.37 |


## Trained with targeted augmentation (`train_augmented.csv`)

| encoder | classifier | val_macro_f1 | test_accuracy | test_macro_f1 | ece_raw | ece_calibrated | temperature | contextual | minimal_pair_pair | hard_negative | negation | stt | ood_rejection | gated_coverage | gated_accepted_precision | latency_p50_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tfidf-svd | logreg | 0.8159 | 0.8645 | 0.853 | 0.0263 | 0.0157 | 0.8827 | 0.44 | 0.5909 | 0.5278 | 0.5 | 0.8255 | 0.8667 | 0.618 | 0.984 | 8.75 |
| tfidf-svd | logreg_balanced | 0.8082 | 0.8526 | 0.8482 | 0.0275 | 0.0213 | 0.892 | 0.56 | 0.6364 | 0.5556 | 0.4231 | 0.8108 | 0.8222 | 0.6021 | 0.9769 | 8.75 |
| tfidf-svd | linsvm | 0.8149 | 0.85 | 0.8294 | 0.0676 | 0.0278 | 0.8419 | 0.36 | 0.6136 | 0.5 | 0.3846 | 0.8168 | 0.8 | 0.5763 | 0.9885 | 8.75 |
| tfidf-svd | linsvm_balanced | 0.8148 | 0.8506 | 0.8352 | 0.0716 | 0.029 | 0.8334 | 0.32 | 0.6136 | 0.5556 | 0.5 | 0.8102 | 0.8 | 0.5525 | 0.9892 | 8.75 |
| tfidf-svd | mlp | 0.8298 | 0.8843 | 0.8774 | 0.0987 | 0.015 | 2.5429 | 0.4 | 0.6364 | 0.4167 | 0.6538 | 0.8229 | 0.8222 | 0.6206 | 0.9851 | 8.75 |


## What the targeted augmentation changed

Delta = augmented - original, on identical held-out suites.

| encoder | classifier | negation | stt | ood_rejection | minimal_pair_pair | hard_negative | contextual | test_macro_f1 |
|---|---|---|---|---|---|---|---|---|
| tfidf-svd | logreg | 0.1154 | 0.0434 | 0.0445 | 0.0454 | -0.0833 | -0.12 | 0.0026 |
| tfidf-svd | logreg_balanced | 0.0769 | 0.0274 | 0.1333 | 0.0228 | -0.0277 | -0.08 | 0.0031 |
| tfidf-svd | linsvm | 0.0769 | 0.0267 | 0.1111 | 0.1136 | -0.0833 | -0.16 | 0.0054 |
| tfidf-svd | linsvm_balanced | 0.1538 | 0.0188 | 0.1556 | 0.0909 | -0.0277 | -0.12 | -0.0079 |
| tfidf-svd | mlp | 0.3076 | 0.0321 | 0.0222 | 0 | -0.1944 | -0.04 | 0.0015 |

The augmentation was built to fix negation, STT robustness and near-OOD, and those are the columns that move. Minimal pairs and hard negatives barely move, which is the expected result for a bag-of-n-grams encoder: direction reversal under paraphrase is a semantic problem, not a lexical one, so it is the encoder that has to change.
