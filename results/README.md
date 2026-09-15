# Experimental Results

This directory contains aggregate, non-biometric outputs from the completed modelling experiments.

## Contents

| File | Description |
|---|---|
| `hog_linear_svc_test_classification_report.csv` | HOG and linear SVM test-set metrics |
| `cnn_colab_test_classification_report.csv` | Custom CNN test-set metrics |
| `cnn_colab_training_history.csv` | Custom CNN epoch-level learning history |
| `resnet18_test_classification_report.csv` | ResNet-18 test-set metrics |
| `resnet18_training_history.csv` | ResNet-18 staged fine-tuning history |
| `efficientnet_320_cv_oof_classification_report.csv` | EfficientNet-B0 out-of-fold class metrics |
| `efficientnet_320_cv_fold_summary.csv` | Grouped cross-validation fold metrics |
| `efficientnet_320_cv_training_history.csv` | Fold- and epoch-level training history |
| `figures/` | Confusion matrices, learning curves, and aggregate split visualizations |
| `subtype_preliminary_baseline/` | Historical image-stratified classical subtype baselines |
| `subtype_preliminary_transfer_baseline/` | Historical image-stratified frozen-embedding baselines |
| `subtype_repeated_stratified/` | Historical repeated image-stratified estimates |
| `subtype_embedding_model_selection/` | Candidate frozen-embedding model comparisons |
| `subtype_grouped_model_selection/` | Governing subject-grouped subtype development evidence |
| `quantification_foundation/` | Restored annotation and cohort coverage audit |
| `quantification_reference/` | Aggregate pattern-intensity and examiner-minutiae reference results |

Subject identifiers, subject-level predictions, fingerprint previews, trained weights, and dataset archives are intentionally excluded.

## Interpretation

The HOG, custom CNN, and ResNet-18 results use subject-disjoint held-out tests. EfficientNet-B0 is evaluated with five-fold grouped out-of-fold predictions over a separate 110-subject development cohort. Direct comparisons should account for these protocol differences.

Only `subtype_grouped_model_selection/` should be used as current subtype
development evidence. Earlier image-stratified directories are retained for
transparent research history and are explicitly marked preliminary. The
quantification outputs are descriptive cohort summaries, not identity or
clinical conclusions.
