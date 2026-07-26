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

Subject identifiers, subject-level predictions, fingerprint previews, trained weights, and dataset archives are intentionally excluded.

## Interpretation

The HOG, custom CNN, and ResNet-18 results use subject-disjoint held-out tests. EfficientNet-B0 is evaluated with five-fold grouped out-of-fold predictions over a separate 110-subject development cohort. Direct comparisons should account for these protocol differences.
