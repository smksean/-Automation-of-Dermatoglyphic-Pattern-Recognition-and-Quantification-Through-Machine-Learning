# Notebook Workflow

The notebooks constitute the ordered analytical workflow for dataset construction, quality assessment, leakage-resistant partitioning, model development, and evaluation.

| Sequence | Notebook | Purpose |
|---:|---|---|
| 1 | `01_data_prep.ipynb` | Parse SD 302g EBTS/IRR annotations and link them to fingerprint images |
| 2 | `02_dataset_review_and_split.ipynb` | Audit labels, characterize the dataset, and create subject-disjoint partitions |
| 3 | `03_model_training_baseline.ipynb` | Train and evaluate the HOG and linear SVM baseline |
| 4 | `04_cnn_preprocessing_for_colab.ipynb` | Construct the 160 × 160 CNN training package |
| 5 | `04_model_training_cnn.ipynb` | Provide the local custom-CNN training workflow |
| 6 | `05_cnn_training_colab.ipynb` | Train and evaluate the custom CNN in Google Colab |
| 7 | `06_resnet18_finetuning_colab.ipynb` | Fine-tune an ImageNet-pretrained ResNet-18 |
| 8 | `07_efficientnet_320_grouped_cv_colab.ipynb` | Perform five-fold subject-grouped EfficientNet-B0 cross-validation |

Notebook outputs are cleared before version control to avoid embedding biometric images, subject-linked records, and large binary payloads. Aggregate numerical results and non-biometric figures are maintained in the repository-level `results/` directory.

All modelling partitions are constructed at subject level. The EfficientNet notebook operates only on the designated cross-validation cohort and asserts that the locked-holdout archive is absent from the Colab runtime.
