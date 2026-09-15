# Notebook Workflow

The notebooks constitute the ordered analytical workflow for dataset construction, quality assessment, leakage-resistant partitioning, model development, and evaluation.

| Sequence | Notebook | Purpose |
|---:|---|---|
| 1 | `01_data_prep.ipynb` | Parse SD 302g EBTS/IRR annotations and link them to fingerprint images |
| 2 | `02_dataset_review_and_split.ipynb` | Audit labels, characterize the dataset, and create subject-disjoint partitions |
| 3 | `03_model_training_baseline.ipynb` | Train and evaluate the HOG and linear SVM baseline |
| 4 | `04_cnn_preprocessing_for_colab.ipynb` | Construct the 160 x 160 CNN training package |
| 5 | `04_model_training_cnn.ipynb` | Provide the local custom-CNN training workflow |
| 6 | `05_cnn_training_colab.ipynb` | Train and evaluate the custom CNN in Google Colab |
| 7 | `06_resnet18_finetuning_colab.ipynb` | Fine-tune an ImageNet-pretrained ResNet-18 |
| 8 | `07_efficientnet_320_grouped_cv_colab.ipynb` | Perform five-fold subject-grouped EfficientNet-B0 cross-validation |
| 9 | `08_subtype_modeling_preliminary.ipynb` | Review completed expert subtype labels and summarize preliminary subtype baselines |
| 10 | `09_subtype_model_selection.ipynb` | Compare subtype prototype candidates and select practical arch/whorl subtype models |
| 11 | `10_quantification_data_foundation.ipynb` | Verify restored SD302 provenance, subject linkage, and expert feature coverage for quantification |
| 12 | `11_subtype_grouped_validation.ipynb` | Rerun subtype model selection with repeated subject-grouped folds and update deployment evidence |
| 13 | `12_quantification_reference_analysis.ipynb` | Calculate pattern intensity and examiner-minutiae reference summaries, and define why TFRC is outside the primary reported endpoint |

Notebook outputs are cleared before version control to avoid embedding biometric images, subject-linked records, and large binary payloads. Local `*_executed.ipynb` copies are ignored; they provide run inspection without becoming publication artifacts. Aggregate numerical results and non-biometric figures are maintained in the repository-level `results/` directory.

All modelling partitions are constructed at subject level. The EfficientNet notebook operates only on the designated cross-validation cohort and asserts that the locked-holdout archive is absent from the Colab runtime.

The preliminary subtype notebook remains historical exploratory work. Restored
subject identifiers now support the governing repeated subject-grouped subtype
evaluation in notebook 11. Notebook 12 begins the quantification workflow while
keeping subject-level biometric records under ignored `data/` paths.
