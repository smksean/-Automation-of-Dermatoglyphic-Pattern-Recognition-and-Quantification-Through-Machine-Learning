# Automation of Dermatoglyphic Pattern Recognition and Quantification Through Machine Learning

**Researcher:** Emudianughe Orevaoghene Betsy  
**Matric Number:** 244664  
**Project Type:** Feasible project planner, dataset guide, classifier plan, and deployment roadmap

## 1. Project Overview

This project will build a web-based dermatoglyphic fingerprint analysis system using NIST SD302 fingerprint images and SD302g EBTS/IRR annotations.

The system will:

- Prepare a labelled fingerprint dataset from SD302g `.irr` records.
- Train a fingerprint pattern classifier.
- Support broad pattern classification and subtype classification.
- Extract dermatoglyphic features such as cores, deltas, and minutiae where available.
- Generate a structured report through a web application.

This is now a **classification-first project**, because the workspace has confirmed usable pattern labels in the raw SD302g records.

## 2. Why the Plan Changed

The earlier plan was cautious because the image folders themselves do not contain labels like arch, loop, or whorl. After inspecting the raw SD302g `.irr` records, the labels were found in EBTS field `9.307`.

That means the project does not need to invent labels from scratch. The labels already exist in coded form inside the raw data.

Examples:

- `AU` = arch
- `LS` = left-slant loop
- `RS` = right-slant loop
- `WU` = whorl
- `UC` = unclassifiable
- `AU+PA` = plain arch
- `AU+TA` = tented arch
- `WU+PW` = plain whorl
- `WU+CP` = central pocket loop whorl
- `WU+DL` = double loop whorl
- `WU+AW` = accidental whorl

## 3. Confirmed Workspace Data

Local data currently available:

| Data Source | Description | Count |
|---|---|---:|
| `sd302a` | Challenger rolled fingerprint PNG images | 13,630 PNGs |
| `sd302b` | Baseline rolled, slap, and slap-segmented PNG images | 13,796 PNGs |
| `sd302d` | Auxiliary plain fingerprint PNG images | 5,141 PNGs |
| `ebts` | SD302g EBTS/IRR records with EFS annotations | 2,380 IRR files |

Overall inventory:

- 30,568 PNG files
- 2,380 IRR files
- 2,380 IRR files with `9.307` pattern labels
- 2,598 total repeated `9.307` pattern entries
- 203 IRR files with more than one pattern entry

## 4. Confirmed Pattern Label Counts

These are the final counts extracted from all repeated `9.307` entries in the current workspace:

| Label | Meaning | Count |
|---|---|---:|
| `RS` | Right-slant loop | 785 |
| `LS` | Left-slant loop | 734 |
| `WU` | Whorl, subclass unknown | 717 |
| `AU` | Arch, subclass unknown | 177 |
| `AU+TA` | Tented arch | 66 |
| `WU+PW` | Plain whorl | 39 |
| `WU+DL` | Double loop whorl | 22 |
| `UC` | Unclassifiable | 20 |
| `AU+PA` | Plain arch | 18 |
| `WU+AW` | Accidental whorl | 12 |
| `WU+CP` | Central pocket loop whorl | 8 |

This supports a real classifier. The broad classes are strong enough for training, while some subtypes are small and need careful handling.

## 5. Aim

To develop and deploy a web-based system that classifies dermatoglyphic fingerprint patterns and produces structured feature reports using NIST SD302 fingerprint images and SD302g EFS annotation labels.

## 6. Objectives

1. Parse SD302g `.irr` files and extract pattern labels from field `9.307`.
2. Link each extracted label to its matching fingerprint PNG image.
3. Build a clean training dataset for fingerprint pattern classification.
4. Train a broad dermatoglyphic classifier for arch, loop, whorl, and unclassifiable patterns.
5. Train or prototype subtype classification for plain arch, tented arch, whorl subtypes, and left/right loop classes.
6. Convert left/right loop labels into radial/ulnar loop labels where hand and finger metadata support it.
7. Extract available feature information from EFS fields such as cores, deltas, and minutiae.
8. Add a manual visual verification workflow so a small sample of labels can be checked by a human reviewer.
9. Build a web application for upload, prediction, feature summary, and report generation.
10. Validate the classifier using standard machine learning metrics.

## 7. Project Scope

### In Scope

- Dataset preparation from local SD302 files
- Label extraction from SD302g `9.307`
- Matching IRR records to PNG images
- Manual review sample generation
- Broad pattern classification
- Subtype classification prototype
- Feature extraction from annotations and images
- Web-based reporting
- Model validation and documentation

### Out of Scope for the One-Month Build

- Full clinical diagnosis
- Fully validated palm a-b ridge count unless SD302c palm data is added
- Perfect subtype accuracy for very small classes such as `WU+CP`, `WU+AW`, and `WU+DL`
- Claiming medical interpretation without expert review

## 8. Dataset Preparation Plan

The first notebook is:

`notebooks/01_data_prep.ipynb`

It prepares the data by:

- Reading all `.irr` files
- Extracting `9.307` labels
- Parsing subject ID, device, resolution, capture type, and finger position
- Matching labels to local PNG files
- Saving processed CSV files
- Creating a manual review sample

Generated data files:

| File | Purpose |
|---|---|
| `data/processed/irr_image_labels.csv` | One row per IRR/image record |
| `data/processed/irr_pattern_entries.csv` | One row per repeated `9.307` entry |
| `data/processed/pattern_label_counts.csv` | Label count summary |
| `data/processed/manual_review_sample.csv` | Small review sample |
| `data/processed/manual_review_visual_sample.csv` | Review sample with image paths |
| `data/processed/manual_review_images/` | Labelled images for client review |
| `data/processed/manual_review_contact_sheet.png` | Large visual contact sheet |
| `data/processed/manual_review_package.zip` | Client-ready review package |

## 9. Manual Label Verification

Manual verification is needed because the labels come from encoded raw `.irr` records. The data already contains labels, but a human check confirms that:

- The parser extracted the correct code.
- The code was linked to the correct image.
- The visible fingerprint pattern agrees with the extracted label.
- Ambiguous or multi-label records are identified early.

A small labelled image package has been created for client review. Each review image includes the fingerprint and a visible label banner.

Important review note:

Some IRR records contain multiple repeated `9.307` entries. These should be treated carefully. For training, we should either:

- use only single-label records first, or
- create a multi-label strategy for records with more than one pattern entry.

The safest first classifier should use clean single-label examples.

## 10. Classification Plan

### Stage 1: Broad Classifier

Train a broad classifier using the main pattern labels:

| Model Class | Source Labels |
|---|---|
| Arch | `AU`, `AU+PA`, `AU+TA` |
| Left-slant loop | `LS` |
| Right-slant loop | `RS` |
| Whorl | `WU`, `WU+PW`, `WU+CP`, `WU+DL`, `WU+AW` |
| Unclassifiable | `UC` |

This is the most realistic first model because the class counts are stronger.

### Stage 2: Subtype Classifier

Train or prototype subtype classification:

| Subtype | Label |
|---|---|
| Plain arch | `AU+PA` |
| Tented arch | `AU+TA` |
| Left-slant loop | `LS` |
| Right-slant loop | `RS` |
| Plain whorl | `WU+PW` |
| Central pocket loop whorl | `WU+CP` |
| Double loop whorl | `WU+DL` |
| Accidental whorl | `WU+AW` |

Subtype classification is feasible, but class imbalance must be handled carefully.

Recommended approaches:

- Use class weights.
- Use image augmentation.
- Report per-class metrics.
- Keep broad-class performance as the main success metric.
- Treat very small subtype results as prototype results, not final clinical-grade results.

### Stage 3: Radial/Ulnar Conversion

The dataset gives `LS` and `RS`, not direct radial/ulnar labels.

To convert to radial or ulnar loop:

- use finger position,
- infer hand side from position code,
- combine hand side with slant direction.

This should be implemented as a rule-based post-processing step after loop prediction.

## 11. Quantification and Feature Plan

The original slides included quantification, and that remains part of the project.

From SD302g annotations and image processing, the system can support:

- Core extraction
- Delta extraction
- Minutiae extraction
- Ridge ending count
- Bifurcation count
- Foreground area
- Ridge density proxy
- Skeleton length proxy
- Image quality notes

Possible dermatoglyphic indices:

- **TRC:** possible as an approximate or partial measure when core/delta and ridge count logic are reliable.
- **PII:** possible from delta/triradius count summaries where available.
- **a-b ridge count:** requires palm data, likely SD302c, and should remain future work unless that dataset is added.

## 12. Web Application Plan

The web application should provide:

- Login or simple authenticated access
- Fingerprint upload
- Image preview
- Preprocessing preview
- Pattern prediction
- Subtype prediction where supported
- Confidence score
- Feature summary
- Downloadable report
- Admin/researcher batch mode

Report output should include:

- Predicted broad class
- Predicted subtype
- Extracted or predicted core/delta/minutiae summary
- Ridge count notes if available
- Image quality notes
- Model limitations
- Recommendation for expert interpretation where needed

## 13. One-Month Development Plan

| Week | Focus | Deliverables |
|---|---|---|
| Week 1 | Data preparation and manual review | Parsed labels, image-label CSVs, review package, cleaned single-label dataset |
| Week 2 | Preprocessing and baseline classifier | Image loader, train/validation/test split, first broad classifier, baseline metrics |
| Week 3 | Subtype and feature extraction | Subtype prototype, core/delta/minutiae extraction, feature summary table |
| Week 4 | Web app, report, and validation | Upload interface, prediction API, report generation, final validation summary |

## 14. Methodology

### Step 1: Build Labelled Dataset

Use `9.307` from SD302g to create the labelled dataset.

Recommended fields:

- `subject_id`
- `device`
- `resolution`
- `capture_type`
- `finger_position`
- `irr_path`
- `png_path`
- `primary_label`
- `all_pattern_labels`
- `broad_class`
- `subtype`
- `num_pattern_labels`

### Step 2: Clean Training Data

Start with records that:

- have a matching PNG image,
- have a usable label,
- are single-label or have a clear primary label,
- pass basic image quality checks.

Keep multi-label records separate for later review.

### Step 3: Preprocess Images

Apply:

- grayscale loading,
- resizing,
- contrast normalisation,
- ridge enhancement,
- region-of-interest cropping,
- augmentation for minority classes.

### Step 4: Train Classifier

Start simple:

- baseline CNN or transfer-learning model,
- broad labels first,
- subtype classifier second.

Metrics:

- accuracy,
- precision,
- recall,
- F1-score,
- confusion matrix,
- per-class performance.

### Step 5: Add Feature Extraction

Use annotation fields and computer vision to extract:

- cores,
- deltas,
- minutiae,
- ridge endings,
- bifurcations,
- quality indicators.

### Step 6: Build Web Report

The web report should combine:

- classification output,
- feature output,
- confidence score,
- limitations,
- reviewer notes if available.

## 15. Validation Plan

| Component | Validation Method |
|---|---|
| Label extraction | Compare extracted counts against expected `9.307` counts |
| Image matching | Spot-check image paths against IRR filenames |
| Manual review | Ask client/reviewer to inspect labelled image samples |
| Broad classifier | Accuracy, precision, recall, F1-score, confusion matrix |
| Subtype classifier | Per-class F1-score and imbalance-aware reporting |
| Multi-label records | Review separately before training |
| Feature extraction | Visual check of core/delta/minutiae overlays |
| Web app | Test upload, prediction, report download, and error handling |

## 16. Expected Outcome

The realistic expected output is:

- A labelled SD302-based dermatoglyphic dataset.
- A trained broad fingerprint pattern classifier.
- A subtype classifier prototype.
- A manual review workflow for label verification.
- A feature extraction and quantification summary pipeline.
- A web application for upload, prediction, and reporting.
- A documented validation report.

This directly supports the original slide aim: automation of dermatoglyphic pattern recognition and quantification through machine learning.

## 17. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Small subtype classes | Weak subtype performance | Use broad classifier first, class weighting, augmentation, and cautious reporting |
| Multi-label IRR records | Confusing training labels | Separate multi-label records and review them manually |
| Left/right loop is not radial/ulnar | Possible terminology mismatch | Convert using hand/finger metadata as post-processing |
| Palm a-b ridge count requires palm data | Cannot fully validate a-b count from current workspace | Mark as future work unless SD302c is added |
| Biometric data sensitivity | Privacy and ethics concerns | Use data under NIST terms, avoid unnecessary redistribution, secure uploads |
| One-month timeline | Limited polish and validation depth | Prioritise classifier, review pack, core feature summaries, and deployable demo |

## 18. Ethical and Practical Considerations

- Use SD302 data only under the agreed NIST terms.
- Do not redistribute raw dataset files externally.
- Share only limited review packages when appropriate for internal/client verification.
- Do not claim medical diagnosis from automated outputs.
- Clearly state that subtype predictions require expert interpretation.
- Avoid storing uploaded biometric images longer than necessary.
- Document model limitations and class imbalance.

## 19. Budget Summary

| Budget Item | Description and Source | Estimated Cost |
|---|---|---:|
| Local SD302 Dataset | Already available in workspace | NGN 0 |
| Development Libraries | Python, OpenCV, scikit-image, scikit-learn, FastAPI, React | NGN 0 |
| Development IDE | VS Code / Cursor | NGN 0 |
| Code Repository | Git and GitHub | NGN 0 |
| Hosting | Render.com or equivalent free tier | NGN 0 |
| Database | SQLite locally or free PostgreSQL tier | NGN 0 |
| PDF/Report Generation | Open-source library | NGN 0 |
| Operating Expenses | Internet, electricity, software, interim reports | NGN 150,000 |
| Dissemination Expenses | Journal submission fees, if applicable | NGN 100,000 |
| Equipment Costs | Existing machine assumed | NGN 0 |
| Travel Costs | No direct cost listed | NGN 0 |
| **Total Estimated Project Expenditure** | Operating and dissemination expenses | **NGN 250,000** |

## 20. Final Project Statement

This project will deliver a feasible dermatoglyphic fingerprint classifier and reporting system using NIST SD302 data. The system will extract coded pattern labels from SD302g EFS annotations, link them to fingerprint images, train broad and subtype classifiers, support manual label verification, extract dermatoglyphic features, and present results through a web-based report.

## 21. References

- Ashbaugh, D. R. (1999). *Quantitative-qualitative friction ridge analysis: An introduction to basic and advanced ridgeology*. CRC Press.
- Bazen, A. M., & Gerez, S. H. (2002). Systematic methods for the computation of the directional fields and singular points of fingerprints. *IEEE Transactions on Pattern Analysis and Machine Intelligence, 24*(7), 905-919. https://doi.org/10.1109/TPAMI.2002.1017618
- Biewald, L. (2020). Experiment tracking with Weights and Biases. https://www.wandb.com
- Cappelli, R., Ferrara, M., & Maltoni, D. (2010). Minutia cylinder-code: A new representation and matching technique for fingerprint recognition. *IEEE Transactions on Pattern Analysis and Machine Intelligence, 32*(12), 2128-2141. https://doi.org/10.1109/TPAMI.2010.30
- Cummins, H., & Midlo, C. (1943). *Finger prints, palms and soles: An introduction to dermatoglyphics*. Blakiston.
- FastAPI. (2021). FastAPI framework. https://fastapi.tiangolo.com
- Fiumara, G., Flanagan, P., Grantham, J., Ko, K., Marshall, K., Schwarz, M., Tabassi, E., Woodgate B., & Boehnen, C. (2019). *NIST Special Database 302: Nail to Nail Fingerprint Challenge*. NIST Technical Note 2007. https://doi.org/10.6028/NIST.TN.2007
- Fiumara, G., Schwarz, M., Heising, J., Peterson, J., Flanagan, P., & Marshall, K. (2021). *NIST Special Database 302 Supplemental Release of Latent Annotations*. NIST Technical Note 2190. https://doi.org/10.6028/NIST.TN.2190
- Fiumara, G., Tabassi, E., Flanagan, P., Grantham, J., Ko, K., Marshall, K., Schwarz, M., Woodgate, B., & Boehnen, C. (2018). *Nail to Nail Fingerprint Challenge: Prize Analysis*. NIST Interagency Report 8210. https://doi.org/10.6028/NIST.IR.8210
- Galton, F. (1892). *Finger prints*. Macmillan.
- Guo, T., Xu, J., Shi, Z., Shi, D., & Chen, J. (2019). Fingerprint pattern classification using convolutional neural networks. *Neural Computing and Applications, 32*(15), 11587-11601. https://doi.org/10.1007/s00521-019-04658-7
