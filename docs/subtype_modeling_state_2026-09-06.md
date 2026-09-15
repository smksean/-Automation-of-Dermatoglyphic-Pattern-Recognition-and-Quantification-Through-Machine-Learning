# Subtype Modeling State

Date: 2026-09-06

> **Restoration update, 2026-09-13:** The limitation recorded below has been
> resolved. Fresh NIST SD302a, SD302b, and refreshed SD302g archives were
> checksum-verified, and the deterministic review `record_key` restored the
> original `subject_id` for all 637 accepted subtype rows. Future subtype
> evaluation must therefore be grouped by subject. The earlier ungrouped results
> remain preliminary historical results and should not be used as final evidence.

This document records the research state after importing the completed expert
subtype-review annotations from the private Supabase project.

## Source

The subtype annotations were exported from the private Supabase project used by
`annotation_app/`. The export joins:

- `review_items`
- `annotations`
- `annotation_events`
- the private Storage object
  `exports/subtype_labeling_latest.csv`
- the private review PNGs in the `fingerprint-review` bucket

The local export is stored under:

```text
data/processed/subtype_review_completed_2026-09-06/
```

This directory contains row-level biometric research data and is intentionally
ignored by Git.

## Completion Audit

| Measure | Count |
|---|---:|
| Review items | 672 |
| Saved annotations | 672 |
| Completed annotations | 672 |
| Pending items | 0 |
| Invalid saved annotations | 0 |
| Orphan annotations | 0 |
| Audit events | 759 |

The labeling phase is complete. However, completion does not mean that every
row is suitable for clean supervised subtype training.

## Review Outcomes

| Review action | Count |
|---|---:|
| accept | 637 |
| adjudicate | 35 |

Rows marked `adjudicate`, `exclude`, or `unclear` must remain outside the clean
first-pass training set until a named adjudicator resolves them.

## Accepted Subtype Counts

| Subtype | Count |
|---|---:|
| plain_arch | 75 |
| tented_arch | 26 |
| plain_whorl | 487 |
| central_pocket_loop_whorl | 20 |
| double_loop_whorl | 26 |
| accidental_whorl | 3 |

The first modeling pass should treat `accidental_whorl` cautiously because only
three accepted examples are currently available.

## Prepared Modeling Files

The export and preparation scripts produce:

```text
data/processed/subtype_review_completed_2026-09-06/
  subtype_labeling_latest.csv
  accepted_subtype_labels.csv
  adjudication_or_unclear_subtype_labels.csv
  subtype_review_summary.json
  review_images/
  model_input_images_320/
  subtype_modeling_dataset.csv
  subtype_modeling_first_pass_recommended.csv
  arch_subtype_modeling_dataset.csv
  whorl_subtype_modeling_dataset.csv
  subtype_modeling_summary.json
```

The `review_images/` files are the reviewer-facing PNGs and include a text
banner. They must not be used directly for model training. The modeling prep
script crops the fingerprint-only `320 x 320` region into
`model_input_images_320/` to avoid text leakage.

## Recommended First-Pass Modeling Scope

Use:

```text
subtype_modeling_first_pass_recommended.csv
```

for the first subtype-modeling pass.

This gives 634 accepted examples:

| Subtype | Count |
|---|---:|
| plain_arch | 75 |
| tented_arch | 26 |
| plain_whorl | 487 |
| central_pocket_loop_whorl | 20 |
| double_loop_whorl | 26 |

Recommended experiments:

1. Binary arch subtype model: `plain_arch` vs `tented_arch`.
2. Whorl subtype model: `plain_whorl` vs `central_pocket_loop_whorl` vs
   `double_loop_whorl`.
3. Keep `accidental_whorl` as descriptive evidence only until more cases are
   reviewed or a class-combination strategy is explicitly justified.

## Methodological Position

The subtype review set has documented non-holdout provenance. It was generated
from `data/processed/efficientnet_320_package/`, selected only generic arch and
whorl records whose `experiment_role` was not `locked_holdout`, and explicitly
excluded 136 locked-holdout generic records. The package audit documented 672
review images from 146 subjects with zero locked-holdout images included.

The Supabase export itself does not contain `subject_id`, but it preserves the
deterministic `record_key`. The 2026-09-13 restoration reproduced that key from
the refreshed NIST source path and recovered all 637 accepted rows across 144
subjects. The private join is stored at
`data/processed/sd302_2026_restoration/subtype_subject_linkage.csv`.

The grouped cohort contains 101 arch images from 31 subjects and 536 whorl images
from 128 subjects. Accidental whorl remains descriptive only because its three
images come from only three subjects. Final arch and three-class whorl model
selection must now be rerun with subject-grouped folds.

## Reproducibility Commands

Export the completed Supabase review data:

```powershell
$env:SUPABASE_URL = "https://YOUR_PROJECT.supabase.co"
$env:SUPABASE_SERVICE_ROLE_KEY = "YOUR_SERVER_SIDE_SECRET_KEY"
$env:SUPABASE_BUCKET = "fingerprint-review"
python scripts\export_completed_subtype_review_from_supabase.py
```

Prepare fingerprint-only modeling inputs:

```powershell
python scripts\prepare_subtype_modeling_dataset.py
```

Check cloud completion status:

```powershell
python scripts\check_supabase_labeling_completion.py
```

Do not commit exported data, cropped images, secrets, or row-level label files.

## Preliminary Baseline Results

The first subtype baselines were run after preparing the fingerprint-only
`320 x 320` crops. These results are preliminary prototype results. The source
package is documented as non-holdout, but the Supabase export does not include
original subject identifiers, so the current split is stratified by label rather
than grouped by subject.

### Classical Feature Baseline

Command:

```powershell
python scripts\train_preliminary_subtype_baselines.py
```

Aggregate results:

| Task | Model | Accuracy | Balanced accuracy | Macro F1 |
|---|---|---:|---:|---:|
| Arch subtype | Majority baseline | 0.731 | 0.500 | 0.422 |
| Arch subtype | Linear SVC, class balanced | 0.692 | 0.609 | 0.609 |
| Whorl subtype | Majority baseline | 0.910 | 0.333 | 0.318 |
| Whorl subtype | Linear SVC, class balanced | 0.888 | 0.325 | 0.314 |

The classical feature baseline gives a useful first signal for the arch task:
it identifies some tented-arch cases rather than predicting only the majority
plain-arch class. It does not separate the minority whorl subtypes.

### Frozen ResNet-18 Feature Baseline

Command:

```powershell
python scripts\train_preliminary_subtype_transfer_baselines.py
```

Aggregate results:

| Task | Model | Accuracy | Balanced accuracy | Macro F1 |
|---|---|---:|---:|---:|
| Arch subtype | Majority baseline | 0.731 | 0.500 | 0.422 |
| Arch subtype | ResNet-18 embeddings + balanced Linear SVC | 0.462 | 0.406 | 0.405 |
| Arch subtype | ResNet-18 embeddings + balanced logistic regression | 0.577 | 0.485 | 0.485 |
| Whorl subtype | Majority baseline | 0.910 | 0.333 | 0.318 |
| Whorl subtype | ResNet-18 embeddings + balanced Linear SVC | 0.881 | 0.367 | 0.379 |
| Whorl subtype | ResNet-18 embeddings + balanced logistic regression | 0.896 | 0.526 | 0.532 |

For the whorl task, frozen ResNet-18 embeddings improve minority-subtype
recognition compared with the majority baseline:

| Whorl subtype | Precision | Recall | F1 | Test support |
|---|---:|---:|---:|---:|
| plain_whorl | 0.943 | 0.951 | 0.947 | 122 |
| central_pocket_loop_whorl | 0.333 | 0.200 | 0.250 | 5 |
| double_loop_whorl | 0.375 | 0.429 | 0.400 | 7 |

This is a better direction than raw image features, but it remains a baseline.
The minority whorl test supports are very small, so fold-to-fold variance is
expected to be high.

### Repeated Stratified Prototype Evaluation

Command:

```powershell
python scripts\evaluate_subtype_repeated_stratified.py
```

This evaluation uses 5-fold stratified cross-validation repeated 10 times on
the documented non-holdout subtype review set. It does not use subject grouping
because `subject_id` is not present in the Supabase export.

| Prototype task | Model | Mean balanced accuracy | SD | Mean macro F1 | SD |
|---|---|---:|---:|---:|---:|
| Arch subtype | Classical features + balanced Linear SVC | 0.567 | 0.122 | 0.537 | 0.105 |
| Whorl subtype | ResNet-18 embeddings + balanced logistic regression | 0.450 | 0.082 | 0.450 | 0.073 |

Per-class repeated results:

| Task | Label | Mean precision | Mean recall | Mean F1 |
|---|---|---:|---:|---:|
| Arch subtype | plain_arch | 0.792 | 0.609 | 0.682 |
| Arch subtype | tented_arch | 0.323 | 0.525 | 0.391 |
| Whorl subtype | plain_whorl | 0.930 | 0.938 | 0.934 |
| Whorl subtype | central_pocket_loop_whorl | 0.163 | 0.160 | 0.152 |
| Whorl subtype | double_loop_whorl | 0.327 | 0.253 | 0.265 |

These repeated results are the most stable current subtype-prototype evidence.
They support continued subtype modeling but also show that the rare whorl
subtypes remain difficult under the current accepted-label counts.

### Embedding Model Selection

Command:

```powershell
python scripts\select_subtype_embedding_models.py
```

Model selection compared class-balanced classifiers on cached frozen ResNet-18
embeddings using repeated stratified evaluation. The selected practical
prototype models are:

| Task | Selected model | Mean balanced accuracy | Mean macro F1 | Macro F1 change vs notebook 08 |
|---|---|---:|---:|---:|
| Arch subtype | ResNet-18 embeddings + balanced logistic regression (`C=1`) | 0.619 | 0.619 | +0.082 |
| Whorl subtype | ResNet-18 embeddings + balanced logistic regression (`C=0.1`) | 0.471 | 0.464 | +0.014 |

Per-class selected-model results:

| Task | Label | Mean precision | Mean recall | Mean F1 |
|---|---|---:|---:|---:|
| Arch subtype | plain_arch | 0.803 | 0.851 | 0.824 |
| Arch subtype | tented_arch | 0.498 | 0.387 | 0.414 |
| Whorl subtype | plain_whorl | 0.932 | 0.929 | 0.930 |
| Whorl subtype | central_pocket_loop_whorl | 0.183 | 0.170 | 0.162 |
| Whorl subtype | double_loop_whorl | 0.311 | 0.313 | 0.300 |

Fine-tuning a full ResNet-18 on CPU was attempted, but the whorl experiment was
too slow for the current laptop environment. The selected embedding models are
therefore the most practical subtype candidates for near-term reporting and app
integration.

### Subject-Grouped Model Selection Update

Command:

```powershell
python scripts\select_subtype_grouped_models.py
```

After subject linkage was restored, five class-complete repetitions of five-fold
`StratifiedGroupKFold` were run with zero subject overlap. These results supersede
the image-stratified values above as the governing subtype evidence:

| Task | Deployment model | Images | Subjects | Mean balanced accuracy | Mean macro F1 | Fold SD |
|---|---|---:|---:|---:|---:|---:|
| Arch | ResNet-18 embeddings + balanced logistic regression (`C=1`) | 101 | 31 | 0.567 | 0.527 | 0.118 |
| Whorl | ResNet-18 embeddings + balanced logistic regression (`C=0.1`) | 533 | 128 | 0.469 | 0.446 | 0.086 |

Linear SVC (`C=0.1`) achieved the highest arch mean macro F1 at 0.534, only
0.007 above logistic regression and well within fold variability. Logistic
regression remains the deployment candidate because the application requires
probability outputs. Minority-class performance remains limited: selected-model
mean F1 is approximately 0.31 for tented arch, 0.15 for central-pocket-loop
whorl, and 0.25 for double-loop whorl.

## Selected Prototype Artifacts

The selected classifiers were trained on all accepted first-pass subtype rows
and saved privately under:

```text
data/processed/subtype_review_completed_2026-09-06/selected_subtype_model_artifacts/
```

Artifacts:

```text
arch_subtype_classifier.pkl
whorl_subtype_classifier.pkl
selected_subtype_artifacts_manifest.json
```

The helper module [`broad_classifier/subtype_inference.py`](../broad_classifier/subtype_inference.py)
loads these private artifacts, extracts frozen ResNet-18 embeddings from a
fingerprint crop, and returns subtype probabilities. It should only be invoked
after the broad classifier predicts `arch` or `whorl`; loop predictions do not
have a subtype model in the current scope.

The artifacts are intentionally not committed because they are derived from
private expert-reviewed biometric data.

## Streamlit Integration

The broad-classifier Streamlit application now treats subtype prediction as an
optional second stage. It passes the same in-memory, fingerprint-only 320 x 320
CLAHE image to the selected subtype model only when the broad ensemble predicts
`arch` or `whorl`.

The interface keeps the research claims separate:

- the broad result retains its documented subject-grouped development evidence;
- the subtype result is labelled as a conditional, non-holdout prototype;
- subtype scores below 70% are flagged for expert review;
- central-pocket-loop and double-loop whorl predictions are always flagged
  because their selected-model cross-validation performance remains weak;
- a missing or failed subtype artifact does not suppress the broad prediction;
- loop predictions do not invoke a subtype classifier.

The application reads private subtype artifacts from the default artifact
directory above or from `SUBTYPE_CLASSIFIER_MODEL_DIR` when that environment
variable is set.

## Current Modeling Interpretation

The project can now move from label collection to subtype modeling. The current
claim should be framed as a non-holdout, expert-reviewed subtype prototype. If
the private EfficientNet metadata is later restored, each review row can be
joined back to `subject_id`, `finger_position`, and experiment role, allowing
grouped subtype validation to match the leakage-control standard used in the
broad-classifier workflow.

## Metadata Join Requirement

The private metadata file required for research-grade subtype evaluation is:

```text
data/processed/efficientnet_320_package/roll_320_clahe_metadata.csv
```

This file is intentionally excluded from Git. Once it is restored from secure
private storage or regenerated from the raw workflow, run:

```powershell
python scripts\join_completed_subtype_labels_to_metadata.py
```

The script joins the accepted expert labels back to source metadata using the
deterministic `record_key` generated during review-package construction. The
joined table will be written under the private export folder:

```text
data/processed/subtype_review_completed_2026-09-06/subtype_modeling_with_private_metadata.csv
```

Only after this join should final grouped subtype experiments be run.
