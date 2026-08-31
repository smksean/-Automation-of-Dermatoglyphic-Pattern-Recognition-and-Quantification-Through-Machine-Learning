# Personal Laptop Handoff

This guide is for moving the project off the work computer without committing
restricted biometric data or secrets to Git.

## Current Safe Git State

The Git repository contains the reusable project work:

- notebooks and scripts that build the dataset tables, processed arrays, review
  packages, models, figures, and application artifacts
- application code for the broad classifier and private subtype-review app
- tests
- aggregate results and non-sensitive figures
- example Streamlit secret files with placeholders only
- documentation, including `docs/data_inventory_and_lineage.md`

The repository intentionally does not contain raw SD 302 data, processed
biometric arrays, subtype-review images, annotation exports, local secrets, or
model checkpoint payloads.

## Fastest Restore Path

Use this path if you are allowed to keep a private project copy of the data.

1. Clone the repository on the personal laptop.
2. Restore the private data folders into the same relative paths:

   ```text
   sd302a/
   sd302b/
   sd302d/
   ebts/
   data/
   annotation_exports/        optional local annotation state
   models/efficientnet_320_cv/ optional offline checkpoints
   ```

3. Restore secrets through a password manager, Streamlit Secrets, Supabase, or
   institution-approved private notes. Do not restore secrets through Git.
4. Create the Python environment and install dependencies:

   ```powershell
   python -m venv .venv
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

5. Run the unit tests:

   ```powershell
   .venv\Scripts\python.exe -m unittest discover -s tests -v
   ```

6. Start the subtype-review app locally:

   ```powershell
   Copy-Item .streamlit\secrets.local.example.toml .streamlit\secrets.toml
   .venv\Scripts\streamlit.exe run annotation_app\streamlit_app.py
   ```

7. Start the broad classifier locally:

   ```powershell
   .venv\Scripts\streamlit.exe run broad_classifier\streamlit_app.py --server.port 8502
   ```

The broad classifier can download its five pinned EfficientNet checkpoints when
internet access is available. Keeping `models/efficientnet_320_cv/*.pt` is only
needed for offline use.

## Regeneration Path

Use this path if the personal laptop will re-download the raw data instead of
receiving a private copy of `data/processed/`.

1. Put the official SD 302 downloads into the expected local folders:

   ```text
   sd302a/images/
   sd302b/images/
   sd302d/images/
   ebts/baseline/
   ebts/challengers/
   ```

2. Run the notebooks in the order documented in `notebooks/README.md`.
3. Rebuild the 320 x 320 EfficientNet package:

   ```powershell
   .venv\Scripts\python.exe scripts\build_roll_320_clahe_package.py
   ```

4. Rebuild the active subtype-review package if needed:

   ```powershell
   .venv\Scripts\python.exe scripts\build_unlabeled_subtype_review_package.py
   ```

5. Rebuild or verify deployment model assets:

   ```powershell
   .venv\Scripts\python.exe scripts\verify_broad_classifier_models.py
   ```

Do not use the locked-holdout package for model selection or subtype-review
package construction. It stays private and reserved until the final evaluation
plan is frozen.

## Subtype Review Continuity

The active subtype-review dataset is:

```text
data/processed/unlabeled_subtype_review_package_2026-08-04/
```

It contains 672 non-holdout review images: 115 arch and 557 whorl. These are
derived biometric images and must stay outside Git. The package exists because
generic NIST codes `AU` and `WU` provide broad labels but not exact subtypes.

Use `annotation_app/` for continued expert review. In local mode, the app reads
the private package and writes local state under `annotation_exports/`. In cloud
mode, the package is uploaded to the private Supabase `fingerprint-review`
bucket and reviewer answers are exported privately as
`exports/subtype_labeling_latest.csv`.

Only after expert review should accepted/adjudicated subtype answers become a
future subtype-labelled derived table. That table should remain private if it
contains row-level subject or biometric references.

## Before Clearing The Work Computer

Confirm all of the following:

- the latest safe Git commit is pushed
- official raw-data download links are known, or the raw data has been preserved
  in approved private storage
- `data/processed/` has either been preserved privately or can be regenerated
- Supabase URL, service role key, Streamlit secrets, and reviewer access code are
  preserved outside Git
- the subtype-review state is preserved if any real review answers have been
  collected
- the personal laptop can clone the repo and run at least one local smoke test

If any one of those is missing, do not clear the work computer yet.
