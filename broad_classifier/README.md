# Broad Fingerprint-Pattern Classifier

This directory contains the standalone inference service and local Streamlit
interface for the four-class broad fingerprint-pattern model. It is separate
from the expert subtype-review application in `annotation_app/`.

## Local run

The five private checkpoints must exist under:

```text
models/efficientnet_320_cv/
  efficientnet_b0_320_fold_1.pt
  efficientnet_b0_320_fold_2.pt
  efficientnet_b0_320_fold_3.pt
  efficientnet_b0_320_fold_4.pt
  efficientnet_b0_320_fold_5.pt
```

Install the pinned dependencies and start the app on a separate port from the
annotation application:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\streamlit.exe run broad_classifier\streamlit_app.py --server.port 8502
```

Open `http://localhost:8502`.

Set `BROAD_CLASSIFIER_MODEL_DIR` to an alternative private checkpoint directory
when required. Do not commit checkpoints or uploaded biometric images.

## Verification

Run the full unit-test suite:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Load all five checkpoints and run the deterministic non-holdout smoke test:

```powershell
.venv\Scripts\python.exe scripts\verify_broad_classifier_models.py
```

The verification script reads only cross-validation metadata. It does not open
the locked-holdout arrays or metadata.

## Streamlit Community Cloud

Use the following deployment settings:

- repository: this project's GitHub repository;
- branch: `main`;
- entrypoint: `broad_classifier/streamlit_app.py`;
- Python: a version supported by the pinned PyTorch and torchvision wheels;
- app visibility: private, with only named research users invited.

Community Cloud detects `broad_classifier/requirements.txt` beside the
entrypoint, so the broad app installs only its inference dependencies rather
than the Supabase annotation dependencies in the repository root.

The app also requires all five checkpoints under
`models/efficientnet_320_cv/`. If they are delivered through Git LFS, confirm
that the deployment log shows real checkpoint downloads rather than small LFS
pointer files. Never place biometric test images or Streamlit secrets in Git.
