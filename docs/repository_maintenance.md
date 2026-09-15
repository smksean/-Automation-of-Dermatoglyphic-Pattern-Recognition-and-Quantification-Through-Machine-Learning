# Repository Maintenance

This repository separates reproducible public research artifacts from
restricted biometric data and machine-specific runtime files.

## Commit to Git

- source notebooks with outputs cleared;
- analysis, restoration, training, and deployment code;
- aggregate tables that contain no subject identifiers or image paths;
- non-biometric figures and documented representative examples;
- methodology, lineage, interpretation, and deployment documentation.

## Keep Local

- NIST archives and extracted fingerprint images;
- participant metadata and row-level subject linkage;
- Supabase exports and reviewer images;
- trained weights and serialized subtype classifiers;
- notebook copies ending in `_executed.ipynb`;
- environment files, access tokens, and service credentials.

The root `.gitignore`, `web_app/.gitignore`, and
`inference_service/.gitignore` enforce these boundaries. Before each research
release, inspect `git status`, scan staged files for credentials and subject
identifiers, and confirm that aggregate result files contain no row-level
biometric records.

## Notebook Convention

The numbered notebook without a suffix is the canonical source and remains
output-free in Git. Execute it locally into a sibling `_executed.ipynb` file,
inspect all cells and outputs, and publish only the aggregate tables or figures
written under `results/`. This preserves an auditable analysis narrative while
keeping private runtime evidence outside the public repository.

## Verification

Run from the repository root:

```powershell
python -m unittest discover -s tests -v
python -m compileall -q annotation_app broad_classifier inference_service scripts tests
```

Run frontend checks from `web_app/`:

```powershell
npm run lint
npm run build
```

Deployment-specific dependencies remain isolated in
`inference_service/requirements.txt` and `web_app/package.json`. The root
`requirements.txt` supports the local research and Streamlit workflows.
