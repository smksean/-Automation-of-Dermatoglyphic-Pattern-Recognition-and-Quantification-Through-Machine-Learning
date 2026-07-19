# Notebook Working Style

Use this style for all notebooks and notebook-like scripts in this project:

- Keep cells short and focused on one idea.
- Add a plain Markdown explanation before important code cells.
- Use readable variable names instead of clever shortcuts.
- Show quick checks after important steps.
- Avoid large code blocks unless the logic genuinely belongs together.
- Write code for a human reviewer first, then for the machine.

## Current Notebook Flow

1. `01_data_prep.ipynb` extracts labels from SD302g records and links them to images.
2. `02_dataset_review_and_split.ipynb` reviews the prepared labels and creates the clean model-candidate table.

Run each notebook section by section. Stop after major saved outputs so results can be checked before moving forward.
