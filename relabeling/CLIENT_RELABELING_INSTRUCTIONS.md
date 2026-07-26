# Fingerprint Subtype Relabelling Instructions

## Purpose

This package is for expert review of fingerprint records whose current subtype is
generic, rare, ambiguous, or associated with more than one recorded pattern.
The objective is to produce reliable labels for later subtype-classification
research.

The package intentionally excludes the locked final holdout set. Do not add
other project images to this review.

## Security and privacy

These images are biometric research data.

- Keep the ZIP and extracted files in approved encrypted storage.
- Do not place them in public folders, public repositories, or consumer file
  sharing services.
- Do not send the unencrypted package by ordinary email.
- Give access only to named reviewers working on this project.
- Do not attempt to identify or contact any person represented in the data.
- Return the completed annotation file through the agreed secure channel.
- Retain or delete working copies according to the client's data-handling
  agreement.

## Files

- `images/`: individually numbered 320×320 review images.
- `contact_sheets/`: contact sheets for fast visual review. Each image is
  identified only by `review_id`.
- `fingerprint_relabeling_template.csv`: the annotation file to complete.
- `label_guide.txt`: concise definitions and permitted field values.
- `package_manifest.txt`: package counts and integrity information.

Do not rename image files or modify the identifier and existing-label columns
in the CSV.

## Required review fields

Complete these columns for every row:

1. `confirmed_primary_pattern`
2. `confirmed_secondary_pattern`
3. `confidence`
4. `image_quality`
5. `review_action`
6. `review_notes`
7. `reviewer_id`
8. `review_date`

Use only the permitted values listed in `label_guide.txt`. Leave
`confirmed_secondary_pattern` blank when no genuine secondary pattern is
present.

## Pattern options

- `plain_arch`
- `tented_arch`
- `left_slant_loop`
- `right_slant_loop`
- `plain_whorl`
- `central_pocket_loop_whorl`
- `double_loop_whorl`
- `accidental_whorl`
- `unclear`

The organization's official classification standard takes precedence over the
short visual descriptions supplied in this package.

## Review procedure

1. Open the image named in `image_file`.
2. Judge the fingerprint before consulting the existing label when possible.
3. Enter the most specific defensible primary pattern.
4. Add a secondary pattern only if a second genuine pattern is present.
5. Record confidence as `high`, `medium`, or `low`.
6. Record image quality as `good`, `usable`, or `poor`.
7. Choose a review action:
   - `accept` — label is sufficiently clear for modelling.
   - `adjudicate` — a second expert must decide.
   - `exclude` — image quality or ambiguity makes the record unsuitable.
8. Explain uncertain, multi-pattern, or excluded records in `review_notes`.
9. Enter the reviewer's project identifier and the review date in ISO format
   (`YYYY-MM-DD`).

## Quality control

- A second qualified reviewer should independently review all `low`-confidence,
  `adjudicate`, and multi-pattern records.
- At least 10% of other records should be double-reviewed.
- Reviewers should not resolve disagreements by averaging labels. A named
  adjudicator should make the final decision and document it.
- Before return, confirm that every row has a valid primary pattern, confidence,
  quality, action, reviewer ID, and date.
- Return only the completed CSV unless another return format was agreed.

## Important modelling note

Generic labels such as `arch` and `whorl` are not permitted as confirmed
subtypes. If the image cannot support a specific subtype, use `unclear` and
select `adjudicate` or `exclude`.
