# Arch and Whorl Subtype Review Instructions

## Purpose

This package contains only development-set fingerprint images whose current
main pattern is `arch` or `whorl` but whose exact subtype is not yet specified.
The existing main type is printed on every image. The reviewer must assign the
most specific defensible subtype or mark the image `unclear`.

The locked final holdout is intentionally excluded. Do not add its images to
this review or use it for model selection.

## Important label provenance

The current main types were not predicted by the machine-learning model. They
were derived from the first examiner-supplied pattern code in EBTS field
`9.307`:

- `AU` -> `arch`
- `WU` -> `whorl`

The extraction and code mapping have been checked programmatically. This does
not constitute an independent visual confirmation of every main-type label.
Some records contain alternative examiner codes; those images display an
additional warning and are marked in the CSV.

## Security and privacy

The images are biometric research data.

- Keep the package in approved encrypted storage.
- Do not upload it to public repositories or consumer file-sharing services.
- Do not send it through WhatsApp or ordinary unencrypted email.
- Give access only to named reviewers working on this project.
- Do not attempt to identify or contact represented individuals.
- Return the completed CSV through the agreed secure channel.
- Retain or delete working copies under the project's data-handling agreement.

## Package contents

- `images/arch/`: generic arch images needing `plain_arch` or `tented_arch`.
- `images/whorl/`: generic whorl images needing a specific whorl subtype.
- `contact_sheets/`: class-separated sheets for convenient review.
- `subtype_labeling_template.csv`: the only file the reviewer should edit.
- `label_guide.txt`: permitted values and condensed visual definitions.
- `broad_label_audit.txt`: what was verified about the existing main labels.
- `package_manifest.txt`: image counts and holdout exclusions.
- `CHECKSUMS_SHA256.txt`: integrity hashes for the package files.

Do not rename the images or alter the locked columns in the CSV.

## Permitted subtype labels

For an image whose current main type is `arch`:

- `plain_arch`
- `tented_arch`
- `unclear`

For an image whose current main type is `whorl`:

- `plain_whorl`
- `central_pocket_loop_whorl`
- `double_loop_whorl`
- `accidental_whorl`
- `unclear`

Left- and right-slant loops are not included because this project's current
classification scheme does not assign them a further subtype.

## Fields to complete

Complete these columns for every row:

1. `confirmed_subtype`
2. `confidence`
3. `image_quality`
4. `review_action`
5. `main_type_issue`
6. `review_notes`
7. `reviewer_id`
8. `review_date`

Permitted values:

- `confidence`: `high`, `medium`, or `low`
- `image_quality`: `good`, `usable`, or `poor`
- `review_action`: `accept`, `adjudicate`, or `exclude`
- `main_type_issue`: leave blank when the displayed main type is acceptable;
  otherwise enter `incorrect` or `uncertain`
- `review_date`: ISO format `YYYY-MM-DD`

## Review procedure

1. Match the image to its `review_id` in the CSV.
2. Confirm that the displayed main type is visually plausible.
3. Select only a subtype permitted for that main type.
4. If the subtype cannot be determined reliably, enter `unclear` rather than
   guessing.
5. If the displayed main type appears incorrect or uncertain, enter `unclear`,
   complete `main_type_issue`, choose `adjudicate`, and explain the issue.
6. Treat every image with `alternative_type_warning=yes` conservatively. These
   records already contain alternative examiner codes.
7. Use `exclude` when image quality or missing pattern area prevents reliable
   classification.
8. Complete the reviewer identifier and date.

## Quality control

- A second qualified reviewer should independently review every low-confidence,
  adjudication, exclusion, and alternative-code record.
- At least 10% of the remaining records should be double-reviewed.
- A named adjudicator should resolve disagreements; do not average labels.
- Before return, verify that every row contains valid review values.
- Return the completed `subtype_labeling_template.csv` without changing its
  filename.

The organization's official fingerprint classification standard takes
precedence over the condensed guide included in this package.
