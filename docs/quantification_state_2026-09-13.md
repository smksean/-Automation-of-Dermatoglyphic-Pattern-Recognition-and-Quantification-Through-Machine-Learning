# Quantification State

Date: 2026-09-13

## Purpose

This document records the first auditable quantification layer for the restored
SD302 research cohort. The requested parameters are treated as distinct
measurements because they have different sources and validity requirements:

- pattern intensity is calculated from ten-finger pattern classifications;
- minutiae are summarized from examiner-marked EFS annotations;
- total finger ridge count was assessed as a candidate endpoint, but requires
  validated core-to-delta ridge-count annotations or a separately qualified
  image-derived counting protocol.

## Canonical Cohort

The analysis uses the uniform SD302b baseline device V series: 1000-ppi rolled
impressions. This avoids mixing capture devices within a subject-level total.

| Eligibility measure | Count |
|---|---:|
| Restored subjects | 200 |
| Canonical impressions | 1,999 |
| Subjects with all ten canonical images | 199 |
| Subjects with ten classifiable patterns | 188 |
| Canonical unclassifiable impressions | 11 |

The 11 `UC` impressions are not assigned a guessed pattern. Their subjects are
excluded from pattern intensity but may remain in ten-finger minutiae summaries
when all ten images are present.

## Pattern Intensity

For one subject, pattern intensity is:

```text
PII = number of loops + 2 * number of whorls
```

The calculation requires exactly ten classifiable fingers and ranges from 0 to
20. This is equivalent to the percentage formulation `(2 * % whorls + % loops)
/ 10` reported in published dermatoglyphic methods.

| Statistic | Result |
|---|---:|
| Eligible subjects | 188 |
| Mean | 12.213 |
| Standard deviation | 3.872 |
| Median | 12 |
| Interquartile range | 10 to 15 |
| Range | 1 to 20 |

Method reference: [Qualitative and quantitative dermatoglyphics of CKDu](https://pmc.ncbi.nlm.nih.gov/articles/PMC6967092/).

## Minutiae

SD302g EFS field `9.331` contains examiner-marked minutia coordinates and types.
The analysis counts ridge endings and bifurcations in the canonical impressions.
These values describe supplied annotations, not every biological minutia on a
finger.

| Ten-finger measure | Subjects | Mean | Median | Interquartile range |
|---|---:|---:|---:|---:|
| All marked minutiae | 199 | 1,129.573 | 1,113 | 931 to 1,289.5 |
| Ridge endings | 199 | 845.286 | 802 | 640.5 to 1,001.5 |
| Bifurcations | 199 | 284.186 | 272 | 214.5 to 348.5 |

Across 381 same-subject, same-finger alternative-device pairs, broad-pattern
agreement was 94.23%. Minutiae counts had a median absolute difference of 20 and
a Spearman correlation of 0.726 relative to device V. Therefore image coverage,
capture conditions, and annotation context must accompany interpretation of
minutiae totals.

## Total Finger Ridge Count

The ANSI/NIST definition counts intervening ridges crossed by a straight line
from core to delta, excluding the core and delta ridges. Field `9.322` can store
these counts, but a complete scan found it absent from all 2,380 restored SD302g
IRR records. See the [ANSI/NIST field definition](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=910136)
and [NIST marking guidance](https://tsapps.nist.gov/trainingtool/AdditionalFeatures/CoreDeltaRidge.html).

The project has the required images and core/delta coordinates, so an
image-derived estimate could be explored in a separate methodological study.
In this phase, TFRC is not reported as a primary quantitative outcome because
the restored expert annotations do not include validated ridge-count values.
This keeps the reported quantification layer tied to directly available
examiner annotations and reproducible ten-finger pattern aggregation.

## Interpretation Boundary

Pattern intensity and minutiae results are descriptive research measurements.
They are not identity conclusions, clinical diagnoses, or evidence that an
uploaded image belongs to a demographic or disease group. TFRC should be
described only as a candidate endpoint outside the reported primary
quantification layer for this phase.

## Web Implementation Update

Date: 2026-09-18

The web application now keeps single-fingerprint analysis and ten-finger PII as
separate tasks. One fingerprint receives a 0, 1, or 2 intensity contribution;
a complete PII is calculated only from ten labeled positions. The ten files are
uploaded individually and processed by one batch job that loads each broad-model
checkpoint once across the set.

Uploaded-image feature analysis reports foreground coverage, ridge contrast,
sharpness, local orientation coherence, and a ridge-density proxy. These are
descriptive image-derived measurements. Examiner-marked EFS minutiae remain the
reference quantification source: automated ridge-ending/bifurcation counts,
core/delta locations, and TFRC are visibly marked as validation-required or not
reported instead of being inferred from an unqualified skeleton heuristic.
