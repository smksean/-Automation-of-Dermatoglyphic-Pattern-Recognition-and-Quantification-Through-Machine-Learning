# SD302 2026 Restoration Record

## Purpose

This record documents restoration of the restricted NIST Special Database 302
sources after loss of the original local research workspace. Raw biometric data
and subject-linked derivatives remain local and are excluded from Git.

## Acquired Sources

The September 2026 request supplied participant metadata, the current errata,
SD302a, SD302b, and the refreshed SD302g release. Every file was verified against
the SHA-256 value in the NIST delivery notice before use.

SD302g contains 2,380 exemplar IRR transactions, matching the count documented by
the original project. The refreshed records include:

- field `9.307`: examiner pattern classification;
- field `9.320`: core features;
- field `9.321`: delta features;
- field `9.331`: minutiae features.

These fields support restoration of subject-linked pattern labels and the
reported quantification layer based on pattern intensity and examiner-marked
minutiae.

## Restricted Layout

```text
data/raw/nist_sd302_2026/
  archives/                 Verified NIST delivery files
  linked_images/            Optional selective PNG extraction
  sd302g/                   Optional selective IRR extraction

data/processed/sd302_2026_restoration/
  irr_feature_inventory.csv
  irr_pattern_entries.csv
  efs_feature_points.csv
  subtype_subject_linkage.csv
  restoration_manifest.json
```

All paths above are ignored by Git. The archive names and internal paths are
retained unchanged to preserve traceability.

## Space-Aware Procedure

Run from the repository root:

```powershell
python scripts/restore_sd302_2026.py
```

The default run verifies all checksums, parses IRR records inside the ZIP archive,
preserves coordinate-level EFS features, and checks image membership without
extracting biometric images. Once the linkage summary has been reviewed, linked
images can be extracted explicitly:

```powershell
python scripts/restore_sd302_2026.py --extract-images
```

IRR extraction is optional because the indexer can read the records in place:

```powershell
python scripts/restore_sd302_2026.py --extract-irr
```

## Interpretation Boundary

Examiner annotations provide the basis for pattern-intensity aggregation and
minutiae summaries. Automated image-derived ridge counts remain outside the
primary reported outcomes unless supported by a qualified counting protocol and
validated reference values.

## Verified Restoration Results

The 2026-09-13 archive-first run produced the following private audit:

| Measure | Result |
|---|---:|
| SD302g IRR records | 2,380 |
| De-identified subjects | 200 |
| Linked source PNGs | 2,380 of 2,380 |
| Records with core annotations | 2,234 |
| Records with delta annotations | 2,195 |
| Records with minutiae annotations | 2,380 |
| Coordinate-level EFS feature rows | 273,296 |
| Accepted subtype rows linked to subject | 637 of 637 |
| Selectively extracted image size | 1.45 GB |

The extraction left the SD302g records inside their verified archive. This is
intentional: the parser can stream the annotation records directly, while only
the linked high-resolution PNGs occupy additional working storage.

## Subtype Metadata Recovery

The subtype review package retained a 16-character `record_key` generated from
the original subject identifier, finger position, collection, and Windows-relative
PNG path. Recreating this deterministic key from the refreshed SD302 archives
restores the private subject linkage without comparing biometric pixels. The
restoration manifest records the matched and unmatched row counts; grouped subtype
validation must use `subtype_subject_linkage.csv`, not the earlier ungrouped export.
