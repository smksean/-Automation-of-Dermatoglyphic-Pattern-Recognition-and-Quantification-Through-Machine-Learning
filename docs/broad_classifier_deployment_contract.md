# Broad-Pattern Classifier Deployment Contract

## Purpose

The deployment is a private research prototype that classifies one uploaded
rolled fingerprint impression into one of four broad dermatoglyphic patterns.
It is not a forensic identification system and must not be used as an
autonomous decision-making tool.

This application is separate from the expert subtype-review application in
`annotation_app/`.

## Output classes and fixed order

The class order is fixed by the training package and must not be changed:

| Class index | Internal label | Display label |
|---:|---|---|
| 0 | `arch` | Arch |
| 1 | `left_slant_loop` | Left-slant loop |
| 2 | `right_slant_loop` | Right-slant loop |
| 3 | `whorl` | Whorl |

## Accepted input

- One fingerprint image per prediction.
- PNG, JPEG, or lossless TIFF input.
- The image must contain a rolled fingerprint impression rather than a latent,
  slap, contactless, or unrelated image.
- The server must reject empty, corrupt, extremely small, or unreasonably large
  uploads before inference.

## Deterministic preprocessing

Inference must reproduce the development preprocessing exactly:

1. Decode the upload as a grayscale image.
2. Identify foreground pixels using intensity `< 245`.
3. Retain rows and columns whose foreground density exceeds `0.01`.
4. Expand the detected crop by `6%` of its largest dimension.
5. Apply CLAHE before resizing, using `clipLimit=2.0` and
   `tileGridSize=(12, 12)`.
6. Resize with aspect ratio preserved and center on a white `320 x 320` canvas.
7. Replicate the grayscale channel to three channels.
8. Convert to a tensor and apply ImageNet normalization:
   - mean: `[0.485, 0.456, 0.406]`
   - standard deviation: `[0.229, 0.224, 0.225]`

No random augmentation is permitted during inference.

## Model architecture

- Torchvision `efficientnet_b0`.
- Four-output classifier head:
  `Dropout(p=0.35) -> Linear(in_features=1280, out_features=4)`.
- The application must construct the architecture without downloading
  ImageNet weights at runtime, then load the saved trained state dictionaries.
- Inference must run in evaluation mode with gradients disabled.

## Deployment artifact

The initial deployable model is the ensemble of these five cross-validation
checkpoints:

- `efficientnet_b0_320_fold_1.pt`
- `efficientnet_b0_320_fold_2.pt`
- `efficientnet_b0_320_fold_3.pt`
- `efficientnet_b0_320_fold_4.pt`
- `efficientnet_b0_320_fold_5.pt`

The deployment repository must remain lightweight enough for the hosting
platform to clone. Checkpoint payloads may be retrieved after application
startup only from the frozen, commit-pinned source URLs. Every download must be
accepted only after its exact expected byte size and SHA-256 digest match the
deployment manifest, and valid cached files should be reused.

For each upload, calculate softmax probabilities for every checkpoint and
average the five probability vectors. The predicted class is the class with
the largest mean probability.

This ensemble rule must be frozen before the locked holdout is opened. The
reported 91.73% development accuracy is grouped out-of-fold performance, not a
measured accuracy claim for the ensemble or for arbitrary user uploads.

## User-facing result

The interface must show:

- predicted broad pattern;
- mean model probability for the predicted class;
- all four mean class probabilities;
- agreement across the five fold models;
- a preview of the deterministic preprocessed image; and
- a research-use warning.

The probability must be described as a model score, not a guarantee or a
calibrated statement of correctness. Low agreement or a small margin between
the two highest probabilities must be visibly flagged for manual review.

## Privacy and retention

- Uploaded biometric images must be processed transiently in memory.
- The application must not write uploads or predictions to disk by default.
- The application must not log image bytes, filenames, or embedded metadata.
- Cloud deployment requires private access and institutional authorization for
  biometric processing.

## Required verification before deployment

1. Verify every checkpoint's size and SHA-256 hash.
2. Load all five checkpoints with no missing or unexpected parameters.
3. Confirm the fixed class order.
4. Unit-test preprocessing against known local development images.
5. Confirm repeated inference is deterministic.
6. Confirm corrupt and unsupported uploads are rejected safely.
7. Compare app predictions with direct notebook inference on a private sample.
8. Evaluate the frozen ensemble once on the locked holdout before making a
   final independent-performance claim.

