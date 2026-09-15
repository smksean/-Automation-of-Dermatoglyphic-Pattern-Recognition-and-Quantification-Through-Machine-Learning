# Web Application Image Assets

These images were generated as synthetic editorial artwork for the standalone
research web application. They do not contain study participants, source-dataset
fingerprints, reviewer images, or model-evaluation evidence.

## Assets

| Web asset | Intended use | Source master |
|---|---|---|
| `hero-fingerprint-lab.webp` | First-page hero and project identity | `../../assets/source/hero-fingerprint-lab.png` |
| `methodology-pipeline.webp` | Preprocessing and inference-method section | `../../assets/source/methodology-pipeline.png` |
| `research-context.webp` | Research background and study-context section | `../../assets/source/research-context.png` |
| `methodology-diagram.svg` | Exact two-stage model and validation methodology | `../../assets/source/methodology-diagram-preview.png` |

The WebP files are the deployment versions. PNG files are retained under
`web_app/assets/source/` as high-quality masters and are outside the public
deployment directory.

## Generation Record

All three assets were generated with the built-in OpenAI image-generation tool
on 2026-09-12. The shared direction requested bright, academically credible
scientific imagery; synthetic ridge structures; deep green, neutral, and muted
coral accents; and no people, identities, readable text, logos, or crime-scene
associations.

- Hero prompt: a wide optical fingerprint-analysis scene with the subject on the
  right and clear copy space on the left.
- Methodology prompt: a left-to-right sequence from raw synthetic impression to
  enhanced ridge image to restrained machine-learning interpretation.
- Research-context prompt: an overhead research workspace connecting classical
  dermatoglyphic study materials with modern computational analysis.

These assets must not be cited as examples of classifier performance. The final
site should use project-generated plots, documented metrics, and explicitly
approved de-identified samples for empirical claims.

## Formal Methodology Diagram

`methodology-diagram.svg` is a code-authored scientific figure, not generated
artwork. Its pipeline details come from
`docs/broad_classifier_deployment_contract.md`; its broad development metrics
come from the repository `README.md`; and its subtype models, class counts, and
repeated-stratified macro F1 values come from the private selected-artifact
manifest. The figure deliberately reports broad and subtype evidence in
separate panels because their validation designs are different.

The earlier pastel dashboard-style concept is preserved under
`web_app/assets/source/methodology-diagram-concept-v1.svg`. The active diagram
uses a restrained publication style with square technical boxes, thin rules,
compact typography, and colorblind-safe blue/orange stage markers.
