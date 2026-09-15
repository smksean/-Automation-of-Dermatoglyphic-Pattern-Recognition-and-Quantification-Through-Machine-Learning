# Dermatoglyphic Research Web Application

This directory contains the standalone Next.js application intended for Vercel.
It is isolated from the working Streamlit application and does not replace any
notebook, model artifact, result, or deployment code in `broad_classifier/`.

## Current Status

- Responsive research interface: implemented.
- Upload validation and transient browser preview: implemented.
- Broad and conditional subtype report states: implemented.
- Methodology, aggregate evidence, study background, and governance views:
  implemented.
- Vercel API proxy contract: implemented at `app/api/predict/route.ts`.
- Python inference service: not connected yet.

## Production Deployment

The research interface is deployed on Vercel at:

https://webapp-nine-delta-34.vercel.app

Deployment verification covers every public route at desktop and mobile
viewports, the methodology SVG, figure enlargement, and model-stage switching.
The `/api/predict` route intentionally returns HTTP 503 until a private Python
inference service is deployed and `INFERENCE_API_URL` is configured in Vercel.

The interface never fabricates a model result. When `INFERENCE_API_URL` is not
configured, the prediction route returns a clear service-unavailable response.

## Local Run

```powershell
cd web_app
npm install
npm run dev
```

Open `http://localhost:3000`.

Quality checks:

```powershell
npm run lint
npm run build
npm run check:visual
```

The visual check uses the locally installed Microsoft Edge browser and writes
desktop/mobile review captures under `assets/source/`.

## Inference Contract

Set the private backend base URL in `.env.local`:

```text
INFERENCE_API_URL=http://127.0.0.1:8000
```

The Next.js `/api/predict` route validates one PNG, JPEG, or TIFF image up to
4 MB and forwards it to `<INFERENCE_API_URL>/predict`. The backend must return
the typed broad/subtype response declared in `lib/prediction.ts`.

Uploaded images are not written to disk by the frontend. A production backend
must preserve the same transient-processing and no-logging contract.

## Research Claims

Public images under `public/results/` are copies of aggregate repository figures.
Synthetic editorial assets are documented under `public/images/README.md`. The
formal methodology figure is code-authored from the deployment contract and
selected subtype artifact manifest. No private reviewer images, subject metadata,
or source fingerprint collections are included in this application directory.

## Interface Rationale

The interface is structured as a research record and working analysis surface,
not a promotional landing page. The September 2026 revision follows three
published design principles:

- Make claims verifiable and show who or what stands behind them, following the
  [Stanford Guidelines for Web Credibility](https://credibility.stanford.edu/guidelines/).
- Earn trust through clear identity, plain language, privacy disclosure, and
  accessible task design, following the
  [U.S. Web Design System principles](https://designsystem.digital.gov/design-principles/).
- Use constrained line lengths, mobile-first layouts, and a consistent type and
  spacing scale, following the
  [GOV.UK Design System layout guidance](https://design-system.service.gov.uk/styles/layout/).

These principles are reflected in the visible protocol register, numbered study
sections, restrained sans-serif typography, explicit validation labels, and the
separation of broad-pattern evidence from exploratory subtype evidence. Synthetic
editorial images remain documented in the asset directory but are not used in the
current interface.
