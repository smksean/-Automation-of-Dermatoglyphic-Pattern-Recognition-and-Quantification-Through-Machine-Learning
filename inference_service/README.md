# Inference Service

This FastAPI service exposes the trained broad-pattern ensemble and optional
arch/whorl subtype classifiers to the Vercel web application. Uploaded image
bytes are decoded and processed in memory and are not written to disk.

## Local run

From the repository root, install the service dependencies and start the API:

```powershell
python -m pip install -r inference_service/requirements.txt
$env:API_TOKEN="replace-with-a-long-random-value"
python -m uvicorn inference_service.app:app --port 8000
```

The service uses the checked local broad-model directory and private subtype
artifact directory when they exist. Application startup downloads and verifies
all five broad checkpoints before `GET /health` can pass, but does not load the
broad ensemble into memory. To stay within constrained CPU-container memory,
inference loads each broad checkpoint into one reusable EfficientNet
architecture sequentially and averages the same five probability vectors used
by the local application.

## Container deployment

Build from the repository root so the Dockerfile can copy only the inference
code selected by `.dockerignore`:

```powershell
docker build -f inference_service/Dockerfile -t dermatoglyphic-inference .
docker run --rm -p 8000:7860 -e API_TOKEN="replace-with-a-long-random-value" dermatoglyphic-inference
```

For subtype support, place `arch_subtype_classifier.pkl` and
`whorl_subtype_classifier.pkl` in `inference_service/artifacts/` only inside the
private deployment context. The directory is Git-ignored. Raw images, labels,
participant metadata, and notebooks are excluded from the container context.

The included `render.yaml` defines a free Docker web service in Frankfurt. The
sequential broad runtime measured 459.2 MB peak RSS during a real broad-plus-
subtype request on the local CPU environment, compared with 558.8 MB when all
five broad models remained resident. This is close to Render's 512 MB free-tier
limit, so the free deployment is suitable for research demonstration rather
than availability-critical production.

On Render, the deployment automation stores each private classifier as an
ASCII base64 secret file named `arch_subtype_classifier.b64` and
`whorl_subtype_classifier.b64`. At runtime, the service decodes both into
ephemeral storage and verifies their fixed SHA-256 digests before loading them.
Neither classifier is included in the source repository or built image. Set
`API_TOKEN` on Render and use the same value in Vercel.

The container caches the fixed ImageNet ResNet-18 feature-extractor weights
used by subtype training. Complete two-stage requests are serialized so
overlapping uploads cannot multiply model memory on the single-worker research
deployment.

Configure the Vercel project with the backend URL and the same token:

```text
INFERENCE_API_URL=https://your-private-service.example
INFERENCE_API_TOKEN=replace-with-a-long-random-value
```
