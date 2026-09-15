FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_CACHE_DIR=/tmp/dermatoglyphic-models \
    TORCH_HOME=/opt/torch-cache \
    PORT=7860

WORKDIR /service

COPY inference_service/requirements.txt inference_service/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -r inference_service/requirements.txt && \
    mkdir -p "$TORCH_HOME" && \
    python -c "from torchvision.models import ResNet18_Weights, resnet18; resnet18(weights=ResNet18_Weights.DEFAULT)"

COPY broad_classifier broad_classifier
COPY inference_service inference_service

RUN groupadd --gid 1000 appsecrets && \
    useradd --create-home --uid 10001 --gid 1000 appuser && \
    chown -R appuser:appsecrets /service "$TORCH_HOME"
USER appuser

EXPOSE 7860
CMD ["sh", "-c", "uvicorn inference_service.app:app --host 0.0.0.0 --port ${PORT:-7860}"]
