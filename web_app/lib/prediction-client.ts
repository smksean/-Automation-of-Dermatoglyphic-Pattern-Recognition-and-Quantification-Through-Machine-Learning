"use client";

import { ApiError, BatchPredictionResponse, PredictionResponse } from "@/lib/prediction";

const POLL_INTERVAL_MS = 2000;
const MAX_POLL_ATTEMPTS = 120;

type JobSubmission = { jobId: string; status: "queued" | "running" };
type JobStatus<Result> = {
  status: "queued" | "running" | "complete" | "failed";
  result?: Result;
  error?: string;
  progress?: number;
  message?: string;
};

export class PredictionRequestError extends Error {
  detail?: string;

  constructor(error: ApiError) {
    super(error.error);
    this.name = "PredictionRequestError";
    this.detail = error.detail;
  }
}

function wait(milliseconds: number) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

export async function requestPrediction(
  file: File,
  onProgress?: (message: string) => void,
): Promise<PredictionResponse> {
  onProgress?.("Submitting fingerprint securely");
  const body = new FormData();
  body.append("image", file);
  const response = await fetch("/api/predict", { method: "POST", body });
  const payload = (await response.json()) as JobSubmission | ApiError;
  if (!response.ok) throw new PredictionRequestError(payload as ApiError);

  const { jobId } = payload as JobSubmission;
  onProgress?.("Running model analysis");

  for (let attempt = 0; attempt < MAX_POLL_ATTEMPTS; attempt += 1) {
    await wait(POLL_INTERVAL_MS);
    const jobResponse = await fetch(`/api/predict/${jobId}`, { cache: "no-store" });
    const job = (await jobResponse.json()) as JobStatus<PredictionResponse> | ApiError;
    if (!jobResponse.ok) throw new PredictionRequestError(job as ApiError);

    const jobStatus = job as JobStatus<PredictionResponse>;
    if (jobStatus.status === "complete" && jobStatus.result) return jobStatus.result;
    if (jobStatus.status === "failed") {
      throw new PredictionRequestError({
        error: "Analysis request failed.",
        detail: jobStatus.error,
      });
    }
    onProgress?.(
      jobStatus.status === "queued"
        ? "Waiting for the inference service"
        : "Running model analysis",
    );
  }

  throw new PredictionRequestError({
    error: "Analysis is taking longer than expected.",
    detail: "Submit the image again after the inference service becomes available.",
  });
}

export async function requestBatchPrediction(
  images: Array<{ fingerId: string; file: File }>,
  onProgress?: (message: string, progress: number) => void,
): Promise<BatchPredictionResponse> {
  if (images.length !== 10) {
    throw new PredictionRequestError({
      error: "Ten labeled fingerprints are required for complete PII.",
    });
  }

  onProgress?.("Creating a secure ten-finger batch", 0);
  const createResponse = await fetch("/api/batch", { method: "POST" });
  const createPayload = (await createResponse.json()) as JobSubmission | ApiError;
  if (!createResponse.ok) throw new PredictionRequestError(createPayload as ApiError);
  const { jobId } = createPayload as JobSubmission;

  for (let index = 0; index < images.length; index += 1) {
    const item = images[index];
    const body = new FormData();
    body.append("image", item.file);
    onProgress?.(`Uploading ${index + 1} of 10: ${item.fingerId.replaceAll("-", " ")}`, (index + 0.25) / 10 * 0.1);
    const uploadResponse = await fetch(
      `/api/batch/${jobId}/${item.fingerId}`,
      { method: "POST", body },
    );
    const uploadPayload = (await uploadResponse.json()) as ApiError | { uploadedCount: number };
    if (!uploadResponse.ok) throw new PredictionRequestError(uploadPayload as ApiError);
  }

  onProgress?.("Starting one optimized batch analysis", 0.1);
  const startResponse = await fetch(`/api/batch/${jobId}/start`, { method: "POST" });
  const startPayload = (await startResponse.json()) as JobSubmission | ApiError;
  if (!startResponse.ok) throw new PredictionRequestError(startPayload as ApiError);

  for (let attempt = 0; attempt < MAX_POLL_ATTEMPTS * 5; attempt += 1) {
    await wait(POLL_INTERVAL_MS);
    const response = await fetch(`/api/predict/${jobId}`, { cache: "no-store" });
    const payload = (await response.json()) as JobStatus<BatchPredictionResponse> | ApiError;
    if (!response.ok) throw new PredictionRequestError(payload as ApiError);
    const job = payload as JobStatus<BatchPredictionResponse>;
    if (job.status === "complete" && job.result) return job.result;
    if (job.status === "failed") {
      throw new PredictionRequestError({
        error: "Ten-finger analysis failed.",
        detail: job.error,
      });
    }
    onProgress?.(job.message ?? "Running optimized batch analysis", job.progress ?? 0.1);
  }

  throw new PredictionRequestError({
    error: "Ten-finger analysis is taking longer than expected.",
    detail: "The batch can be submitted again after the inference service becomes available.",
  });
}
