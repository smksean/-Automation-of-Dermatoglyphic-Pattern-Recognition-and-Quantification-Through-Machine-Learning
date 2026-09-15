"use client";

import {
  AlertTriangle,
  CheckCircle2,
  FileImage,
  LoaderCircle,
  LockKeyhole,
  ScanLine,
  ShieldCheck,
  UploadCloud,
  X,
} from "lucide-react";
import { ChangeEvent, DragEvent, useEffect, useRef, useState } from "react";
import {
  ApiError,
  broadLabels,
  PredictionResponse,
  subtypeLabels,
} from "@/lib/prediction";

const ACCEPTED_TYPES = new Set(["image/png", "image/jpeg", "image/tiff"]);
const MAX_FILE_BYTES = 4 * 1024 * 1024;
const POLL_INTERVAL_MS = 2000;
const MAX_POLL_ATTEMPTS = 120;

type JobSubmission = { jobId: string; status: "queued" | "running" };
type JobStatus = {
  status: "queued" | "running" | "complete" | "failed";
  result?: PredictionResponse;
  error?: string;
};

function wait(milliseconds: number) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

function formatBytes(bytes: number) {
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function ProbabilityList({ values }: { values: Record<string, number> }) {
  return (
    <div className="probability-list">
      {Object.entries(values)
        .sort(([, first], [, second]) => second - first)
        .map(([label, value]) => (
          <div className="probability-row" key={label}>
            <div><span>{broadLabels[label as keyof typeof broadLabels] ?? subtypeLabels[label as keyof typeof subtypeLabels] ?? label}</span><strong>{(value * 100).toFixed(1)}%</strong></div>
            <progress max="1" value={value}>{value}</progress>
          </div>
        ))}
    </div>
  );
}

export function AnalysisWorkspace() {
  const inputRef = useRef<HTMLInputElement>(null);
  const previewUrlRef = useRef<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [status, setStatus] = useState<"idle" | "ready" | "running" | "result" | "error">("idle");
  const [error, setError] = useState<ApiError | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [analysisProgress, setAnalysisProgress] = useState("Preparing analysis");

  useEffect(() => () => {
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
  }, []);

  function replacePreview(candidate: File | null) {
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    const nextUrl = candidate ? URL.createObjectURL(candidate) : null;
    previewUrlRef.current = nextUrl;
    setPreviewUrl(nextUrl);
  }

  function acceptFile(candidate: File | undefined) {
    setError(null);
    setResult(null);
    if (!candidate) return;
    if (!ACCEPTED_TYPES.has(candidate.type)) {
      setFile(null);
      replacePreview(null);
      setStatus("error");
      setError({ error: "Unsupported image format.", detail: "Use PNG, JPEG, or TIFF." });
      return;
    }
    if (candidate.size > MAX_FILE_BYTES) {
      setFile(null);
      replacePreview(null);
      setStatus("error");
      setError({ error: "This image is larger than 4 MB.", detail: "Use an export that preserves the fingerprint ridges within the deployment upload limit." });
      return;
    }
    setFile(candidate);
    replacePreview(candidate);
    setStatus("ready");
  }

  function onInputChange(event: ChangeEvent<HTMLInputElement>) {
    acceptFile(event.target.files?.[0]);
    event.target.value = "";
  }

  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    acceptFile(event.dataTransfer.files?.[0]);
  }

  function clearFile() {
    setFile(null);
    replacePreview(null);
    setResult(null);
    setError(null);
    setStatus("idle");
  }

  async function analyze() {
    if (!file) return;
    setStatus("running");
    setAnalysisProgress("Submitting fingerprint securely");
    setError(null);
    const body = new FormData();
    body.append("image", file);
    try {
      const response = await fetch("/api/predict", { method: "POST", body });
      const payload = (await response.json()) as JobSubmission | ApiError;
      if (!response.ok) {
        setError(payload as ApiError);
        setStatus("error");
        return;
      }
      const { jobId } = payload as JobSubmission;
      setAnalysisProgress("Running two-stage model analysis");

      for (let attempt = 0; attempt < MAX_POLL_ATTEMPTS; attempt += 1) {
        await wait(POLL_INTERVAL_MS);
        const jobResponse = await fetch(`/api/predict/${jobId}`, { cache: "no-store" });
        const job = (await jobResponse.json()) as JobStatus | ApiError;
        if (!jobResponse.ok) {
          setError(job as ApiError);
          setStatus("error");
          return;
        }
        const jobStatus = job as JobStatus;
        if (jobStatus.status === "complete" && jobStatus.result) {
          setResult(jobStatus.result);
          setStatus("result");
          return;
        }
        if (jobStatus.status === "failed") {
          setError({ error: "Analysis request failed.", detail: jobStatus.error });
          setStatus("error");
          return;
        }
        setAnalysisProgress(
          jobStatus.status === "queued"
            ? "Waiting for the inference service"
            : "Running two-stage model analysis",
        );
      }

      setError({
        error: "Analysis is taking longer than expected.",
        detail: "Submit the image again after the inference service becomes available.",
      });
      setStatus("error");
    } catch {
      setError({ error: "Analysis request failed.", detail: "The selected image remains only in this browser session." });
      setStatus("error");
    }
  }

  return (
    <section className="analysis-band" id="analyze" aria-labelledby="analysis-title">
      <div className="section-heading compact-heading">
        <div>
          <p className="eyebrow">01 / Analysis workspace</p>
          <h2 id="analysis-title">Classify one rolled fingerprint</h2>
        </div>
        <div className="trust-row">
          <span><LockKeyhole size={16} /> No image retention</span>
          <span><ShieldCheck size={16} /> Research use only</span>
        </div>
      </div>

      <div className="analysis-layout">
        <div className="upload-tool">
          <div className="tool-header">
            <div><span>01</span><strong>Select impression</strong></div>
            <small>PNG, JPEG, TIFF | 4 MB maximum</small>
          </div>
          <input ref={inputRef} type="file" accept="image/png,image/jpeg,image/tiff" onChange={onInputChange} hidden />

          {!file ? (
            <div
              className={`dropzone ${dragging ? "is-dragging" : ""}`}
              onDragEnter={() => setDragging(true)}
              onDragLeave={() => setDragging(false)}
              onDragOver={(event) => event.preventDefault()}
              onDrop={onDrop}
            >
              <UploadCloud size={34} strokeWidth={1.5} aria-hidden="true" />
              <strong>Drop one rolled fingerprint here</strong>
              <span>The complete central pattern should be visible and not mirrored.</span>
              <button className="button button-secondary" type="button" onClick={() => inputRef.current?.click()}>
                <FileImage size={17} aria-hidden="true" /> Browse files
              </button>
            </div>
          ) : (
            <div className="image-selection">
              <div className="image-preview">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={previewUrl ?? ""} alt="Selected fingerprint preview" />
              </div>
              <div className="file-meta">
                <FileImage size={18} aria-hidden="true" />
                <div><strong>Fingerprint image ready</strong><span>{formatBytes(file.size)} | {file.type.replace("image/", "").toUpperCase()}</span></div>
                <button className="icon-button" type="button" onClick={clearFile} aria-label="Remove selected image" title="Remove selected image"><X size={18} /></button>
              </div>
            </div>
          )}

          {error && (
            <div className="inline-alert" role="alert">
              <AlertTriangle size={19} aria-hidden="true" />
              <div><strong>{error.error}</strong>{error.detail && <span>{error.detail}</span>}</div>
            </div>
          )}

          <button className="button button-primary analyze-button" type="button" disabled={!file || status === "running"} onClick={analyze}>
            {status === "running" ? <LoaderCircle className="spin" size={18} /> : <ScanLine size={18} />}
            {status === "running" ? analysisProgress : "Analyze fingerprint"}
          </button>
        </div>

        <div className="report-panel" aria-live="polite">
          <div className="tool-header">
            <div><span>02</span><strong>Analysis report</strong></div>
            <small>{status === "result" ? "Complete" : status === "running" ? "Processing" : "Awaiting analysis"}</small>
          </div>

          {!result ? (
            <div className="report-empty">
              <div className="report-icon"><ScanLine size={30} strokeWidth={1.4} /></div>
              <strong>Broad pattern first, subtype when eligible</strong>
              <p>The report will separate ensemble evidence from the exploratory arch or whorl subtype result.</p>
              <ol>
                <li><span>1</span> Four-class broad prediction</li>
                <li><span>2</span> Five-fold probability and agreement review</li>
                <li><span>3</span> Conditional subtype prediction</li>
              </ol>
            </div>
          ) : (
            <div className="report-content">
              <div className="result-heading">
                <div><span>Broad pattern</span><h3>{broadLabels[result.predictedClass]}</h3></div>
                <strong>{(result.score * 100).toFixed(1)}%</strong>
              </div>
              <div className="report-metrics">
                <div><span>Fold agreement</span><strong>{Math.round(result.agreement * 5)} of 5</strong></div>
                <div><span>Top-two margin</span><strong>{(result.topTwoMargin * 100).toFixed(1)} pts</strong></div>
                <div><span>Processing</span><strong>{result.processingSeconds.toFixed(2)} s</strong></div>
              </div>
              <ProbabilityList values={result.probabilities} />
              {result.subtype && (
                <div className="subtype-result">
                  <span>Conditional subtype</span>
                  <div><h4>{subtypeLabels[result.subtype.predictedSubtype]}</h4><strong>{(result.subtype.score * 100).toFixed(1)}%</strong></div>
                  <ProbabilityList values={result.subtype.probabilities} />
                </div>
              )}
              {(result.needsReview || result.subtype?.needsReview) ? (
                <div className="review-callout"><AlertTriangle size={18} /><span>Qualified review is recommended for this result.</span></div>
              ) : (
                <div className="consistent-callout"><CheckCircle2 size={18} /><span>No model-level review trigger was activated.</span></div>
              )}
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
