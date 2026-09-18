"use client";

import {
  AlertTriangle,
  Calculator,
  Check,
  LoaderCircle,
  Printer,
  RotateCcw,
  Upload,
  X,
} from "lucide-react";
import { ChangeEvent, useState } from "react";
import {
  BatchPredictionResponse,
  broadLabels,
  patternIntensityContributions,
  PredictionResponse,
} from "@/lib/prediction";
import { PredictionRequestError, requestBatchPrediction } from "@/lib/prediction-client";

const ACCEPTED_TYPES = new Set(["image/png", "image/jpeg", "image/tiff"]);
const MAX_FILE_BYTES = 4 * 1024 * 1024;
const MAX_BATCH_BYTES = 20 * 1024 * 1024;

const fingerSlots = [
  { id: "right-thumb", hand: "Right hand", label: "Thumb", short: "RT" },
  { id: "right-index", hand: "Right hand", label: "Index", short: "RI" },
  { id: "right-middle", hand: "Right hand", label: "Middle", short: "RM" },
  { id: "right-ring", hand: "Right hand", label: "Ring", short: "RR" },
  { id: "right-little", hand: "Right hand", label: "Little", short: "RL" },
  { id: "left-thumb", hand: "Left hand", label: "Thumb", short: "LT" },
  { id: "left-index", hand: "Left hand", label: "Index", short: "LI" },
  { id: "left-middle", hand: "Left hand", label: "Middle", short: "LM" },
  { id: "left-ring", hand: "Left hand", label: "Ring", short: "LR" },
  { id: "left-little", hand: "Left hand", label: "Little", short: "LL" },
] as const;

type FingerId = (typeof fingerSlots)[number]["id"];
type FingerFiles = Partial<Record<FingerId, File>>;
type FingerResults = Partial<Record<FingerId, PredictionResponse>>;
type FingerErrors = Partial<Record<FingerId, string>>;

function validateFile(file: File): string | null {
  if (!ACCEPTED_TYPES.has(file.type)) return "Use a PNG, JPEG, or TIFF image.";
  if (file.size > MAX_FILE_BYTES) return "This image exceeds the 4 MB limit.";
  return null;
}

export function TenFingerQuantification() {
  const [files, setFiles] = useState<FingerFiles>({});
  const [results, setResults] = useState<FingerResults>({});
  const [errors, setErrors] = useState<FingerErrors>({});
  const [batchError, setBatchError] = useState<string | null>(null);
  const [batchResult, setBatchResult] = useState<BatchPredictionResponse | null>(null);
  const [progress, setProgress] = useState("");
  const [progressValue, setProgressValue] = useState(0);
  const [running, setRunning] = useState(false);

  const selectedCount = fingerSlots.filter(({ id }) => files[id]).length;
  const selectedBytes = Object.values(files).reduce((total, file) => total + (file?.size ?? 0), 0);
  const completedCount = fingerSlots.filter(({ id }) => results[id]).length;
  const isComplete = completedCount === fingerSlots.length;
  const counts = batchResult?.counts ?? { arch: 0, loop: 0, whorl: 0 };
  const pii = batchResult?.pii ?? 0;
  const reviewCount = batchResult?.reviewCount ?? 0;
  const featureReviewCount = batchResult?.featureReviewCount ?? 0;

  function selectFile(id: FingerId, event: ChangeEvent<HTMLInputElement>) {
    const candidate = event.target.files?.[0];
    event.target.value = "";
    if (!candidate) return;
    setBatchError(null);
    setBatchResult(null);
    const validationError = validateFile(candidate);
    const proposedBatchBytes = selectedBytes - (files[id]?.size ?? 0) + candidate.size;
    const batchSizeError = proposedBatchBytes > MAX_BATCH_BYTES
      ? "The complete ten-finger set must remain within 20 MB."
      : null;
    setErrors((current) => ({ ...current, [id]: validationError ?? batchSizeError ?? undefined }));
    if (validationError || batchSizeError) return;
    setFiles((current) => ({ ...current, [id]: candidate }));
    setResults((current) => ({ ...current, [id]: undefined }));
  }

  function removeFile(id: FingerId) {
    setFiles((current) => ({ ...current, [id]: undefined }));
    setResults((current) => ({ ...current, [id]: undefined }));
    setErrors((current) => ({ ...current, [id]: undefined }));
    setBatchResult(null);
    setBatchError(null);
  }

  function resetAll() {
    setFiles({});
    setResults({});
    setErrors({});
    setBatchResult(null);
    setBatchError(null);
    setProgress("");
    setProgressValue(0);
  }

  async function analyzeAll() {
    if (selectedCount !== fingerSlots.length || running) return;
    setRunning(true);
    setErrors({});
    setBatchError(null);
    setBatchResult(null);
    setResults({});
    setProgressValue(0);
    try {
      const response = await requestBatchPrediction(
        fingerSlots.map((slot) => ({ fingerId: slot.id, file: files[slot.id]! })),
        (message, value) => {
          setProgress(message);
          setProgressValue(value);
        },
      );
      const nextResults: FingerResults = {};
      for (const item of response.fingerResults) {
        nextResults[item.fingerId as FingerId] = item.prediction;
      }
      setResults(nextResults);
      setBatchResult(response);
      setProgressValue(1);
    } catch (error) {
      const message = error instanceof PredictionRequestError
        ? `${error.message}${error.detail ? ` ${error.detail}` : ""}`
        : "The model service could not complete the ten-finger batch.";
      setBatchError(message);
    }
    setProgress("");
    setRunning(false);
  }

  return (
    <div className="ten-finger-workspace" aria-labelledby="ten-finger-title">
      <div className="quant-workspace-heading">
        <div>
          <p className="eyebrow">02 / Complete quantification</p>
          <h2 id="ten-finger-title">Calculate a ten-finger Pattern Intensity Index</h2>
          <p>
            Assign one rolled impression to each finger. One optimized server job loads each
            checkpoint once across the batch, then combines the ten predictions into a PII from 0 to 20.
          </p>
        </div>
        <div className="pii-formula" aria-label="Pattern intensity formula">
          <Calculator size={22} aria-hidden="true" />
          <div><span>Formula</span><strong>Loops + (2 × whorls)</strong></div>
        </div>
      </div>

      <div className="ten-finger-layout">
        <div className="finger-entry-panel">
          <div className="tool-header">
            <div><span>01</span><strong>Assign ten impressions</strong></div>
            <small>{selectedCount} of 10 selected · {(selectedBytes / (1024 * 1024)).toFixed(1)} of 20 MB</small>
          </div>
          <div className="hand-groups">
            {["Right hand", "Left hand"].map((hand) => (
              <fieldset className="hand-group" key={hand} disabled={running}>
                <legend>{hand}</legend>
                {fingerSlots.filter((slot) => slot.hand === hand).map((slot) => {
                  const file = files[slot.id];
                  const result = results[slot.id];
                  const error = errors[slot.id];
                  return (
                    <div className={`finger-slot ${result ? "is-complete" : ""} ${error ? "has-error" : ""}`} key={slot.id}>
                      <span className="finger-code">{slot.short}</span>
                      <div className="finger-slot-copy">
                        <strong>{slot.label}</strong>
                        {result ? (
                          <span>{broadLabels[result.predictedClass]} · {Math.round(result.score * 100)}% · quality {result.featureAnalysis.quality.score}/100</span>
                        ) : error ? (
                          <span>{error}</span>
                        ) : file ? (
                          <span title={file.name}>{file.name}</span>
                        ) : (
                          <span>No image selected</span>
                        )}
                      </div>
                      {result ? (
                        <Check size={18} aria-label="Complete" />
                      ) : file ? (
                        <button type="button" className="icon-button" onClick={() => removeFile(slot.id)} aria-label={`Remove ${hand} ${slot.label}`}><X size={17} /></button>
                      ) : (
                        <label className="slot-upload">
                          <Upload size={16} aria-hidden="true" /> <span>Add</span>
                          <input type="file" accept="image/png,image/jpeg,image/tiff" onChange={(event) => selectFile(slot.id, event)} hidden />
                        </label>
                      )}
                    </div>
                  );
                })}
              </fieldset>
            ))}
          </div>
          <div className="quant-actions">
            <button className="button button-primary" type="button" onClick={analyzeAll} disabled={selectedCount !== 10 || running || isComplete}>
              {running ? <LoaderCircle className="spin" size={18} /> : <Calculator size={18} />}
              {running ? `${Math.round(progressValue * 100)}% complete` : batchError ? "Retry optimized batch" : "Analyze all ten together"}
            </button>
            {(selectedCount > 0 || completedCount > 0) && (
              <button className="button button-secondary" type="button" onClick={resetAll} disabled={running}>
                <RotateCcw size={16} /> Reset
              </button>
            )}
          </div>
          {running && (
            <div className="batch-progress" role="status">
              <span>{progress}</span>
              <progress max="1" value={progressValue}>{progressValue}</progress>
            </div>
          )}
          {batchError && <div className="batch-error" role="alert"><AlertTriangle size={17} /><span>{batchError}</span></div>}
        </div>

        <div className="pii-report-panel" aria-live="polite">
          <div className="tool-header">
            <div><span>02</span><strong>Quantification report</strong></div>
            <small>{isComplete ? "Complete" : `${completedCount} of 10 analyzed`}</small>
          </div>
          {completedCount === 0 ? (
            <div className="pii-empty">
              <Calculator size={34} strokeWidth={1.4} aria-hidden="true" />
              <strong>Complete all ten predictions to calculate PII</strong>
              <p>The files upload separately, then run together as one optimized batch. A single image cannot represent a person’s complete ten-finger pattern intensity.</p>
              <div><span>Arch<strong>0</strong></span><span>Loop<strong>1</strong></span><span>Whorl<strong>2</strong></span></div>
            </div>
          ) : (
            <div className="pii-report">
              <div className="pii-score-block">
                <div><span>Pattern Intensity Index</span><strong>{isComplete ? pii : "—"}</strong><small>{isComplete ? "out of 20" : "awaiting all ten fingers"}</small></div>
                <Calculator size={38} strokeWidth={1.3} aria-hidden="true" />
              </div>
              {batchResult && <p className="batch-runtime">Optimized batch runtime: <strong>{batchResult.processingSeconds.toFixed(1)} seconds</strong></p>}
              <div className="pattern-counts">
                <div><span>Arches</span><strong>{counts.arch}</strong><small>0 points each</small></div>
                <div><span>Loops</span><strong>{counts.loop}</strong><small>1 point each</small></div>
                <div><span>Whorls</span><strong>{counts.whorl}</strong><small>2 points each</small></div>
              </div>
              <div className="finger-result-table" role="table" aria-label="Ten-finger classification results">
                {fingerSlots.map((slot) => {
                  const result = results[slot.id];
                  return (
                    <div role="row" key={slot.id}>
                      <span role="cell">{slot.short}</span>
                      <strong role="cell">{result ? broadLabels[result.predictedClass] : "Pending"}</strong>
                      <span role="cell">{result ? `Q ${result.featureAnalysis.quality.score}` : "—"}</span>
                      <span role="cell">{result ? patternIntensityContributions[result.predictedClass] : "—"}</span>
                    </div>
                  );
                })}
              </div>
              {isComplete && (reviewCount > 0 || featureReviewCount > 0) && (
                <div className="review-callout"><AlertTriangle size={18} /><span>{reviewCount} model prediction{reviewCount === 1 ? "" : "s"} and {featureReviewCount} image-quality result{featureReviewCount === 1 ? "" : "s"} triggered review; interpret this PII cautiously.</span></div>
              )}
              {isComplete && reviewCount === 0 && featureReviewCount === 0 && (
                <div className="consistent-callout"><Check size={18} /><span>All ten model and image-quality checks completed without a review trigger.</span></div>
              )}
              {isComplete && (
                <button className="button button-secondary print-report" type="button" onClick={() => window.print()}>
                  <Printer size={16} /> Print or save report
                </button>
              )}
              <p className="pii-boundary">This is a model-derived research measure—not identity verification, diagnosis, TFRC, or an automated minutiae count.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
