import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Crosshair,
  GitBranch,
  Ruler,
} from "lucide-react";
import { FeatureAnalysis } from "@/lib/prediction";

function percentage(value: number) {
  return `${Math.round(value * 100)}%`;
}

export function FeatureAnalysisPanel({ analysis }: { analysis: FeatureAnalysis }) {
  const gradeLabel = {
    good: "Screen passed",
    review: "Review advised",
    insufficient: "Screen failed",
  }[analysis.quality.grade];

  return (
    <section className="feature-analysis" aria-labelledby="feature-analysis-title">
      <div className="feature-analysis-head">
        <div>
          <span>Quantification layer · image-derived</span>
          <h4 id="feature-analysis-title">Quantitative feature profile</h4>
          <p>Deterministic descriptors computed from the uploaded impression, reported separately from model probabilities and examiner annotations.</p>
        </div>
        <strong className={`quality-grade quality-${analysis.quality.grade}`}>
          Heuristic screen {analysis.quality.score}/100 · {gradeLabel}
        </strong>
      </div>

      {analysis.overlayDataUrl && (
        <figure className="feature-overlay">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={analysis.overlayDataUrl} alt="Fingerprint ridge-flow quality overlay" />
          <figcaption><i /> Foreground boundary <i /> Local ridge-flow direction</figcaption>
        </figure>
      )}

      <div className="feature-metrics">
        <div><span>Foreground coverage</span><strong>{percentage(analysis.measurements.foregroundCoverage)}</strong><small>Foreground area / cropped area</small></div>
        <div><span>Orientation coherence</span><strong>{analysis.measurements.orientationCoherence.toFixed(3)}</strong><small>Mean block coherence, 0–1</small></div>
        <div><span>Ridge-density proxy</span><strong>{analysis.measurements.ridgeDensityProxy.toFixed(3)}</strong><small>Ridge pixels / foreground pixels</small></div>
        <div><span>Ridge contrast</span><strong>{analysis.measurements.ridgeContrast.toFixed(1)}</strong><small>Foreground grayscale SD</small></div>
        <div><span>Sharpness proxy</span><strong>{Math.round(analysis.measurements.sharpness).toLocaleString()}</strong><small>Variance of Laplacian</small></div>
        <div><span>Source dimensions</span><strong>{analysis.measurements.sourceWidth} × {analysis.measurements.sourceHeight}</strong><small>Pixels (width × height)</small></div>
      </div>

      <p className="screening-method">
        <strong>Screening-score method.</strong> Weighted heuristic: foreground coverage 20%, ridge contrast 25%, sharpness 25%, and orientation coherence 30%. It is not a calibrated NFIQ quality score.
      </p>

      {analysis.quality.reasons.length > 0 ? (
        <div className="feature-quality-note feature-quality-warning">
          <AlertTriangle size={17} />
          <div><strong>Quality review</strong>{analysis.quality.reasons.map((reason) => <span key={reason}>{reason}</span>)}</div>
        </div>
      ) : (
        <div className="feature-quality-note feature-quality-pass">
          <CheckCircle2 size={17} /><div><strong>Heuristic screen passed</strong><span>The image supports descriptive ridge-flow analysis within the stated scope.</span></div>
        </div>
      )}

      <div className="requested-feature-status">
        <article>
          <GitBranch size={18} />
          <div><span>Minutiae</span><strong>Validation required</strong><p>{analysis.minutiae.reason}</p></div>
        </article>
        <article>
          <Crosshair size={18} />
          <div><span>Core and delta</span><strong>Validation required</strong><p>{analysis.landmarks.reason}</p></div>
        </article>
        <article>
          <Ruler size={18} />
          <div><span>Finger ridge count / TFRC</span><strong>Not reported</strong><p>{analysis.ridgeCount.reason}</p></div>
        </article>
      </div>

      <p className="feature-scope"><Activity size={14} /> {analysis.scope}</p>
    </section>
  );
}
