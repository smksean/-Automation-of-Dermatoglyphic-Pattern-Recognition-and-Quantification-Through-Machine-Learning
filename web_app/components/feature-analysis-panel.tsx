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
    good: "Suitable",
    review: "Review advised",
    insufficient: "Insufficient",
  }[analysis.quality.grade];

  return (
    <section className="feature-analysis" aria-labelledby="feature-analysis-title">
      <div className="feature-analysis-head">
        <div>
          <span>Image-derived feature analysis</span>
          <h4 id="feature-analysis-title">Ridge visibility and evidence boundaries</h4>
        </div>
        <strong className={`quality-grade quality-${analysis.quality.grade}`}>
          {analysis.quality.score}/100 · {gradeLabel}
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
        <div><span>Foreground coverage</span><strong>{percentage(analysis.measurements.foregroundCoverage)}</strong><small>Usable print area</small></div>
        <div><span>Ridge-flow coherence</span><strong>{percentage(analysis.measurements.orientationCoherence)}</strong><small>Directional consistency</small></div>
        <div><span>Ridge-density proxy</span><strong>{percentage(analysis.measurements.ridgeDensityProxy)}</strong><small>Not a ridge count</small></div>
        <div><span>Ridge contrast</span><strong>{analysis.measurements.ridgeContrast.toFixed(1)}</strong><small>Grayscale spread proxy</small></div>
        <div><span>Sharpness</span><strong>{Math.round(analysis.measurements.sharpness).toLocaleString()}</strong><small>Edge-variance proxy</small></div>
        <div><span>Source resolution</span><strong>{analysis.measurements.sourceWidth} × {analysis.measurements.sourceHeight}</strong><small>Uploaded pixels</small></div>
      </div>

      {analysis.quality.reasons.length > 0 ? (
        <div className="feature-quality-note feature-quality-warning">
          <AlertTriangle size={17} />
          <div><strong>Quality review</strong>{analysis.quality.reasons.map((reason) => <span key={reason}>{reason}</span>)}</div>
        </div>
      ) : (
        <div className="feature-quality-note feature-quality-pass">
          <CheckCircle2 size={17} /><div><strong>Quality checks passed</strong><span>The image supports descriptive ridge-flow analysis.</span></div>
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
