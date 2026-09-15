"use client";

import { AlertTriangle, Binary, Fingerprint, GitBranch, Layers3 } from "lucide-react";
import { useState } from "react";

type ModelKey = "broad" | "arch" | "whorl" | "review";

const models = {
  broad: {
    label: "Broad classifier",
    role: "Primary four-class model",
    model: "EfficientNet-B0 ensemble",
    summary: "Five subject-grouped checkpoints vote on arch, left loop, right loop, and whorl.",
    tone: "primary",
    metrics: [
      ["Accuracy", "91.73%"], ["Macro F1", "88.90%"], ["Images", "1,281"], ["Subjects", "110"],
    ],
    bars: [
      ["Arch", 75.56], ["Left loop", 92.62], ["Right loop", 91.47], ["Whorl", 95.96],
    ],
    note: "Grouped out-of-fold development performance. A separate 334-image, 30-subject holdout remains locked.",
  },
  arch: {
    label: "Arch subtype",
    role: "Conditional exploratory model",
    model: "ResNet-18 embeddings + balanced logistic regression",
    summary: "Runs only after the broad model routes an impression to the arch family.",
    tone: "secondary",
    metrics: [
      ["Macro F1", "0.527"], ["Balanced accuracy", "0.567"], ["Images", "101"], ["Subjects", "31"],
    ],
    bars: [["Macro F1", 52.7], ["Balanced accuracy", 56.7]],
    note: "Repeated subject-grouped development estimate. Minority subtype behavior remains limited and requires expert review.",
  },
  whorl: {
    label: "Whorl subtype",
    role: "Conditional exploratory model",
    model: "ResNet-18 embeddings + balanced logistic regression",
    summary: "Runs only after the broad model routes an impression to the whorl family.",
    tone: "secondary",
    metrics: [
      ["Macro F1", "0.446"], ["Balanced accuracy", "0.469"], ["Images", "533"], ["Subjects", "128"],
    ],
    bars: [["Macro F1", 44.6], ["Balanced accuracy", 46.9]],
    note: "Repeated subject-grouped development estimate. The output is a prototype aid, not an autonomous conclusion.",
  },
  review: {
    label: "Review gate",
    role: "Interpretation control",
    model: "Confidence and agreement checks",
    summary: "Surfaces uncertain broad results and fragile subtype cases for qualified review.",
    tone: "warning",
    metrics: [
      ["Signals", "4"], ["Action", "Review"], ["Identity use", "None"], ["Retention", "Off"],
    ],
    bars: [["Low confidence", 100], ["Low fold agreement", 100], ["Small class margin", 100], ["Minority subtype", 100]],
    note: "These are review triggers, not performance scores. They communicate when model output should receive extra scrutiny.",
  },
} as const;

const icons = { broad: Layers3, arch: Fingerprint, whorl: Binary, review: AlertTriangle };

export function ModelExplorer() {
  const [selected, setSelected] = useState<ModelKey>("broad");
  const active = models[selected];

  return (
    <div className="model-explorer">
      <div className="model-route" aria-label="Interactive inference route">
        <div className="model-route-intro">
          <span>Inference route</span>
          <strong>Select a stage to inspect its evidence</strong>
        </div>
        <div className="route-nodes" role="tablist" aria-label="Model stages">
          {(Object.keys(models) as ModelKey[]).map((key, index) => {
            const Icon = icons[key];
            return (
              <button key={key} type="button" role="tab" aria-selected={selected === key} onClick={() => setSelected(key)}>
                <span>{String(index + 1).padStart(2, "0")}</span>
                <Icon size={18} aria-hidden="true" />
                <strong>{models[key].label}</strong>
              </button>
            );
          })}
        </div>
        <p className="route-rule"><GitBranch size={16} aria-hidden="true" /> Subtype models are conditionally routed; they do not replace the broad result.</p>
      </div>

      <article className={`model-record model-record-${active.tone}`} aria-live="polite">
        <div className="model-record-head">
          <div><span>{active.role}</span><h3>{active.model}</h3></div>
          <span className="evidence-level">{selected === "broad" ? "Primary" : selected === "review" ? "Control" : "Exploratory"}</span>
        </div>
        <p className="model-summary">{active.summary}</p>
        <dl className="model-record-metrics">
          {active.metrics.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}
        </dl>
        <div className="metric-bars" aria-label={`${active.label} metrics`}>
          {active.bars.map(([label, value]) => (
            <div className="metric-bar" key={label}>
              <div><span>{label}</span><strong>{selected === "review" ? "Trigger" : `${value}%`}</strong></div>
              <span className="metric-track"><span style={{ width: `${value}%` }} /></span>
            </div>
          ))}
        </div>
        <p className="model-record-note">{active.note}</p>
      </article>
    </div>
  );
}
