import {
  BookOpen,
  BrainCircuit,
  Calculator,
  Cpu,
  Database,
  FileText,
  Layers3,
  LockKeyhole,
  Scale,
  ShieldAlert,
  Split,
} from "lucide-react";
import { FigureZoom } from "@/components/figure-zoom";
import { ModelExplorer } from "@/components/model-explorer";

const broadClasses = [
  { label: "Arch", precision: "75.89%", recall: "75.22%", f1: "75.56%", support: 113 },
  { label: "Left-slant loop", precision: "91.46%", recall: "93.81%", f1: "92.62%", support: 388 },
  { label: "Right-slant loop", precision: "90.69%", recall: "92.27%", f1: "91.47%", support: 401 },
  { label: "Whorl", precision: "98.07%", recall: "93.93%", f1: "95.96%", support: 379 },
];

export function MethodologySection() {
  return (
    <section className="method-band" id="methodology" aria-labelledby="method-title">
      <div className="section-heading">
        <div>
          <p className="eyebrow">02 / Methodology</p>
          <h2 id="method-title">A traceable two-stage inference design</h2>
        </div>
        <p>Broad-pattern evidence and subtype-prototype evidence remain separate because their validation protocols are not equivalent.</p>
      </div>

      <FigureZoom
        className="method-figure"
        src="/images/methodology-diagram.svg"
        alt="Two-stage fingerprint classification methodology diagram"
        width={1600}
        height={1000}
        caption={<><strong>Figure 1.</strong> Deterministic preprocessing, broad-pattern ensemble inference, conditional subtype routing, and separate evidence boundaries.</>}
      />

      <div className="method-mobile" aria-label="Two-stage fingerprint classification methodology">
        <article>
          <span>A</span>
          <div><strong>Prepare the rolled impression</strong><p>Validate the image, isolate the foreground, enhance contrast, resize, and normalize.</p></div>
        </article>
        <article>
          <span>B</span>
          <div><strong>Classify the broad pattern</strong><p>Average the four-class probabilities from five EfficientNet-B0 checkpoints.</p></div>
        </article>
        <article>
          <span>C</span>
          <div><strong>Route eligible subtype cases</strong><p>Arch and whorl results enter separate frozen-embedding logistic models; loops retain the broad result.</p></div>
        </article>
        <article>
          <span>D</span>
          <div><strong>Report evidence within scope</strong><p>Broad and subtype estimates use subject-grouped evaluation and are not forensic identity conclusions.</p></div>
        </article>
      </div>

      <div className="method-notes">
        <article><Database size={20} /><div><strong>Source and linkage</strong><p>Rolled impressions were linked to examiner-supplied ANSI/NIST pattern annotations before modelling.</p></div></article>
        <article><Split size={20} /><div><strong>Leakage control</strong><p>Broad-model partitions were grouped by subject so one individual could not enter both training and evaluation.</p></div></article>
        <article><Layers3 size={20} /><div><strong>Conditional subtypes</strong><p>Only broad arch and whorl results enter the corresponding expert-reviewed subtype prototype.</p></div></article>
      </div>
    </section>
  );
}

export function EvidenceSection() {
  return (
    <section className="evidence-band" id="evidence" aria-labelledby="evidence-title">
      <div className="section-heading">
        <div>
          <p className="eyebrow">03 / Evaluation evidence</p>
          <h2 id="evidence-title">Performance improved with transfer learning and grouped evaluation</h2>
        </div>
        <p>The reported EfficientNet result is grouped out-of-fold development performance, not a guarantee for arbitrary uploaded images.</p>
      </div>

      <div className="headline-metrics">
        <div><span>Grouped OOF accuracy</span><strong>91.73%</strong><small>EfficientNet-B0</small></div>
        <div><span>Grouped OOF macro F1</span><strong>88.90%</strong><small>Four broad classes</small></div>
        <div><span>Development images</span><strong>1,281</strong><small>Across 110 subjects</small></div>
        <div><span>Locked holdout</span><strong>334</strong><small>30 reserved subjects</small></div>
      </div>

      <div className="evidence-grid">
        <FigureZoom
          src="/results/experiment-performance.png"
          alt="Comparison of broad-pattern model accuracy and macro F1"
          width={1200}
          height={720}
          caption={<><strong>Model progression.</strong> Aggregate results improved from fixed HOG features through CNN and transfer-learning approaches.</>}
        />
        <FigureZoom
          src="/results/efficientnet-confusion-matrix.png"
          alt="EfficientNet-B0 grouped out-of-fold confusion matrix"
          width={1000}
          height={820}
          caption={<><strong>Class-level errors.</strong> Arch remained the smallest and most difficult broad category.</>}
        />
      </div>

      <div className="performance-table-wrap">
        <div className="table-heading"><strong>EfficientNet-B0 class-level performance</strong><span>Grouped out-of-fold predictions</span></div>
        <table>
          <thead><tr><th>Broad class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead>
          <tbody>{broadClasses.map((row) => <tr key={row.label}><th>{row.label}</th><td>{row.precision}</td><td>{row.recall}</td><td>{row.f1}</td><td>{row.support}</td></tr>)}</tbody>
        </table>
      </div>

      <div className="subtype-evidence">
        <div><p className="eyebrow">Exploratory subtype evidence</p><h3>Useful first-stage models, with clear minority-class limits</h3></div>
        <div className="subtype-stat"><span>Arch model</span><strong>0.527</strong><small>Grouped macro F1 | 101 images | 31 subjects</small></div>
        <div className="subtype-stat"><span>Whorl model</span><strong>0.446</strong><small>Grouped macro F1 | 533 images | 128 subjects</small></div>
        <p className="subtype-caveat">Subject identifiers were restored for every accepted label. Scores use repeated subject-grouped development folds; minority subtype performance remains limited and requires expert review.</p>
      </div>
    </section>
  );
}

export function QuantificationSection() {
  const measures = [
    { label: "Pattern intensity", value: "12.213", detail: "Mean PII across 188 complete classifiable ten-finger subjects" },
    { label: "Examiner minutiae", value: "1,129.573", detail: "Mean ten-finger total across 199 complete canonical subjects" },
    { label: "Ridge endings", value: "845.286", detail: "Mean ten-finger examiner-marked ridge-ending total" },
    { label: "Bifurcations", value: "284.186", detail: "Mean ten-finger examiner-marked bifurcation total" },
  ];

  return (
    <section className="quant-band" id="quantification" aria-labelledby="quant-title">
      <div className="section-heading">
        <div>
          <p className="eyebrow">04 / Quantification layer</p>
          <h2 id="quant-title">Measured outputs stay tied to available examiner annotations</h2>
        </div>
        <p>Pattern intensity and minutiae summaries are reported as descriptive cohort measures from the restored SD302 annotations.</p>
      </div>

      <div className="quant-layout">
        <div className="quant-measures">
          {measures.map((measure) => (
            <article key={measure.label}>
              <span>{measure.label}</span>
              <strong>{measure.value}</strong>
              <p>{measure.detail}</p>
            </article>
          ))}
        </div>

        <aside className="quant-boundary" aria-label="Ridge-count reporting boundary">
          <Calculator size={24} aria-hidden="true" />
          <h3>Total finger ridge count status</h3>
          <p>
            TFRC was assessed as a candidate endpoint, but it is not reported as
            a primary quantitative outcome because the restored SD302g records do
            not include validated expert field 9.322 ridge-count annotations.
          </p>
          <dl>
            <div><dt>Reported</dt><dd>Pattern intensity, minutiae totals, ridge endings, bifurcations</dd></div>
            <div><dt>Source</dt><dd>SD302b canonical rolls linked to SD302g examiner EFS features</dd></div>
            <div><dt>Boundary</dt><dd>No identity matching, diagnosis, or unsupported ridge-count claim</dd></div>
          </dl>
        </aside>
      </div>
    </section>
  );
}

export function ModelInformationSection() {
  const trainingStages = [
    { label: "Classical baseline", detail: "Fixed image descriptors and standard classifiers established the initial broad-pattern benchmark." },
    { label: "Custom CNN", detail: "A compact convolutional model tested whether project-specific features improved on handcrafted baselines." },
    { label: "Transfer learning", detail: "EfficientNet-B0 produced the strongest broad-pattern development result under grouped evaluation." },
    { label: "Conditional subtypes", detail: "Frozen ResNet-18 embeddings with balanced logistic regression were selected for arch and whorl subtype prototypes." },
  ];

  return (
    <section className="model-band" id="models" aria-labelledby="models-title">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Model information</p>
          <h2 id="models-title">Training process, selected models, and deployment behavior</h2>
        </div>
        <p>The application separates high-performing broad-pattern inference from exploratory subtype models so each output is interpreted at the right evidence level.</p>
      </div>

      <ModelExplorer />

      <div className="training-record">
        <div className="training-record-heading"><Cpu size={20} aria-hidden="true" /><div><span>Development record</span><strong>How model selection progressed</strong></div></div>
        <div className="training-timeline" aria-label="Training process">
          {trainingStages.map((stage, index) => (
            <details key={stage.label} open={index === 2}>
              <summary><span>{String(index + 1).padStart(2, "0")}</span><strong>{stage.label}</strong><small>{index === 2 ? "Selected broad approach" : index === 3 ? "Selected subtype approach" : "Candidate approach"}</small></summary>
              <p>{stage.detail}</p>
            </details>
          ))}
        </div>
      </div>

      <div className="model-evidence-row">
        <article>
          <BrainCircuit size={21} aria-hidden="true" />
          <strong>Input and preprocessing</strong>
          <p>Single rolled impressions are validated, enhanced, resized, normalized, and routed through the broad classifier before any subtype model is considered.</p>
        </article>
        <article>
          <Database size={21} aria-hidden="true" />
          <strong>Partition discipline</strong>
          <p>Development evaluation uses subject-grouped folds so the same person does not appear on both sides of a model assessment split.</p>
        </article>
        <article>
          <FileText size={21} aria-hidden="true" />
          <strong>Reporting boundary</strong>
          <p>Broad predictions, subtype predictions, and quantification summaries are research outputs, not identity determinations or autonomous forensic conclusions.</p>
        </article>
      </div>
    </section>
  );
}

export function StudySection() {
  return (
    <section className="study-band" id="study" aria-labelledby="study-title">
      <div className="study-content">
        <p className="eyebrow">05 / Research background</p>
        <h2 id="study-title">From examiner annotations to reproducible pattern recognition</h2>
        <p>The study investigates how classical features, convolutional networks, and transfer learning classify rolled fingerprint patterns while controlling identity leakage between development partitions.</p>
        <div className="objective-list">
          <div><BookOpen size={19} /><span>Compare classical machine learning, custom CNNs, and pretrained visual representations.</span></div>
          <div><Scale size={19} /><span>Measure accuracy, macro F1, and class-level behavior under subject-disjoint evaluation.</span></div>
          <div><LockKeyhole size={19} /><span>Preserve a locked subject holdout and keep restricted biometric records outside the public repository.</span></div>
        </div>
      </div>
      <aside className="study-facts" aria-label="Study scope and provenance">
        <h3>Study record</h3>
        <dl>
          <div><dt>Input modality</dt><dd>Single rolled fingerprint impression</dd></div>
          <div><dt>Broad classes</dt><dd>Arch, left loop, right loop, whorl</dd></div>
          <div><dt>Broad validation</dt><dd>Subject-grouped out-of-fold evaluation</dd></div>
          <div><dt>Subtype validation</dt><dd>Repeated subject-grouped evaluation</dd></div>
          <div><dt>Public release</dt><dd>Aggregate results and application code only</dd></div>
        </dl>
        <p>Subtype subject linkage was restored from the refreshed SD302 records. These remain development estimates rather than independent external-validation results.</p>
      </aside>
    </section>
  );
}

export function GovernanceSection() {
  return (
    <section className="governance-band" aria-labelledby="governance-title">
      <div><ShieldAlert size={28} /><p className="eyebrow">Responsible interpretation</p><h2 id="governance-title">Pattern classification is not identity determination</h2></div>
      <p>This application is a controlled research implementation. Uploaded biometric images must be processed transiently, predictions require qualified interpretation, and no output should be used as an autonomous forensic conclusion.</p>
      <ul>
        <li>No person identification or database matching</li>
        <li>No upload retention or training reuse by default</li>
        <li>Explicit uncertainty and expert-review triggers</li>
      </ul>
    </section>
  );
}
