import Link from "next/link";
import { ArrowDown } from "lucide-react";

export function Hero() {
  return (
    <section className="hero" id="top" aria-labelledby="hero-title">
      <div className="hero-content">
        <p className="eyebrow">Research record / DML-01</p>
        <h1 id="hero-title">Fingerprint pattern and subtype classification</h1>
        <p className="hero-copy">
          A reproducible study of broad-pattern classification and expert-labelled
          arch and whorl subtype models for rolled fingerprint impressions.
        </p>
        <Link className="text-link" href="/analysis">
          Open analysis workspace <ArrowDown size={15} aria-hidden="true" />
        </Link>
      </div>
      <dl className="study-register" aria-label="Study record">
        <div><dt>Status</dt><dd><span className="status-dot" /> Active research prototype</dd></div>
        <div><dt>Protocol</dt><dd>Version 0.2</dd></div>
        <div><dt>Last reviewed</dt><dd>12 September 2026</dd></div>
        <div><dt>Primary endpoint</dt><dd>Grouped out-of-fold macro F1</dd></div>
      </dl>
    </section>
  );
}
