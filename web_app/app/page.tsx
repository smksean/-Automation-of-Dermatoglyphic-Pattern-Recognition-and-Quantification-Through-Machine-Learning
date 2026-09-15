import Link from "next/link";
import { ArrowRight, BarChart3, BookOpen, BrainCircuit, Calculator, Microscope, ScanLine } from "lucide-react";
import { Hero } from "@/components/hero";

const routes = [
  { href: "/analysis", label: "Analysis workspace", detail: "Upload one rolled impression and run the two-stage classifier.", icon: ScanLine },
  { href: "/methods", label: "Methodology", detail: "Review preprocessing, broad inference, subtype routing, and evidence boundaries.", icon: Microscope },
  { href: "/results", label: "Evaluation results", detail: "Inspect grouped broad-pattern and exploratory subtype performance.", icon: BarChart3 },
  { href: "/quantification", label: "Quantification", detail: "See pattern intensity and examiner-marked minutiae summaries.", icon: Calculator },
  { href: "/models", label: "Model information", detail: "Document training process, selected models, artifacts, and deployment logic.", icon: BrainCircuit },
  { href: "/study", label: "Study notes", detail: "Read study scope, provenance, governance, and interpretation boundaries.", icon: BookOpen },
];

export default function Home() {
  return (
    <>
      <Hero />
      <section className="site-index" aria-labelledby="site-index-title">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Study navigation</p>
            <h2 id="site-index-title">A multi-page research record</h2>
          </div>
          <p>Each page focuses on one part of the project so the application reads like a structured research site rather than a single long report.</p>
        </div>
        <div className="route-grid">
          {routes.map(({ href, label, detail, icon: Icon }) => (
            <Link href={href} key={href}>
              <Icon size={21} aria-hidden="true" />
              <strong>{label}</strong>
              <span>{detail}</span>
              <ArrowRight size={16} aria-hidden="true" />
            </Link>
          ))}
        </div>
      </section>
    </>
  );
}
