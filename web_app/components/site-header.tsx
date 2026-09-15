"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const links = [
  { href: "/analysis", label: "Analysis" },
  { href: "/methods", label: "Methods" },
  { href: "/results", label: "Results" },
  { href: "/quantification", label: "Quantification" },
  { href: "/models", label: "Models" },
  { href: "/study", label: "Study notes" },
];

export function SiteHeader() {
  const pathname = usePathname();

  return (
    <header className="site-header">
      <Link className="brand" href="/" aria-label="Dermatoglyphic ML study home">
        <span className="brand-mark" aria-hidden="true">DP</span>
        <span>
          <strong>Dermatoglyphic ML Study</strong>
          <small>Pattern recognition research</small>
        </span>
      </Link>
      <nav className="primary-nav" aria-label="Primary navigation">
        {links.map(({ href, label }) => (
          <Link href={href} key={href} aria-current={pathname === href ? "page" : undefined}>
            {label}
          </Link>
        ))}
      </nav>
      <span className="prototype-status">Protocol v0.2</span>
    </header>
  );
}
