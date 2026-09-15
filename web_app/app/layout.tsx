import type { Metadata } from "next";
import { Inter } from "next/font/google";
import { SiteHeader } from "@/components/site-header";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-sans",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Dermatoglyphic Pattern Research",
  description:
    "A research application for broad fingerprint-pattern and exploratory subtype classification.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className={inter.variable} id="top">
        <SiteHeader />
        <main>{children}</main>
        <footer>
          <strong>Dermatoglyphic ML Study</strong>
          <span>Research prototype | Broad patterns, subtypes, and quantification</span>
          <a href="#top">Back to top</a>
        </footer>
      </body>
    </html>
  );
}
