"use client";

import Image from "next/image";
import { Maximize2, X } from "lucide-react";
import { useEffect, useRef, useState, type ReactNode } from "react";

type FigureZoomProps = {
  src: string;
  alt: string;
  width: number;
  height: number;
  className?: string;
  caption?: ReactNode;
};

export function FigureZoom({ src, alt, width, height, className, caption }: FigureZoomProps) {
  const [open, setOpen] = useState(false);
  const closeButton = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!open) return;
    closeButton.current?.focus();
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    document.addEventListener("keydown", closeOnEscape);
    return () => document.removeEventListener("keydown", closeOnEscape);
  }, [open]);

  return (
    <>
      <figure className={className}>
        <button className="figure-open" type="button" onClick={() => setOpen(true)} aria-label={`Enlarge: ${alt}`}>
          <Image src={src} alt={alt} width={width} height={height} />
          <span><Maximize2 size={15} aria-hidden="true" /> Inspect figure</span>
        </button>
        {caption ? <figcaption>{caption}</figcaption> : null}
      </figure>

      {open ? (
        <div className="figure-dialog" role="dialog" aria-modal="true" aria-label={alt} onMouseDown={(event) => {
          if (event.target === event.currentTarget) setOpen(false);
        }}>
          <div className="figure-dialog-panel">
            <button ref={closeButton} className="figure-close" type="button" onClick={() => setOpen(false)} aria-label="Close enlarged figure">
              <X size={20} aria-hidden="true" />
            </button>
            <Image src={src} alt={alt} width={width} height={height} priority />
            {caption ? <div className="figure-dialog-caption">{caption}</div> : null}
          </div>
        </div>
      ) : null}
    </>
  );
}
