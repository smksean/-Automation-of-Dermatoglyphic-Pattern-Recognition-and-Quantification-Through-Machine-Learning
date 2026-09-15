import { NextResponse } from "next/server";

const MAX_UPLOAD_BYTES = 4 * 1024 * 1024;
const ALLOWED_TYPES = new Set(["image/png", "image/jpeg", "image/tiff"]);

export const runtime = "nodejs";
export const maxDuration = 60;

export async function POST(request: Request) {
  const inferenceUrl = process.env.INFERENCE_API_URL?.replace(/\/$/, "");
  if (!inferenceUrl) {
    return NextResponse.json(
      {
        error: "Inference service is not configured.",
        detail:
          "The research interface is ready, but INFERENCE_API_URL has not been set for this deployment.",
      },
      { status: 503 },
    );
  }

  const formData = await request.formData();
  const image = formData.get("image");
  if (!(image instanceof File)) {
    return NextResponse.json({ error: "Select one fingerprint image." }, { status: 400 });
  }
  if (!ALLOWED_TYPES.has(image.type)) {
    return NextResponse.json(
      { error: "Unsupported image format. Use PNG, JPEG, or TIFF." },
      { status: 415 },
    );
  }
  if (image.size > MAX_UPLOAD_BYTES) {
    return NextResponse.json(
      { error: "The deployment upload limit is 4 MB per image." },
      { status: 413 },
    );
  }

  const upstreamBody = new FormData();
  upstreamBody.append("image", image, "fingerprint-upload");

  try {
    const headers = new Headers();
    const inferenceToken = process.env.INFERENCE_API_TOKEN;
    if (inferenceToken) headers.set("Authorization", `Bearer ${inferenceToken}`);
    const response = await fetch(`${inferenceUrl}/predict`, {
      method: "POST",
      body: upstreamBody,
      headers,
      cache: "no-store",
    });
    const payload = (await response.json()) as {
      error?: string;
      detail?: string | Array<{ msg?: string }>;
    };
    if (!response.ok) {
      const upstreamDetail = Array.isArray(payload.detail)
        ? payload.detail.map((item) => item.msg).filter(Boolean).join(" ")
        : payload.detail;
      const error =
        response.status === 401
          ? "Inference service authorization failed."
          : payload.error ?? "The inference service rejected this image.";
      return NextResponse.json(
        { error, detail: upstreamDetail },
        { status: response.status },
      );
    }
    return NextResponse.json(payload, { status: response.status });
  } catch {
    return NextResponse.json(
      {
        error: "The inference service could not be reached.",
        detail: "No image was retained by this web application.",
      },
      { status: 502 },
    );
  }
}
