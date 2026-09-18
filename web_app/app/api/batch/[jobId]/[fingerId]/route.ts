import { NextResponse } from "next/server";

const MAX_UPLOAD_BYTES = 4 * 1024 * 1024;
const ALLOWED_TYPES = new Set(["image/png", "image/jpeg", "image/tiff"]);
const FINGER_IDS = new Set([
  "right-thumb", "right-index", "right-middle", "right-ring", "right-little",
  "left-thumb", "left-index", "left-middle", "left-ring", "left-little",
]);

export const runtime = "nodejs";
export const maxDuration = 120;

export async function POST(
  request: Request,
  context: { params: Promise<{ jobId: string; fingerId: string }> },
) {
  const inferenceUrl = process.env.INFERENCE_API_URL?.replace(/\/$/, "");
  if (!inferenceUrl) {
    return NextResponse.json({ error: "Inference service is not configured." }, { status: 503 });
  }
  const { jobId, fingerId } = await context.params;
  if (!/^[A-Za-z0-9_-]{20,64}$/.test(jobId) || !FINGER_IDS.has(fingerId)) {
    return NextResponse.json({ error: "Invalid ten-finger batch request." }, { status: 400 });
  }
  const formData = await request.formData();
  const image = formData.get("image");
  if (!(image instanceof File)) {
    return NextResponse.json({ error: "Select one fingerprint image." }, { status: 400 });
  }
  if (!ALLOWED_TYPES.has(image.type)) {
    return NextResponse.json({ error: "Unsupported image format. Use PNG, JPEG, or TIFF." }, { status: 415 });
  }
  if (image.size > MAX_UPLOAD_BYTES) {
    return NextResponse.json({ error: "The deployment upload limit is 4 MB per image." }, { status: 413 });
  }

  const upstreamBody = new FormData();
  upstreamBody.append("image", image, `fingerprint-${fingerId}`);
  try {
    const headers = new Headers();
    const token = process.env.INFERENCE_API_TOKEN;
    if (token) headers.set("Authorization", `Bearer ${token}`);
    const response = await fetch(`${inferenceUrl}/batch-jobs/${jobId}/images/${fingerId}`, {
      method: "POST",
      body: upstreamBody,
      headers,
      cache: "no-store",
    });
    const payload = (await response.json()) as { error?: string; detail?: string };
    if (!response.ok) {
      return NextResponse.json(
        {
          error: response.status === 401 ? "Inference service authorization failed." : payload.error ?? "The inference service rejected this fingerprint.",
          detail: payload.detail,
        },
        { status: response.status },
      );
    }
    return NextResponse.json(payload, {
      status: response.status,
      headers: { "Cache-Control": "no-store" },
    });
  } catch {
    return NextResponse.json({ error: "The fingerprint could not be uploaded to the batch." }, { status: 502 });
  }
}
