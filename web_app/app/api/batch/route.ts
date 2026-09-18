import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const maxDuration = 120;

export async function POST() {
  const inferenceUrl = process.env.INFERENCE_API_URL?.replace(/\/$/, "");
  if (!inferenceUrl) {
    return NextResponse.json(
      { error: "Inference service is not configured." },
      { status: 503 },
    );
  }
  try {
    const headers = new Headers();
    const token = process.env.INFERENCE_API_TOKEN;
    if (token) headers.set("Authorization", `Bearer ${token}`);
    const response = await fetch(`${inferenceUrl}/batch-jobs`, {
      method: "POST",
      headers,
      cache: "no-store",
    });
    const payload = (await response.json()) as { error?: string; detail?: string };
    if (!response.ok) {
      return NextResponse.json(
        {
          error: response.status === 401 ? "Inference service authorization failed." : payload.error ?? "The ten-finger batch could not be created.",
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
    return NextResponse.json(
      { error: "The inference service could not be reached." },
      { status: 502 },
    );
  }
}
