import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const maxDuration = 30;

export async function GET(
  _request: Request,
  context: { params: Promise<{ jobId: string }> },
) {
  const inferenceUrl = process.env.INFERENCE_API_URL?.replace(/\/$/, "");
  if (!inferenceUrl) {
    return NextResponse.json(
      { error: "Inference service is not configured." },
      { status: 503 },
    );
  }

  const { jobId } = await context.params;
  if (!/^[A-Za-z0-9_-]{20,64}$/.test(jobId)) {
    return NextResponse.json({ error: "Invalid analysis job." }, { status: 400 });
  }

  try {
    const headers = new Headers();
    const inferenceToken = process.env.INFERENCE_API_TOKEN;
    if (inferenceToken) headers.set("Authorization", `Bearer ${inferenceToken}`);
    const response = await fetch(`${inferenceUrl}/jobs/${jobId}`, {
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
      return NextResponse.json(
        {
          error: payload.error ?? "The analysis job could not be retrieved.",
          detail: upstreamDetail,
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
      {
        error: "The inference service could not be reached.",
        detail: "The analysis can be submitted again when the service is available.",
      },
      { status: 502 },
    );
  }
}
