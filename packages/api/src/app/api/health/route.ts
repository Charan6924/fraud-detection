import { NextResponse } from "next/server";

export async function GET() {
  let container = { status: "unreachable" };
  try {
    const modelRes = await fetch(`${process.env.MODEL_SERVICE_URL}/health`);
    container = await modelRes.json();
  } catch {
    // model service unreachable — return partial status
  }

  return NextResponse.json({
    status: "ok",
    container,
    timestamp: new Date().toISOString(),
  });
}
