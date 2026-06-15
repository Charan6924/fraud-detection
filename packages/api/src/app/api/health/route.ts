import { NextResponse } from "next/server";

export async function GET() {
  const modelRes = await fetch(`${process.env.MODEL_SERVICE_URL}/health`);
  const container = await modelRes.json();

  return NextResponse.json({
    status: "ok",
    container,
    timestamp: new Date().toISOString(),
  });
}
