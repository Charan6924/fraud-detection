import { DynamoDBClient } from "@aws-sdk/client-dynamodb";
import { DynamoDBDocumentClient, PutCommand } from "@aws-sdk/lib-dynamodb";
import { NextRequest, NextResponse } from "next/server";
import type { TransactionInput, PredictionResult } from "core";

const ddbClient = process.env.DYNAMODB_ENDPOINT
  ? new DynamoDBClient({ endpoint: process.env.DYNAMODB_ENDPOINT, region: "local", credentials: { accessKeyId: "fake", secretAccessKey: "fake" } })
  : new DynamoDBClient({});
const ddb = DynamoDBDocumentClient.from(ddbClient);

export async function POST(req: NextRequest) {
  const input: TransactionInput = await req.json();

  const modelRes = await fetch(`${process.env.MODEL_SERVICE_URL}/predict`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(input),
  });

  if (!modelRes.ok) {
    const detail = await modelRes.text();
    return NextResponse.json({ error: "Model invocation failed", detail }, { status: 502 });
  }

  const result: PredictionResult = await modelRes.json();

  await ddb.send(new PutCommand({
    TableName: process.env.PREDICTIONS_TABLE!,
    Item: {
      id: crypto.randomUUID(),
      input,
      result,
      timestamp: new Date().toISOString(),
    },
  }));

  return NextResponse.json(result);
}
