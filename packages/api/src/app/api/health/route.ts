import { LambdaClient, InvokeCommand } from "@aws-sdk/client-lambda";     
import { NextResponse } from "next/server";

const lambda = new LambdaClient({region:process.env.AWS_REGION})

export async function GET(){
    const cmd = new InvokeCommand({
        FunctionName: process.env.CONTAINER_LAMBDA_ARN!,
        Payload: JSON.stringify({
            version: "2.0",
            rawPath: "/health",
            headers: {},
            requestContext: { http: { method: "GET", path: "/health" } },
            body: null,
            isBase64Encoded: false,
        }),
    });

    const { Payload } = await lambda.send(cmd);
    const response = JSON.parse(new TextDecoder().decode(Payload!));
    const container = typeof response.body === "string" ? JSON.parse(response.body) : response;

    return NextResponse.json({
        status: "ok",
        container,
        timestamp: new Date().toISOString(),
    });
}
