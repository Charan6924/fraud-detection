import { LambdaClient, InvokeCommand } from "@aws-sdk/client-lambda";                                                                   
import { DynamoDBClient } from "@aws-sdk/client-dynamodb";                                                                              
import { DynamoDBDocumentClient, PutCommand } from "@aws-sdk/lib-dynamodb";         
import { NextRequest, NextResponse } from "next/server";
import type { TransactionInput, PredictionResult } from "core";   

const lambda = new LambdaClient({region:process.env.AWS_REGION})
const ddb = DynamoDBDocumentClient.from(new DynamoDBClient({}));    

export async function POST(req : NextRequest){                                                                                  
    const input : TransactionInput = await req.json()

    const cmd = new InvokeCommand({                                                                                                       
      FunctionName: process.env.CONTAINER_LAMBDA_ARN!,                                                                                    
      Payload: JSON.stringify({                                                                                                           
        version: "2.0",                                                                                                                   
        rawPath: "/predict",                                                                                                              
        headers: { "content-type": "application/json" },                                                                                  
        requestContext: { http: { method: "POST", path: "/predict" } },                                                                   
        body: JSON.stringify(input),                                                                                                      
        isBase64Encoded: false,                                                                                                           
      }),                                                                                                                                 
    });  

    const { Payload, FunctionError } = await lambda.send(cmd);                                                                            
                                                                                                                                          
    if (FunctionError) {                                                                                                                  
      const err = Payload ? JSON.parse(new TextDecoder().decode(Payload)) : {};                                                           
      return NextResponse.json({ error: "Model invocation failed", detail: err }, { status: 502 });                                       
    }    

    const response = JSON.parse(new TextDecoder().decode(Payload!));
    const result : PredictionResult = typeof response.body === "string" ? JSON.parse(response.body) : response;

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
