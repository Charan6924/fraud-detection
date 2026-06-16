// query the database for previous transactions
import { DynamoDBClient } from "@aws-sdk/client-dynamodb";
import { ScanCommand } from "@aws-sdk/lib-dynamodb";
import { DynamoDBDocumentClient, GetCommand } from "@aws-sdk/lib-dynamodb";
import { NextRequest, NextResponse } from "next/server";

const ddbClient = process.env.DYNAMODB_ENDPOINT
  ? new DynamoDBClient({ endpoint: process.env.DYNAMODB_ENDPOINT, region: "local", credentials: { accessKeyId: "fake", secretAccessKey: "fake" } })
  : new DynamoDBClient({});
const ddb = DynamoDBDocumentClient.from(ddbClient);

export async function GET(req : NextRequest){
    const id = req.nextUrl.searchParams.get("id");
    if (id){
        const result = await ddb.send(new GetCommand({
            TableName : process.env.PREDICTIONS_TABLE!,
            Key : {id},
        }));

        if (!result.Item){
            return NextResponse.json({error : "Transaction not found"}, {status : 404})
        }
        return NextResponse.json(result.Item);  
    }
    else{
        const result = await ddb.send(new ScanCommand({
        TableName: process.env.PREDICTIONS_TABLE!,
        Limit: 50,
        }));
        const items = (result.Items || []).sort(
            (a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
        );

        return NextResponse.json(items);
    }
}
