/// <reference path="./.sst/platform/config.d.ts" />

export default $config({
  app(input) {
    return {
      name: "fraud-detection",
      removal: input?.stage === "production" ? "retain" : "remove",
      home: "aws",
    };
  },
  async run() {
    // DynamoDB table for prediction records
    const predictionsTable = new sst.aws.Dynamo("Predictions", {
      fields: { id: "string" },
      primaryIndex: { hashKey: "id" },
    });

    // Container Lambda for model inference
    const modelFn = new sst.aws.Function("ModelContainer", {
      container: { file: "container/Dockerfile" },
      memory: "2048 MB",
      timeout: "30 seconds",
      url: true,
    });

    // Next.js API
    const api = new sst.aws.Nextjs("Api", {
      path: "packages/api",
      link: [predictionsTable, modelFn],
      environment: {
        CONTAINER_LAMBDA_ARN: modelFn.arn,
        PREDICTIONS_TABLE: predictionsTable.name,
      },
    });

    return {
      api: api.url,
      model: modelFn.url,
    };
  },
});
