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
    const vpc = new sst.aws.Vpc("ModelVpc");
    const cluster = new sst.aws.Cluster("ModelCluster", { vpc });

    const modelService = new sst.aws.Service("ModelService", {
      cluster,
      image: { dockerfile: "container/Dockerfile" },
      memory: "8 GB",
      cpu: "4 vCPU",
      loadBalancer: {
        rules: [{ listen: "80/http" }],
      },
    });

    // Next.js API
    const api = new sst.aws.Nextjs("Api", {
    path: "packages/api",
    link: [predictionsTable, modelService],
    environment: {
      MODEL_SERVICE_URL: modelService.url,
      PREDICTIONS_TABLE: predictionsTable.name,
      },
    });

    return {
      api: api.url,
      model: modelService.url,
    };
  }
});
    