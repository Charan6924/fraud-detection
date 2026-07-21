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

    // S3 bucket for training artifacts (features, models, reference)
    const artifactsBucket = new sst.aws.Bucket("Artifacts");

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
      link: [predictionsTable],
      environment: {
        MODEL_SECRET: process.env.MODEL_SECRET!,
      },
    });

    // Next.js API
    const api = new sst.aws.Nextjs("Api", {
    path: "packages/api",
    link: [predictionsTable, modelService],
    environment: {
      MODEL_SERVICE_URL: modelService.url,
      PREDICTIONS_TABLE: predictionsTable.name,
      API_KEY: process.env.API_KEY!,
      MODEL_SECRET: process.env.MODEL_SECRET!,
      UPSTASH_REDIS_REST_URL : process.env.UPSTASH_REDIS_REST_URL!,
      UPSTASH_REDIS_REST_TOKEN: process.env.UPSTASH_REDIS_REST_TOKEN!
    },
    });

    const monitorFunction = new sst.aws.Function("MonitorFunction", {
      handler : "packages/monitor/src/index.handler",
      link : [predictionsTable],
      environment : {
        MODEL_SERVICE_URL : modelService.url,
        GH_TOKEN : process.env.GH_TOKEN!,
        MODEL_SECRET : process.env.MODEL_SECRET!,
      }
    });

    new sst.aws.Cron("MonitorCron", {
      schedule : "rate(6 hours)",
      job : monitorFunction.arn,
    });

    return {
      api: api.url,
      model: modelService.url,
      artifactsBucket: artifactsBucket.name,
    };
  }
});
    