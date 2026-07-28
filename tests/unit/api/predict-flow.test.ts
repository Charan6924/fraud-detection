import type { PredictionResult } from "core";

const mockDdbSend = jest.fn();
const mockCheckRateLimit = jest.fn();
const mockGetCachedPrediction = jest.fn();
const mockSetCacheEntry = jest.fn();
const mockFetch = jest.fn();

jest.mock("@aws-sdk/client-dynamodb", () => ({
  DynamoDBClient: jest.fn(() => ({})),
}));
jest.mock("@aws-sdk/lib-dynamodb", () => ({
  DynamoDBDocumentClient: { from: jest.fn(() => ({ send: mockDdbSend })) },
  PutCommand: jest.fn(),
}));
jest.mock("core", () => ({
  checkRateLimit: mockCheckRateLimit,
  getCachedPrediction: mockGetCachedPrediction,
  setCacheEntry: mockSetCacheEntry,
}));

const mockNextResponseJson = jest.fn();
jest.mock("next/server", () => ({
  NextRequest: jest.fn(),
  NextResponse: { json: (...args: unknown[]) => mockNextResponseJson(...args) },
}));

globalThis.fetch = mockFetch;

import { POST } from "../../../packages/api/src/app/api/predict/route";

const validPayload = { TransactionAmt: 200.0, card1: 5001 };
const result: PredictionResult = {
  prediction: 1,
  probability: 0.92,
  probabilities_per_model: { xgboost: 0.9, random_forest: 0.85, logistic_regression: 0.78 },
  model_version: "ensemble_v1",
};

beforeEach(() => {
  jest.clearAllMocks();
  process.env.MODEL_SERVICE_URL = "http://model:80";
  process.env.MODEL_SECRET = "secret";
  process.env.PREDICTIONS_TABLE = "predictions";

  mockNextResponseJson.mockImplementation(
    (data: unknown, init?: { status: number }) => ({
      status: init?.status ?? 200,
      data,
    }),
  );
});

function mockReq(body: unknown) {
  return { json: jest.fn().mockResolvedValue(body) } as unknown as Request;
}

describe("predict full flow", () => {
  it("full happy path: rate limit OK, cache miss, model returns, DB writes", async () => {
    mockCheckRateLimit.mockResolvedValue(true);
    mockGetCachedPrediction.mockResolvedValue(null);
    mockFetch.mockResolvedValue({ ok: true, json: async () => result } as Response);
    mockDdbSend.mockResolvedValue({});

    const res = await POST(mockReq(validPayload));

    expect(res.status).toBe(200);
    expect(res.data).toMatchObject({
      prediction: 1,
      model_version: "ensemble_v1",
    });
    expect(mockCheckRateLimit).toHaveBeenCalledWith("5001");
    expect(mockGetCachedPrediction).toHaveBeenCalledWith("5001", 200.0);
    expect(mockFetch).toHaveBeenCalledWith(
      "http://model:80/predict",
      expect.objectContaining({
        method: "POST",
        headers: expect.objectContaining({
          "content-type": "application/json",
          "x-model-secret": "secret",
        }),
        body: JSON.stringify(validPayload),
      }),
    );
    expect(mockSetCacheEntry).toHaveBeenCalledWith("5001", 200.0, result);
    expect(mockDdbSend).toHaveBeenCalledTimes(1);
  });

  it("cache hit path skips model call and DB write", async () => {
    mockCheckRateLimit.mockResolvedValue(true);
    mockGetCachedPrediction.mockResolvedValue(result);

    const res = await POST(mockReq(validPayload));

    expect(res.status).toBe(200);
    expect(res.data).toEqual(result);
    expect(mockFetch).not.toHaveBeenCalled();
    expect(mockDdbSend).not.toHaveBeenCalled();
  });

  it("rate limit check runs then cache miss then model call", async () => {
    mockCheckRateLimit.mockResolvedValue(true);
    mockGetCachedPrediction.mockResolvedValue(null);
    mockFetch.mockResolvedValue({ ok: true, json: async () => result } as Response);
    mockDdbSend.mockResolvedValue({});

    const res = await POST(mockReq({ TransactionAmt: 50.0, card1: 9999 }));

    expect(res.status).toBe(200);
    expect(mockCheckRateLimit).toHaveBeenCalled();
    expect(mockGetCachedPrediction).toHaveBeenCalled();
    expect(mockFetch).toHaveBeenCalled();
    expect(mockDdbSend).toHaveBeenCalled();
  });

  it("model error does not cache or write to DB", async () => {
    mockCheckRateLimit.mockResolvedValue(true);
    mockGetCachedPrediction.mockResolvedValue(null);
    mockFetch.mockResolvedValue({
      ok: false,
      status: 500,
      text: async () => "Model exploded",
    } as Response);

    const res = await POST(mockReq(validPayload));

    expect(res.status).toBe(502);
    expect(res.data).toEqual({
      error: "Model invocation failed",
      detail: "Model exploded",
    });
    expect(mockSetCacheEntry).not.toHaveBeenCalled();
    expect(mockDdbSend).not.toHaveBeenCalled();
  });
});
