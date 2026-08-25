jest.mock("@aws-sdk/client-dynamodb", () => ({
  DynamoDBClient: jest.fn(() => ({})),
}));

jest.mock("@aws-sdk/lib-dynamodb", () => ({
  DynamoDBDocumentClient: {
    from: jest.fn(() => ({
      send: jest.fn(),
    })),
  },
  PutCommand: jest.fn(),
}));

const mockCheckRateLimit = jest.fn();
const mockGetCachedPrediction = jest.fn();
const mockSetCacheEntry = jest.fn();

jest.mock("core", () => ({
  checkRateLimit: mockCheckRateLimit,
  getCachedPrediction: mockGetCachedPrediction,
  setCacheEntry: mockSetCacheEntry,
}));

const mockNextResponseJson = jest.fn();
jest.mock("next/server", () => ({
  NextRequest: jest.fn(),
  NextResponse: {
    json: (...args: unknown[]) => mockNextResponseJson(...args),
  },
}));

import { POST } from "../../../packages/api/src/app/api/predict/route";
import type { PredictionResult } from "core";

const validPayload = {
  TransactionAmt: 150.0,
  card1: 10001,
};

const validResult: PredictionResult = {
  prediction: 1,
  probability: 0.87,
  probabilities_per_model: {
    xgboost: 0.85,
    random_forest: 0.72,
    logistic_regression: 0.65,
  },
  model_version: "ensemble_v1",
};

beforeEach(() => {
  jest.clearAllMocks();
  process.env.MODEL_SERVICE_URL = "http://model:80";
  process.env.MODEL_SECRET = "model-secret-42";
  process.env.PREDICTIONS_TABLE = "predictions-table";

  mockNextResponseJson.mockImplementation((data: unknown, init?: { status: number }) => ({
    status: init?.status ?? 200,
    data,
  }));
});

function mockRequest(body: unknown) {
  return {
    json: jest.fn().mockResolvedValue(body),
  } as unknown as Request;
}

describe("predict endpoint", () => {
  it("returns successful prediction for valid transaction", async () => {
    mockCheckRateLimit.mockResolvedValue(true);
    mockGetCachedPrediction.mockResolvedValue(null);
    mockSetCacheEntry.mockResolvedValue(undefined);

    const mockFetch = jest.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      json: async () => validResult,
    } as Response);

    const req = mockRequest(validPayload);
    const res = await POST(req);

    expect(res.status).toBe(200);
    expect(mockCheckRateLimit).toHaveBeenCalledWith("10001");
    expect(mockGetCachedPrediction).toHaveBeenCalledWith("10001", 150.0);
    expect(globalThis.fetch).toHaveBeenCalledWith(
      "http://model:80/predict",
      expect.objectContaining({ method: "POST" }),
    );
    expect(mockSetCacheEntry).toHaveBeenCalledWith("10001", 150.0, validResult);
    expect(res.data).toEqual(validResult);

    mockFetch.mockRestore();
  });

  it("rate limit check is called with card ID", async () => {
    mockCheckRateLimit.mockResolvedValue(false);
    mockGetCachedPrediction.mockResolvedValue(null);

    const mockFetch = jest.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      json: async () => validResult,
    } as Response);

    const req = mockRequest(validPayload);
    const res = await POST(req);

    expect(mockCheckRateLimit).toHaveBeenCalledWith("10001");
    expect(res.status).toBe(429);
    expect(res.data).toEqual({ error: "Rate limit exceeded" });
    expect(mockGetCachedPrediction).not.toHaveBeenCalled();
    expect(globalThis.fetch).not.toHaveBeenCalled();

    mockFetch.mockRestore();
  });

  it("returns cached prediction without calling model", async () => {
    mockCheckRateLimit.mockResolvedValue(true);
    mockGetCachedPrediction.mockResolvedValue(validResult);

    const mockFetch = jest.spyOn(globalThis, "fetch");
    const req = mockRequest(validPayload);
    const res = await POST(req);

    expect(res.status).toBe(200);
    expect(res.data).toEqual(validResult);
    expect(mockFetch).not.toHaveBeenCalled();
    mockFetch.mockRestore();
  });

  it("returns 502 when model service returns error", async () => {
    mockCheckRateLimit.mockResolvedValue(true);
    mockGetCachedPrediction.mockResolvedValue(null);

    const mockFetch = jest.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: false,
      status: 500,
      text: async () => "Internal model error",
    } as Response);

    const req = mockRequest(validPayload);
    const res = await POST(req);

    expect(res.status).toBe(502);
    expect(res.data).toEqual({
      error: "Model invocation failed",
      detail: "Internal model error",
    });

    mockFetch.mockRestore();
  });

  it("handles DynamoDB write failure gracefully", async () => {
    mockCheckRateLimit.mockResolvedValue(true);
    mockGetCachedPrediction.mockResolvedValue(null);
    mockSetCacheEntry.mockResolvedValue(undefined);

    const mockFetch = jest.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      json: async () => validResult,
    } as Response);

    const { DynamoDBDocumentClient } = jest.requireMock("@aws-sdk/lib-dynamodb");
    const ddb = DynamoDBDocumentClient.from();
    ddb.send.mockRejectedValue(new Error("DynamoDB write failed"));

    const req = mockRequest(validPayload);
    const res = await POST(req);

    expect(res.status).toBe(200);
    expect(res.data).toEqual(validResult);

    mockFetch.mockRestore();
  });

  it("handles missing card1 field", async () => {
    mockCheckRateLimit.mockResolvedValue(true);
    mockGetCachedPrediction.mockResolvedValue(null);

    const mockFetch = jest.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      json: async () => validResult,
    } as Response);

    const req = mockRequest({ TransactionAmt: 50.0 });
    const res = await POST(req);

    expect(res.status).toBe(200);
    expect(mockCheckRateLimit).toHaveBeenCalledWith("undefined");
    expect(mockGetCachedPrediction).toHaveBeenCalledWith("undefined", 50.0);

    mockFetch.mockRestore();
  });

  it("throws on invalid JSON body", async () => {
    const req = {
      json: jest.fn().mockRejectedValue(new Error("Invalid JSON")),
    } as unknown as Request;

    await expect(POST(req)).rejects.toThrow("Invalid JSON");
  });
});
