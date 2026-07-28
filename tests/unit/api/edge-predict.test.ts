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
import type { PredictionResult } from "core";

const baseResult: PredictionResult = {
  prediction: 0,
  probability: 0.02,
  probabilities_per_model: { xgboost: 0.01, random_forest: 0.03, logistic_regression: 0.02 },
  model_version: "ensemble_v1",
};

beforeEach(() => {
  jest.clearAllMocks();
  process.env.MODEL_SERVICE_URL = "http://model:80";
  process.env.MODEL_SECRET = "s3cr3t";
  process.env.PREDICTIONS_TABLE = "predictions";
  mockCheckRateLimit.mockResolvedValue(true);
  mockGetCachedPrediction.mockResolvedValue(null);
  mockFetch.mockResolvedValue({ ok: true, json: async () => baseResult } as Response);
  mockDdbSend.mockResolvedValue({});
  mockNextResponseJson.mockImplementation((data: unknown, init?: { status: number }) => ({
    status: init?.status ?? 200,
    data,
  }));
});

function mockReq(body: unknown) {
  return { json: jest.fn().mockResolvedValue(body) } as unknown as Request;
}

describe("predict edge cases", () => {
  it("handles transaction with zero TransactionAmt", async () => {
    const res = await POST(mockReq({ TransactionAmt: 0, card1: 42 }));
    expect(res.status).toBe(200);
  });

  it("handles transaction with negative TransactionAmt", async () => {
    const res = await POST(mockReq({ TransactionAmt: -50.0, card1: 42 }));
    expect(res.status).toBe(200);
  });

  it("handles transaction with very large TransactionAmt", async () => {
    const res = await POST(mockReq({ TransactionAmt: 9.99e9, card1: 42 }));
    expect(res.status).toBe(200);
  });

  it("handles transaction with TransactionAmt as 0.01 (min valid)", async () => {
    const res = await POST(mockReq({ TransactionAmt: 0.01, card1: 42 }));
    expect(res.status).toBe(200);
  });

  it("handles transaction with all optional fields populated", async () => {
    const fullPayload: Record<string, unknown> = {
      TransactionAmt: 150.0,
      card1: 1, card2: 2, card3: 3, card4: 4, card5: 5, card6: 6,
      addr1: 101, addr2: 102, dist1: 10.5, dist2: 20.3,
      P_emaildomain: 5, R_emaildomain: 3,
      DeviceType: 1, DeviceInfo: 2,
      ProductCD: 3,
    };
    for (let i = 1; i <= 14; i++) fullPayload[`C${i}`] = 0.5;
    for (const d of [1, 2, 3, 4, 5, 10, 15]) fullPayload[`D${d}`] = 1.0;
    for (const m of [2, 3, 4, 5, 6, 7, 9]) fullPayload[`M${m}`] = 0.0;
    for (const id of [1, 2, 5, 6, 12, 13, 14, 15, 16, 17, 19, 20, 28, 31, 36, 37, 38]) {
      fullPayload[`id_${id.toString().padStart(2, "0")}`] = 1.0;
    }

    const res = await POST(mockReq(fullPayload));
    expect(res.status).toBe(200);
  });

  it("handles missing optional fields gracefully", async () => {
    const res = await POST(mockReq({ TransactionAmt: 75.0 }));
    expect(res.status).toBe(200);
    expect(mockCheckRateLimit).toHaveBeenCalledWith("undefined");
  });

  it("concurrent requests to same card do not interfere", async () => {
    mockCheckRateLimit
      .mockResolvedValueOnce(true)
      .mockResolvedValueOnce(true);

    const [res1, res2] = await Promise.all([
      POST(mockReq({ TransactionAmt: 10.0, card1: 999 })),
      POST(mockReq({ TransactionAmt: 20.0, card1: 999 })),
    ]);

    expect(res1.status).toBe(200);
    expect(res2.status).toBe(200);
    expect(mockCheckRateLimit).toHaveBeenCalledTimes(2);
    expect(mockCheckRateLimit).toHaveBeenCalledWith("999");
  });
});
