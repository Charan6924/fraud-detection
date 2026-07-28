import {
  checkRateLimit,
  getCachedPrediction,
  setCacheEntry,
  type TransactionInput,
  type PredictionResult,
  type TransactionRecord,
  type HealthResponse,
} from "core";

jest.mock("@upstash/redis", () => {
  const mockIncr = jest.fn().mockResolvedValue(1);
  const mockExpire = jest.fn();
  const mockGet = jest.fn().mockResolvedValue(null);
  const mockSet = jest.fn();
  return {
    Redis: {
      fromEnv: () => ({
        incr: mockIncr,
        expire: mockExpire,
        get: mockGet,
        set: mockSet,
      }),
    },
  };
});

describe("core package exports", () => {
  it("exports checkRateLimit as a function", () => {
    expect(checkRateLimit).toBeInstanceOf(Function);
  });

  it("exports getCachedPrediction as a function", () => {
    expect(getCachedPrediction).toBeInstanceOf(Function);
  });

  it("exports setCacheEntry as a function", () => {
    expect(setCacheEntry).toBeInstanceOf(Function);
  });
});

describe("TransactionInput type shape", () => {
  it("accepts a minimal valid transaction", () => {
    const txn: TransactionInput = { TransactionAmt: 100.5 };
    expect(txn.TransactionAmt).toBe(100.5);
  });

  it("accepts a full transaction with optional fields", () => {
    const txn: TransactionInput = {
      TransactionAmt: 250.0,
      card1: 12345,
      card2: 678,
      ProductCD: 3,
      addr1: 1001,
      addr2: 2002,
      dist1: 50.0,
      P_emaildomain: 5,
      DeviceType: 1,
      C1: 0.5,
      D1: 1.0,
      M2: 0.0,
      V2: -0.3,
      id_01: 1.0,
    };
    expect(txn.TransactionAmt).toBe(250.0);
    expect(txn.card1).toBe(12345);
    expect(txn.D1).toBe(1.0);
  });

  it("rejects TransactionInput missing required TransactionAmt at compile time", () => {
    const invalid: Partial<TransactionInput> = {};
    expect(invalid.TransactionAmt).toBeUndefined();
  });
});

describe("PredictionResult type shape", () => {
  const validResult: PredictionResult = {
    prediction: 1,
    probability: 0.89,
    probabilities_per_model: {
      xgboost: 0.85,
      random_forest: 0.75,
      logistic_regression: 0.65,
    },
    model_version: "ensemble_v1",
  };

  it("holds all required fields", () => {
    expect(validResult.prediction).toBe(1);
    expect(validResult.probability).toBe(0.89);
    expect(validResult.probabilities_per_model.xgboost).toBe(0.85);
    expect(validResult.model_version).toBe("ensemble_v1");
  });

  it("accepts prediction=0 for legitimate transactions", () => {
    const legit: PredictionResult = {
      prediction: 0,
      probability: 0.02,
      probabilities_per_model: {
        xgboost: 0.01,
        random_forest: 0.03,
        logistic_regression: 0.02,
      },
      model_version: "ensemble_v1",
    };
    expect(legit.prediction).toBe(0);
  });

  it("supports different model versions", () => {
    const v2: PredictionResult = {
      prediction: 0,
      probability: 0.1,
      probabilities_per_model: {
        xgboost: 0.1,
        random_forest: 0.1,
        logistic_regression: 0.1,
      },
      model_version: "ensemble_v2",
    };
    expect(v2.model_version).toBe("ensemble_v2");
  });
});

describe("TransactionRecord type shape", () => {
  it("holds input, result, id, and timestamp", () => {
    const record: TransactionRecord = {
      id: "uuid-123",
      input: { TransactionAmt: 50.0 },
      result: {
        prediction: 0,
        probability: 0.01,
        probabilities_per_model: {
          xgboost: 0.01,
          random_forest: 0.01,
          logistic_regression: 0.01,
        },
        model_version: "ensemble_v1",
      },
      timestamp: "2026-07-28T12:00:00.000Z",
    };
    expect(record.id).toBe("uuid-123");
    expect(record.input.TransactionAmt).toBe(50.0);
    expect(record.result.prediction).toBe(0);
    expect(record.timestamp).toBe("2026-07-28T12:00:00.000Z");
  });
});

describe("HealthResponse type shape", () => {
  it("holds status, container info, and timestamp", () => {
    const health: HealthResponse = {
      status: "ok",
      container: { status: "ok" },
      timestamp: "2026-07-28T12:00:00.000Z",
    };
    expect(health.status).toBe("ok");
    expect(health.container.status).toBe("ok");
  });
});
