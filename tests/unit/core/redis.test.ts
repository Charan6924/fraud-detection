import { checkRateLimit, getCachedPrediction, setCacheEntry } from "core";
import type { PredictionResult } from "core";

jest.mock("@upstash/redis", () => {
  const mockIncr = jest.fn();
  const mockExpire = jest.fn();
  const mockGet = jest.fn();
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

const { Redis } = jest.requireMock("@upstash/redis") as {
  Redis: { fromEnv: () => Record<string, jest.Mock> };
};

describe("checkRateLimit", () => {
  const cardId = "card_123";

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("allows first request within limit", async () => {
    const redis = Redis.fromEnv();
    redis.incr.mockResolvedValue(1);

    const result = await checkRateLimit(cardId);

    expect(result).toBe(true);
    expect(redis.incr).toHaveBeenCalledWith(`ratelimit:${cardId}`);
    expect(redis.expire).toHaveBeenCalledWith(`ratelimit:${cardId}`, 60);
  });

  it("allows request at exact max limit", async () => {
    const redis = Redis.fromEnv();
    redis.incr.mockResolvedValue(10);

    const result = await checkRateLimit(cardId);

    expect(result).toBe(true);
    expect(redis.expire).not.toHaveBeenCalled();
  });

  it("denies request over max limit", async () => {
    const redis = Redis.fromEnv();
    redis.incr.mockResolvedValue(11);

    const result = await checkRateLimit(cardId);

    expect(result).toBe(false);
  });

  it("sets expiry only on first request", async () => {
    const redis = Redis.fromEnv();
    redis.incr.mockResolvedValueOnce(1);

    await checkRateLimit(cardId);

    expect(redis.expire).toHaveBeenCalledTimes(1);

    redis.incr.mockResolvedValueOnce(5);
    await checkRateLimit(cardId);

    expect(redis.expire).toHaveBeenCalledTimes(1);
  });

  it("handles different card IDs independently", async () => {
    const redis = Redis.fromEnv();
    redis.incr.mockResolvedValue(1);

    await checkRateLimit("card_a");
    await checkRateLimit("card_b");

    expect(redis.incr).toHaveBeenCalledWith("ratelimit:card_a");
    expect(redis.incr).toHaveBeenCalledWith("ratelimit:card_b");
  });
});

describe("getCachedPrediction", () => {
  const cardId = "card_456";
  const amount = 99.99;

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("returns cached prediction when key exists", async () => {
    const mockResult: PredictionResult = {
      prediction: 1,
      probability: 0.85,
      probabilities_per_model: {
        xgboost: 0.8,
        random_forest: 0.7,
        logistic_regression: 0.6,
      },
      model_version: "ensemble_v1",
    };
    const redis = Redis.fromEnv();
    redis.get.mockResolvedValue(mockResult);

    const result = await getCachedPrediction(cardId, amount);

    expect(result).toEqual(mockResult);
    expect(redis.get).toHaveBeenCalledWith(`cache:${cardId}:${amount}`);
  });

  it("returns null when no cache entry exists", async () => {
    const redis = Redis.fromEnv();
    redis.get.mockResolvedValue(null);

    const result = await getCachedPrediction(cardId, amount);

    expect(result).toBeNull();
  });
});

describe("setCacheEntry", () => {
  const cardId = "card_789";
  const amount = 49.99;
  const result: PredictionResult = {
    prediction: 0,
    probability: 0.05,
    probabilities_per_model: {
      xgboost: 0.03,
      random_forest: 0.04,
      logistic_regression: 0.02,
    },
    model_version: "ensemble_v1",
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("stores prediction result in cache with TTL", async () => {
    const redis = Redis.fromEnv();

    await setCacheEntry(cardId, amount, result);

    expect(redis.set).toHaveBeenCalledWith(`cache:${cardId}:${amount}`, result, {
      ex: 300,
    });
  });
});
