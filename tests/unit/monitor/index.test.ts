const mockFetch = jest.fn();
globalThis.fetch = mockFetch;

beforeEach(() => {
  jest.clearAllMocks();
  process.env.MODEL_SERVICE_URL = "http://model:80";
  process.env.MODEL_SECRET = "model-secret-42";
  process.env.GH_TOKEN = "gh_token_abc";
});

describe("monitor handler", () => {
  it("returns drift_checked=true and dispatch_triggered=true when drift detected and GH dispatch succeeds", async () => {
    mockFetch
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          drift_detected: true,
          drift_share: 0.45,
          drifted_features: ["TransactionAmt", "V2"],
          n_drifted: 2,
          n_total: 100,
        }),
      })
      .mockResolvedValueOnce({
        ok: true,
        status: 204,
      });

    const { handler } = await import(
      "../../../packages/monitor/src/index"
    );
    const result = await handler();

    expect(result).toEqual({
      drift_checked: true,
      dispatch_triggered: true,
    });
    expect(mockFetch).toHaveBeenCalledTimes(2);
    expect(mockFetch).toHaveBeenNthCalledWith(
      1,
      "http://model:80/drift",
      expect.objectContaining({
        method: "POST",
        headers: { "x-model-secret": "model-secret-42" },
      }),
    );
    expect(mockFetch).toHaveBeenNthCalledWith(
      2,
      "https://api.github.com/repos/Charan6924/fraud-detection/dispatches",
      expect.objectContaining({
        method: "POST",
        headers: expect.objectContaining({
          Authorization: "Bearer gh_token_abc",
        }),
        body: JSON.stringify({ event_type: "retrain" }),
      }),
    );
  });

  it("returns drift_checked=true and dispatch_triggered=false when no drift detected", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        drift_detected: false,
        drift_share: 0.1,
        drifted_features: [],
        n_drifted: 0,
        n_total: 100,
      }),
    });

    const { handler } = await import(
      "../../../packages/monitor/src/index"
    );
    const result = await handler();

    expect(result).toEqual({
      drift_checked: true,
      dispatch_triggered: false,
    });
    expect(mockFetch).toHaveBeenCalledTimes(1);
  });

  it("returns no-dispatch when MODEL_SERVICE_URL is not set", async () => {
    delete process.env.MODEL_SERVICE_URL;

    const { handler } = await import(
      "../../../packages/monitor/src/index"
    );
    const result = await handler();

    expect(result).toEqual({
      drift_checked: false,
      reason: "MODEL_SERVICE_URL not set",
    });
    expect(mockFetch).not.toHaveBeenCalled();
  });

  it("returns no-dispatch when drift endpoint fails", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
      text: async () => "Internal error",
    });

    const { handler } = await import(
      "../../../packages/monitor/src/index"
    );
    const result = await handler();

    expect(result).toEqual({
      drift_checked: false,
      reason: "drift endpoint returned 500",
    });
  });

  it("returns no-dispatch when GH_TOKEN is not set", async () => {
    delete process.env.GH_TOKEN;

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        drift_detected: true,
        drift_share: 0.5,
        drifted_features: ["Amt"],
        n_drifted: 1,
        n_total: 50,
      }),
    });

    const { handler } = await import(
      "../../../packages/monitor/src/index"
    );
    const result = await handler();

    expect(result).toEqual({
      drift_checked: true,
      dispatch_triggered: false,
      reason: "no GH token",
    });
    expect(mockFetch).toHaveBeenCalledTimes(1);
  });

  it("returns no-dispatch when GH API fails", async () => {
    mockFetch
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          drift_detected: true,
          drift_share: 0.4,
          drifted_features: ["V3"],
          n_drifted: 1,
          n_total: 50,
        }),
      })
      .mockResolvedValueOnce({
        ok: false,
        status: 403,
        text: async () => "Forbidden",
      });

    const { handler } = await import(
      "../../../packages/monitor/src/index"
    );
    const result = await handler();

    expect(result).toEqual({
      drift_checked: true,
      dispatch_triggered: false,
      reason: "API error",
    });
    expect(mockFetch).toHaveBeenCalledTimes(2);
  });
});
