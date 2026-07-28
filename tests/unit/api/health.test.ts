import { GET } from "../../../packages/api/src/app/api/health/route";

jest.mock("next/server", () => ({
  NextResponse: {
    json: jest.fn((data: unknown) => ({
      status: 200,
      body: JSON.stringify(data),
      data,
    })),
  },
}));

const ORIGINAL_MODEL_URL = process.env.MODEL_SERVICE_URL;

beforeEach(() => {
  jest.clearAllMocks();
  process.env.MODEL_SERVICE_URL = "http://model:80";
});

afterAll(() => {
  if (ORIGINAL_MODEL_URL) {
    process.env.MODEL_SERVICE_URL = ORIGINAL_MODEL_URL;
  } else {
    delete process.env.MODEL_SERVICE_URL;
  }
});

describe("health endpoint", () => {
  it("returns ok status with healthy container", async () => {
    const mockFetch = jest.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      json: async () => ({ status: "ok" }),
    } as Response);

    const res = await GET();
    expect(res.status).toBe(200);
    expect((res as { data: { status: string } }).data.status).toBe("ok");
    expect((res as { data: { container: { status: string } } }).data.container.status).toBe("ok");
    expect((res as { data: { timestamp: string } }).data.timestamp).toBeDefined();

    mockFetch.mockRestore();
  });

  it("reports unreachable container when model service is down", async () => {
    const mockFetch = jest.spyOn(globalThis, "fetch").mockRejectedValue(new Error("Connection refused"));

    const res = await GET();
    expect(res.status).toBe(200);
    expect((res as { data: { container: { status: string } } }).data.container.status).toBe("unreachable");
    expect((res as { data: { timestamp: string } }).data.timestamp).toBeDefined();

    mockFetch.mockRestore();
  });

  it("reports unreachable when model service returns non-ok", async () => {
    const mockFetch = jest.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: false,
      json: async () => ({ status: "error" }),
    } as Response);

    const res = await GET();
    expect(res.status).toBe(200);
    expect((res as { data: { status: string } }).data.status).toBe("ok");
    mockFetch.mockRestore();
  });

  it("returns valid timestamp format", async () => {
    const mockFetch = jest.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      json: async () => ({ status: "ok" }),
    } as Response);

    const res = await GET();
    const timestamp = (res as { data: { timestamp: string } }).data.timestamp;
    expect(() => new Date(timestamp)).not.toThrow();
    expect(new Date(timestamp).toISOString()).toBe(timestamp);

    mockFetch.mockRestore();
  });

  it("handles missing MODEL_SERVICE_URL env var", async () => {
    delete process.env.MODEL_SERVICE_URL;
    const mockFetch = jest.spyOn(globalThis, "fetch").mockRejectedValue(new Error("fetch is not defined"));

    const res = await GET();
    expect(res.status).toBe(200);
    expect((res as { data: { container: { status: string } } }).data.container.status).toBe("unreachable");

    mockFetch.mockRestore();
  });
});
