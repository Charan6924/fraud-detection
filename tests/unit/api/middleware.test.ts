import { middleware } from "../../../packages/api/src/middleware";
import type { NextRequest } from "next/server";

const ORIGINAL_API_KEY = process.env.API_KEY;

function mockRequest(pathname: string, apiKey?: string): NextRequest {
  return {
    nextUrl: { pathname },
    headers: {
      get: (name: string) => {
        if (name === "x-api-key") return apiKey ?? null;
        return null;
      },
    },
  } as unknown as NextRequest;
}

jest.mock("next/server", () => ({
  NextResponse: {
    next: jest.fn(() => ({ status: 200, body: "next" })),
    json: jest.fn((data: unknown, init?: { status: number }) => ({
      status: init?.status ?? 200,
      body: JSON.stringify(data),
      data,
    })),
  },
}));

beforeEach(() => {
  jest.clearAllMocks();
  process.env.API_KEY = "test-api-key-123";
});

afterAll(() => {
  if (ORIGINAL_API_KEY) {
    process.env.API_KEY = ORIGINAL_API_KEY;
  } else {
    delete process.env.API_KEY;
  }
});

describe("middleware", () => {
  it("allows health route without API key", () => {
    const req = mockRequest("/api/health");
    const res = middleware(req);
    expect(res.status).toBe(200);
  });

  it("does not bypass auth for non-matching health-like paths", () => {
    const req = mockRequest("/api/health/db");
    const res = middleware(req);
    expect(res.status).toBe(401);
  });

  it("blocks request with no API key", () => {
    const req = mockRequest("/api/predict");
    const res = middleware(req);
    expect(res.status).toBe(401);
    expect((res as { data: { error: string } }).data.error).toBe("Unauthorized");
  });

  it("blocks request with wrong API key", () => {
    const req = mockRequest("/api/predict", "wrong-key");
    const res = middleware(req);
    expect(res.status).toBe(401);
    expect((res as { data: { error: string } }).data.error).toBe("Unauthorized");
  });

  it("allows request with valid API key", () => {
    const req = mockRequest("/api/predict", "test-api-key-123");
    const res = middleware(req);
    expect(res.status).toBe(200);
  });

  it("allows request with valid API key to transactions endpoint", () => {
    const req = mockRequest("/api/transactions", "test-api-key-123");
    const res = middleware(req);
    expect(res.status).toBe(200);
  });

  it("allows request with valid API key to nested API path", () => {
    const req = mockRequest("/api/v2/predict", "test-api-key-123");
    const res = middleware(req);
    expect(res.status).toBe(200);
  });

  it("rejects request when API_KEY env var is not set", () => {
    delete process.env.API_KEY;
    const req = mockRequest("/api/predict", "some-key");
    const res = middleware(req);
    expect(res.status).toBe(401);
  });

  it("rejects request with empty API key header", () => {
    const req = mockRequest("/api/predict", "");
    const res = middleware(req);
    expect(res.status).toBe(401);
  });
});
