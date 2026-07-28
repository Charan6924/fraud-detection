jest.mock("@aws-sdk/client-dynamodb", () => ({
  DynamoDBClient: jest.fn(() => ({})),
}));

const mockSend = jest.fn();
jest.mock("@aws-sdk/lib-dynamodb", () => ({
  DynamoDBDocumentClient: {
    from: jest.fn(() => ({
      send: mockSend,
    })),
  },
  ScanCommand: jest.fn(),
  GetCommand: jest.fn(),
}));

const mockNextResponseJson = jest.fn();
jest.mock("next/server", () => ({
  NextResponse: {
    json: (...args: unknown[]) => mockNextResponseJson(...args),
  },
  NextRequest: jest.fn(),
}));

import { GET } from "../../../packages/api/src/app/api/transactions/route";

beforeEach(() => {
  jest.clearAllMocks();
  process.env.PREDICTIONS_TABLE = "predictions-table";

  mockNextResponseJson.mockImplementation((data: unknown, init?: { status: number }) => ({
    status: init?.status ?? 200,
    data,
  }));
});

function mockRequest(searchParams?: Record<string, string>) {
  return {
    nextUrl: {
      searchParams: new Map(Object.entries(searchParams ?? {})),
    },
  } as unknown as Request;
}

describe("transactions endpoint", () => {
  it("returns sorted list of transactions when no id is given", async () => {
    const items = [
      { id: "1", timestamp: "2026-07-28T12:00:00.000Z", input: { TransactionAmt: 10 }, result: { prediction: 0 } },
      { id: "2", timestamp: "2026-07-28T13:00:00.000Z", input: { TransactionAmt: 20 }, result: { prediction: 1 } },
      { id: "3", timestamp: "2026-07-28T11:00:00.000Z", input: { TransactionAmt: 30 }, result: { prediction: 0 } },
    ];
    mockSend.mockResolvedValue({ Items: items });

    const req = mockRequest({});
    const res = await GET(req);

    expect(res.status).toBe(200);
    expect(Array.isArray(res.data)).toBe(true);
    expect(res.data).toHaveLength(3);
    // Should be sorted by timestamp descending
    expect(res.data[0].id).toBe("2");
    expect(res.data[1].id).toBe("1");
    expect(res.data[2].id).toBe("3");
  });

  it("returns empty array when table has no items", async () => {
    mockSend.mockResolvedValue({ Items: [] });

    const req = mockRequest({});
    const res = await GET(req);

    expect(res.status).toBe(200);
    expect(res.data).toEqual([]);
  });

  it("returns empty array when Items is undefined", async () => {
    mockSend.mockResolvedValue({});

    const req = mockRequest({});
    const res = await GET(req);

    expect(res.status).toBe(200);
    expect(res.data).toEqual([]);
  });

  it("returns a single transaction by id", async () => {
    const item = {
      id: "txn-42",
      timestamp: "2026-07-28T12:00:00.000Z",
      input: { TransactionAmt: 150.0 },
      result: { prediction: 1, probability: 0.87 },
    };
    mockSend.mockResolvedValue({ Item: item });

    const req = mockRequest({ id: "txn-42" });
    const res = await GET(req);

    expect(res.status).toBe(200);
    expect(res.data).toEqual(item);
  });

  it("returns 404 when transaction id is not found", async () => {
    mockSend.mockResolvedValue({});

    const req = mockRequest({ id: "nonexistent" });
    const res = await GET(req);

    expect(res.status).toBe(404);
    expect(res.data).toEqual({ error: "Transaction not found" });
  });

  it("handles DynamoDB scan failure", async () => {
    mockSend.mockRejectedValue(new Error("DynamoDB error"));

    const req = mockRequest({});
    await expect(GET(req)).rejects.toThrow("DynamoDB error");
  });
});
