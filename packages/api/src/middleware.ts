import { NextResponse } from "next/server";
import { NextRequest } from "next/server";

export function middleware(request : NextRequest){
    if (request.nextUrl.pathname === "/api/health"){
        return NextResponse.next()
    }
    const apiKey = request.headers.get("x-api-key")

    if (!apiKey || apiKey !== process.env.API_KEY) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    return NextResponse.next();
}

export const config = {
    matcher : "/api/:path*"
}
