import { Redis } from "@upstash/redis"
import type { PredictionResult } from "."

export const redis = Redis.fromEnv();

const RATE_LIMIT_WINDOW = 60;
const RATE_LIMIT_MAX = 10;
const CACHE_TTL = 300;

function rateLimitKey(cardId : string){
  return `ratelimit:${cardId}`;
}

function cacheKey(cardId: string, amount : number){
  return `cache:${cardId}:${amount}`
}

export async function checkRateLimit(cardId : string): Promise<boolean> {
  const key = rateLimitKey(cardId);
  const count = await redis.incr(key);
  if (count === 1){
    await redis.expire(key, RATE_LIMIT_WINDOW)
  }

  return count <= RATE_LIMIT_MAX;
}

export async function getCachedPrediction(cardId:string, amount:number): Promise<PredictionResult | null> {
  const key = cacheKey(cardId, amount);
  return redis.get<PredictionResult>(key);
}

export async function setCacheEntry(cardId:string, amount:number, result:PredictionResult): Promise<Void> {
  const key = cacheKey(cardId,amount);
  await redis.set(key,result,{ex:CACHE_TTL});
}
