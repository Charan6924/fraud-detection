FROM node:20-slim AS base
WORKDIR /app

# Dependencies
COPY package.json package-lock.json ./
COPY packages/core/package.json packages/core/package.json
COPY packages/api/package.json packages/api/package.json
RUN npm ci

# Source
COPY tsconfig.json ./
COPY packages/core/ packages/core/
COPY packages/api/ packages/api/
COPY sst-env.d.ts ./

# Build
RUN npm run build -w api

EXPOSE 3000
CMD ["npm", "run", "start", "-w", "api"]
