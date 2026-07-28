import type { Config } from "jest";

const config: Config = {
  projects: [
    {
      displayName: "core",
      rootDir: "packages/core",
      roots: ["<rootDir>", "<rootDir>/../../tests/unit/core"],
      testMatch: ["<rootDir>/../../tests/unit/core/**/*.test.ts"],
      testEnvironment: "node",
      transform: { "^.+\\.ts$": ["ts-jest", { tsconfig: "packages/core/tsconfig.json" }] },
      moduleNameMapper: { "^core$": "<rootDir>/src/index.ts" },
    },
    {
      displayName: "monitor",
      rootDir: "packages/monitor",
      roots: ["<rootDir>", "<rootDir>/../../tests/unit/monitor"],
      testMatch: ["<rootDir>/../../tests/unit/monitor/**/*.test.ts"],
      testEnvironment: "node",
      transform: { "^.+\\.ts$": ["ts-jest", { tsconfig: "packages/monitor/tsconfig.json" }] },
    },
    {
      displayName: "api",
      rootDir: "packages/api",
      roots: ["<rootDir>", "<rootDir>/../../tests/unit/api"],
      testMatch: ["<rootDir>/../../tests/unit/api/**/*.test.ts"],
      testEnvironment: "node",
      transform: { "^.+\\.ts$": ["ts-jest", { tsconfig: "packages/api/tsconfig.json" }] },
      moduleNameMapper: { "^core$": "<rootDir>/../core/src/index.ts" },
    },
  ],
  testPathIgnorePatterns: ["/node_modules/", "/.next/", "/dist/"],
  collectCoverageFrom: ["packages/*/src/**/*.ts"],
  coverageDirectory: "coverage",
};

export default config;
