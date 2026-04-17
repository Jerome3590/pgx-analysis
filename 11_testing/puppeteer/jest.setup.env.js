"use strict";

// Default to production URLs when env vars aren't inherited (e.g. WSL → Windows npx interop).
// Override by setting DASHBOARD_URL / API_BASE_URL before running jest.
if (!process.env.DASHBOARD_URL) {
  process.env.DASHBOARD_URL = "https://jerome-dixon.io/vcu/pgx-risk-calculator/index.html";
}
if (!process.env.API_BASE_URL) {
  process.env.API_BASE_URL = "https://cmv0qislq3.execute-api.us-east-1.amazonaws.com/prod";
}
