import type { UsageSummary } from "../types/usage";
import { request } from "./client";

export function fetchUsage(): Promise<UsageSummary> {
  return request<UsageSummary>("/usage");
}
