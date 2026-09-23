import type { DecimalString } from "./chat";

export interface UsageSummary {
  user_id: string;
  total_credits: DecimalString;
  credits_used: DecimalString;
  credits_remaining: DecimalString;
  request_count: number;
  total_input_tokens: number;
  total_output_tokens: number;
  total_tokens: number;
  total_cost: DecimalString;
  currency: string;
}
