import type { CostBreakdown } from "../types/chat";
import type { UsageSummary } from "../types/usage";

interface Props {
  usage: UsageSummary | null;
  lastCost: CostBreakdown | null;
}

const credits = new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 });
const tokens = new Intl.NumberFormat();

function formatUsd(value: string): string {
  const n = Number(value);
  return n === 0 ? "$0" : `$${n.toPrecision(3)}`;
}

export default function UsageIndicator({ usage, lastCost }: Props) {
  if (!usage) {
    return <div className="usage usage--loading">Loading usage...</div>;
  }
  return (
    <div className="usage" aria-label="Usage">
      <div className="usage__primary">{credits.format(Number(usage.credits_remaining))} credits</div>
      <div>{tokens.format(usage.total_tokens)} tokens</div>
      {lastCost && (
        <div title="Cost of the last request">last: {formatUsd(lastCost.total_cost)}</div>
      )}
    </div>
  );
}
