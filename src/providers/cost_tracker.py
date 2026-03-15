"""Track monthly API spend across providers."""

import json
import logging
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

COST_LOG = Path(__file__).parent.parent.parent / "data" / "cost_log.jsonl"


def log_cost(
    model: str,
    provider: str,
    input_tokens: int,
    output_tokens: int,
    cost: float,
) -> None:
    """Append a cost entry to the JSONL log."""
    COST_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "provider": provider,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost": cost,
    }
    with open(COST_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def get_monthly_spend() -> float:
    """Sum costs for the current calendar month."""
    if not COST_LOG.exists():
        return 0.0
    month_prefix = datetime.now().strftime("%Y-%m")
    total = 0.0
    with open(COST_LOG, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                if entry["timestamp"].startswith(month_prefix):
                    total += entry["cost"]
            except (json.JSONDecodeError, KeyError):
                continue
    return total


def check_budget(budget: float, alert_threshold: float | None = None) -> bool:
    """Check if monthly spend is within budget. Logs warning if near limit.

    Returns True if within budget, False if exceeded.
    """
    spend = get_monthly_spend()
    if alert_threshold and spend >= alert_threshold:
        log.warning(
            "Monthly spend $%.4f approaching budget $%.2f (alert at $%.2f)",
            spend,
            budget,
            alert_threshold,
        )
    if spend >= budget:
        log.error(
            "Monthly budget EXCEEDED: $%.4f >= $%.2f. Extraction blocked.",
            spend,
            budget,
        )
        return False
    return True
