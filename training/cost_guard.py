"""Estimated GPU-compute budget guard; not a guarantee of provider billing."""
from __future__ import annotations

import time


class EstimatedCostGuard:
    def __init__(self, rate_usd_per_hour=None, budget_usd=None, confirmed=False,
                 reserve_seconds=60.0, clock=time.monotonic):
        active = rate_usd_per_hour is not None or budget_usd is not None
        if active and (rate_usd_per_hour is None or budget_usd is None):
            raise ValueError("cost guard requires both GPU rate and budget cap")
        if active and (float(rate_usd_per_hour) <= 0 or float(budget_usd) <= 0):
            raise ValueError("GPU rate and budget cap must be positive")
        if active and not confirmed:
            raise PermissionError("paid GPU cost settings require --confirm-potential-cost")
        if float(reserve_seconds) < 0:
            raise ValueError("reserve_seconds cannot be negative")
        self.active = active
        self.rate = float(rate_usd_per_hour) if active else None
        self.budget = float(budget_usd) if active else None
        self.reserve_seconds = float(reserve_seconds)
        self.clock = clock
        self.started = clock()

    def elapsed_seconds(self):
        return max(0.0, self.clock() - self.started)

    def estimated_cost_usd(self, additional_seconds=0.0):
        if not self.active:
            return 0.0
        return max(0.0, self.elapsed_seconds() + float(additional_seconds)) * self.rate / 3600.0

    def should_stop_before_next_step(self, measured_step_seconds=None):
        if not self.active:
            return False
        reserve = max(self.reserve_seconds, float(measured_step_seconds or 0.0))
        return self.estimated_cost_usd(reserve) >= self.budget

    def report(self):
        return {
            "active": self.active,
            "gpu_rate_usd_per_hour": self.rate,
            "budget_cap_usd": self.budget,
            "elapsed_seconds_since_guard_start": round(self.elapsed_seconds(), 3),
            "estimated_gpu_compute_cost_usd": round(self.estimated_cost_usd(), 6),
            "billing_scope": "GPU rate × local training-process elapsed time only; excludes container startup, CPU/RAM, storage, network, taxes and provider billing granularity",
            "hard_billing_cap": False,
        }
