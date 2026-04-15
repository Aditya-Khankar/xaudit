"""Detects performance context_decay — agent efficiency declining over
the course of a session, detected via CUSUM changepoint analysis."""

import numpy as np
from xaudit.detectors.base import BaseDetector, DetectorResult
from xaudit.recorder.trace_recorder import AgentTrace


class ContextDecayDetector(BaseDetector):
    name = "context_decay"
    version = "1.0.0"
    threshold = 0.40
    window_size = 5

    def detect(self, trace: AgentTrace) -> DetectorResult:
        if trace.total_steps < self.window_size * 2:
            return self._insufficient(
                f"Need at least {self.window_size * 2} steps for context_decay analysis."
            )

        # Efficiency per rolling window: success rate
        events = trace.events
        efficiencies = []
        for i in range(0, len(events) - self.window_size + 1, self.window_size):
            chunk = events[i: i + self.window_size]
            eff = sum(1 for e in chunk if e.success) / len(chunk)
            efficiencies.append(eff)

        if len(efficiencies) < 2:
            return self._insufficient("Insufficient windows for context_decay analysis.")

        eff_array = np.array(efficiencies, dtype=float)

        # Guard: all-zero efficiency (complete failure throughout)
        if eff_array.max() == 0.0:
            return DetectorResult(
                detector_name=self.name,
                detected=True,
                score=1.0,
                threshold=self.threshold,
                evidence={
                    "changepoint_detected": True,
                    "changepoint_step": 0,
                    "efficiency_trend": "complete_failure",
                    "cusum_max": 1.0,
                    "efficiency_values": efficiencies,
                },
                interpretation="Agent had 0% success rate across all windows — complete context_decay.",
                detector_version=self.version,
            )

        # Guard: all-one efficiency (perfect throughout — no context_decay possible)
        if eff_array.min() == 1.0:
            return self._insufficient(
                "Agent had 100% success rate throughout — no context_decay."
            )

        # CUSUM: cumulative sum control chart
        # Detects when efficiency drops below expected level
        target = float(eff_array.mean())
        std = float(eff_array.std())
        k = 0.5 * std if std > 0 else 0.1  # allowance parameter

        cusum = 0.0
        cusum_values = []
        for eff in efficiencies:
            # Lower CUSUM: detects downward shift (context_decay)
            cusum = max(0.0, cusum + (target - eff) - k)
            cusum_values.append(cusum)

        max_cusum = float(max(cusum_values))

        # Normalize score: divide by efficiency range
        eff_range = eff_array.max() - eff_array.min()
        score = self._clamp(max_cusum / (eff_range + 1e-6))

        changepoint_detected = score >= self.threshold
        changepoint_window = int(np.argmax(cusum_values)) if changepoint_detected else None
        changepoint_step = (
            changepoint_window * self.window_size if changepoint_window is not None else None
        )

        return DetectorResult(
            detector_name=self.name,
            detected=changepoint_detected,
            score=score,
            threshold=self.threshold,
            evidence={
                "changepoint_detected": changepoint_detected,
                "changepoint_step": changepoint_step,
                "efficiency_trend": "declining" if changepoint_detected else "stable",
                "cusum_max": round(max_cusum, 3),
                "efficiency_values": [round(e, 3) for e in efficiencies],
                "baseline_efficiency": round(target, 3),
            },
            interpretation=(
                f"Efficiency changepoint at step {changepoint_step}. "
                f"Baseline was {target:.0%}."
                if changepoint_detected
                else f"Agent efficiency stable at ~{target:.0%} throughout."
            ),
            detector_version=self.version,
        )
