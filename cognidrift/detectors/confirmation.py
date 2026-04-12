"""Detects confirmation bias — agent narrows tool usage over time,
seeking confirmation rather than exploring alternative information sources."""

import numpy as np
from cognidrift.detectors.base import BaseDetector, DetectorResult
from cognidrift.recorder.trace_recorder import AgentTrace


class ConfirmationDetector(BaseDetector):
    name = "confirmation_bias"
    version = "1.0.0"
    threshold = 0.60
    window_size = 5

    def detect(self, trace: AgentTrace) -> DetectorResult:
        tool_events = [
            e for e in trace.events
            if e.event_type in ("tool_call", "retrieval")
        ]

        if len(tool_events) < self.window_size * 2:
            return self._insufficient(
                f"Need at least {self.window_size * 2} tool events for confirmation analysis."
            )

        # Compute unique tool count in rolling windows
        diversity_values = []
        for i in range(len(tool_events) - self.window_size + 1):
            window = tool_events[i: i + self.window_size]
            unique = len(set(e.tool_name or e.event_type for e in window))
            diversity_values.append(unique)

        if len(diversity_values) < 2:
            return self._insufficient("Insufficient windows for diversity trend.")

        # Linear regression slope on diversity over time
        x = np.arange(len(diversity_values), dtype=float)
        y = np.array(diversity_values, dtype=float)
        slope = float(np.polyfit(x, y, 1)[0])

        # Normalize: total drop as a fraction of initial diversity
        # Total expected drop = -slope * (len(diversity_values) - 1)
        total_drop = -slope * max(len(diversity_values) - 1, 1)
        score = self._clamp(total_drop / max(diversity_values[0], 1))

        # First half vs second half unique tools
        mid = len(tool_events) // 2
        first_half_tools = set(e.tool_name or e.event_type for e in tool_events[:mid])
        second_half_tools = set(e.tool_name or e.event_type for e in tool_events[mid:])

        # Tool usage distribution
        usage = {}
        for e in tool_events:
            key = e.tool_name or e.event_type
            usage[key] = usage.get(key, 0) + 1

        return DetectorResult(
            detector_name=self.name,
            detected=score >= self.threshold,
            score=score,
            threshold=self.threshold,
            evidence={
                "tool_diversity_slope": round(slope, 4),
                "unique_tools_first_half": len(first_half_tools),
                "unique_tools_second_half": len(second_half_tools),
                "diversity_values": [round(v, 2) for v in diversity_values],
                "tool_usage_distribution": usage,
            },
            interpretation=(
                f"Tool diversity slope: {slope:.3f} per window. "
                + (
                    f"Narrowed from {len(first_half_tools)} to {len(second_half_tools)} unique tools. "
                    "Confirmation bias pattern detected."
                    if score >= self.threshold
                    else "Tool diversity relatively stable — no confirmation bias detected."
                )
            ),
            detector_version=self.version,
        )
