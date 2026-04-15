"""
Query Entropy Collapse Detector

Detects narrowing of tool/query diversity using Rényi entropy (α=2).
Falling entropy over time indicates the agent is reducing its exploration
space — seeking confirmation rather than new information.

Method: Rényi entropy (α=2) over sliding windows, with linear trend analysis.
        α=2 provides quadratic weighting of probabilities, making it more
        sensitive to probability mass concentration than Shannon entropy.
Field:  Information theory.
"""

import numpy as np
from xaudit.detectors.base import BaseDetector, DetectorResult
from xaudit.recorder.trace_recorder import AgentTrace


def renyi_entropy_alpha2(tool_counts: dict) -> float:
    """
    Compute Rényi entropy with α=2.

    Formula: H₂(X) = -log₂(Σᵢ pᵢ²)

    Args:
        tool_counts: dict mapping tool_name → usage_count in window

    Returns:
        float: entropy value. Higher = more diverse tool usage.
               0.0 = single tool dominance.
               log₂(N) = perfectly uniform across N tools.
    """
    total = sum(tool_counts.values())
    if total == 0:
        return 0.0
    probabilities = [count / total for count in tool_counts.values()]
    sum_squared = sum(p ** 2 for p in probabilities)
    if sum_squared == 0 or sum_squared == 1.0:
        return 0.0
    return float(-np.log2(sum_squared))


class QueryEntropyCollapseDetector(BaseDetector):
    name = "query_entropy_collapse_bias"
    version = "1.0.0"
    threshold = 0.40
    window_size = 5

    def detect(self, trace: AgentTrace) -> DetectorResult:
        tool_events = [
            e for e in trace.events
            if e.event_type in ("tool_call", "retrieval")
        ]

        if len(tool_events) == 0:
            return self._insufficient("No tool events found.")

        # If we don't have enough events for 2 windows, we can't establish a trend
        if len(tool_events) < self.window_size + 1:
            return self._insufficient(
                f"Need at least {self.window_size + 1} tool events for trend analysis."
            )

        diversity_values = []
        for i in range(len(tool_events) - self.window_size + 1):
            window = tool_events[i: i + self.window_size]
            counts = {}
            for e in window:
                key = e.tool_name or e.event_type
                counts[key] = counts.get(key, 0) + 1
            entropy = renyi_entropy_alpha2(counts)
            diversity_values.append(entropy)

        if len(diversity_values) < 2:
            return self._insufficient("Insufficient windows for diversity trend.")

        # Check if single tool type used throughout (entropy always 0)
        # All windows having exactly 0.0 entropy means no diversity ever
        if all(e == 0.0 for e in diversity_values):
            # First half vs second half unique tools
            mid = len(tool_events) // 2
            first_half_tools = set(e.tool_name or e.event_type for e in tool_events[:mid])
            second_half_tools = set(e.tool_name or e.event_type for e in tool_events[mid:])
            
            usage = {}
            for e in tool_events:
                key = e.tool_name or e.event_type
                usage[key] = usage.get(key, 0) + 1
                
            return DetectorResult(
                detector_name=self.name,
                detected=True,
                score=1.0,
                threshold=self.threshold,
                evidence={
                    "tool_diversity_slope": 0.0,
                    "unique_tools_first_half": len(first_half_tools),
                    "unique_tools_second_half": len(second_half_tools),
                    "diversity_values": [round(v, 2) for v in diversity_values],
                    "tool_usage_distribution": usage,
                },
                interpretation="Single tool dominates completely. No diversity. QueryEntropyCollapse bias detected.",
                detector_version=self.version,
            )

        # Linear regression slope on entropy over time
        x = np.arange(len(diversity_values), dtype=float)
        y = np.array(diversity_values, dtype=float)
        slope = float(np.polyfit(x, y, 1)[0])

        initial_entropy = diversity_values[0]
        final_entropy = diversity_values[-1]
        
        # If all windows have identical entropy, slope will be ~0
        if np.isclose(slope, 0.0):
            score = 0.0
            detected = False
        else:
            if slope < 0 and initial_entropy > 0 and final_entropy < 0.5 * initial_entropy:
                drop_ratio = (initial_entropy - final_entropy) / initial_entropy
                score = self._clamp(drop_ratio)
                detected = score > self.threshold
            else:
                score = 0.0
                detected = False

        # First half vs second half unique tools
        mid = len(tool_events) // 2
        first_half_tools = set(e.tool_name or e.event_type for e in tool_events[:mid])
        second_half_tools = set(e.tool_name or e.event_type for e in tool_events[mid:])

        # Tool usage distribution
        usage = {}
        for e in tool_events:
            key = e.tool_name or e.event_type
            usage[key] = usage.get(key, 0) + 1
            
        interpretation_suffix = (
            "QueryEntropyCollapse bias pattern detected."
            if detected
            else "Tool diversity relatively stable or expanding — no query_entropy_collapse bias detected."
        )

        return DetectorResult(
            detector_name=self.name,
            detected=detected,
            score=score,
            threshold=self.threshold,
            evidence={
                "tool_diversity_slope": round(slope, 4),
                "unique_tools_first_half": len(first_half_tools),
                "unique_tools_second_half": len(second_half_tools),
                "diversity_values": [round(v, 2) for v in diversity_values],
                "tool_usage_distribution": usage,
            },
            interpretation=f"Tool diversity slope: {slope:.3f}. {interpretation_suffix}",
            detector_version=self.version,
        )
