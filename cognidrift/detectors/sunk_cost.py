"""Detects sunk cost fallacy — agent continuing a failing strategy
past the rational pivot point instead of adapting."""

from cognidrift.detectors.base import BaseDetector, DetectorResult
from cognidrift.recorder.trace_recorder import AgentTrace


class SunkCostDetector(BaseDetector):
    name = "sunk_cost"
    version = "1.0.0"
    threshold = 0.50

    def detect(self, trace: AgentTrace) -> DetectorResult:
        if not trace.failures:
            return self._insufficient("No failure events in trace — sunk cost not applicable.")

        if trace.total_steps < 6:
            return self._insufficient("Trace too short for sunk cost analysis.")

        # First failure step
        first_failure_step = trace.failures[0].step_index
        failure_step_indices = [e.step_index for e in trace.failures]

        # Tools used before first failure
        tools_before_failure = set(
            e.tool_name or e.event_type
            for e in trace.events
            if e.step_index < first_failure_step
               and e.event_type in ("tool_call", "retrieval")
        )

        # Strategy change = first step after failure using a NEW tool
        strategy_change_step = None
        for event in trace.events:
            if event.step_index <= first_failure_step:
                continue
            if event.event_type not in ("tool_call", "retrieval"):
                continue
            tool = event.tool_name or event.event_type
            if tool not in tools_before_failure:
                strategy_change_step = event.step_index
                break

        # If agent never pivoted — maximum sunk cost
        if strategy_change_step is None:
            steps_after_failure = trace.total_steps - first_failure_step
            score = 1.0
            interpretation = (
                f"Agent never changed strategy after first failure at step {first_failure_step}. "
                f"Continued same approach for {steps_after_failure} remaining steps."
            )
        else:
            steps_after_failure = strategy_change_step - first_failure_step
            score = self._clamp(steps_after_failure / max(trace.total_steps, 1))
            interpretation = (
                f"First failure at step {first_failure_step}. "
                f"Strategy changed at step {strategy_change_step} "
                f"({steps_after_failure} steps later). "
                + (
                    "Significant sunk cost — slow to adapt."
                    if score >= self.threshold
                    else "Reasonable adaptation speed."
                )
            )

        return DetectorResult(
            detector_name=self.name,
            detected=score >= self.threshold,
            score=score,
            threshold=self.threshold,
            evidence={
                "first_failure_step": first_failure_step,
                "strategy_change_step": strategy_change_step,
                "steps_after_failure": steps_after_failure
                if strategy_change_step
                else trace.total_steps - first_failure_step,
                "total_steps": trace.total_steps,
                "failure_signals": failure_step_indices,
                "never_pivoted": strategy_change_step is None,
            },
            interpretation=interpretation,
            detector_version=self.version,
        )
