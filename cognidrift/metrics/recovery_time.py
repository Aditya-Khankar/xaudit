from cognidrift.recorder.trace_recorder import AgentTrace


def compute_recovery_time(trace: AgentTrace) -> dict:
    events = trace.events
    failure_indices = [e.step_index for e in trace.failures]

    if not failure_indices:
        return {
            "value": 0,
            "unit": "steps",
            "failure_events": 0,
            "avg_steps_to_recovery": 0,
            "per_failure_recovery": [],
            "interpretation": "No failure events — recovery time not applicable.",
        }

    per_failure = []
    for fail_step in failure_indices:
        # Find next success after this failure
        recovery = None
        for e in events:
            if e.step_index > fail_step and e.success:
                recovery = e.step_index - fail_step
                break
        per_failure.append(recovery)

    # Filter out None (no recovery found after some failures)
    recovered = [r for r in per_failure if r is not None]
    avg = sum(recovered) / len(recovered) if recovered else 0

    return {
        "value": round(avg, 2),
        "unit": "steps",
        "failure_events": len(failure_indices),
        "avg_steps_to_recovery": round(avg, 2),
        "per_failure_recovery": per_failure,
        "interpretation": (
            f"Average {avg:.1f} steps to recover from each failure. "
            f"{len(failure_indices) - len(recovered)} failures had no recovery."
            if failure_indices
            else "No failures detected."
        ),
    }
