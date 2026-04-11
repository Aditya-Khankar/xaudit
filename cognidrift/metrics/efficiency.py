from cognidrift.recorder.trace_recorder import AgentTrace


def compute_efficiency(trace: AgentTrace) -> dict:
    active_events = [
        e for e in trace.events
        if e.event_type in ("tool_call", "retrieval")
    ]
    goal_advancing = [e for e in active_events if e.success]
    total = trace.total_steps

    value = len(goal_advancing) / total if total > 0 else 0.0

    return {
        "value": round(value, 4),
        "goal_advancing_actions": len(goal_advancing),
        "total_actions": total,
        "interpretation": f"{value:.0%} of actions directly advanced the goal.",
    }
