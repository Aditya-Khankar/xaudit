from cognidrift.recorder.trace_recorder import AgentTrace


def compute_exploration_score(trace: AgentTrace) -> dict:
    tool_events = [
        e for e in trace.events
        if e.event_type in ("tool_call", "retrieval")
    ]

    usage: dict[str, int] = {}
    for e in tool_events:
        key = e.tool_name or e.event_type
        usage[key] = usage.get(key, 0) + 1

    unique_used = len(usage)

    return {
        "value": unique_used,
        "unique_tools_used": unique_used,
        "total_tools_available": None,
        "tool_usage_distribution": usage,
        "interpretation": (
            f"Used {unique_used} distinct tool types. "
            "Pass --tool-manifest to compute exact exploration ratio."
        ),
    }
