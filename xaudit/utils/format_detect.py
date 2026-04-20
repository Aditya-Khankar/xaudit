def detect_format(trace_dict: dict) -> str:
    """Auto-detect trace format from structure fingerprints."""
    if not isinstance(trace_dict, dict):
        raise ValueError("Trace must be a JSON object")
    if "run_type" in trace_dict or "runs" in trace_dict:
        return "langsmith"
    if "observations" in trace_dict:
        return "langfuse"
    if "events" in trace_dict:
        return "raw"
    raise ValueError(
        "Could not detect trace format. "
        "Use --format langsmith | langfuse | raw to specify explicitly."
    )
