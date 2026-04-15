"""Input validation — runs before any processing. Prevents malformed or
oversized traces from reaching detectors."""

MAX_EVENTS = 500
MAX_STRING_LENGTH = 50_000
MAX_TRACE_SIZE_BYTES = 10 * 1024 * 1024  # 10MB


class TraceValidationError(ValueError):
    pass


def validate_raw_trace(data: dict) -> None:
    """Validate a raw trace dict before adapter processing."""
    if not isinstance(data, dict):
        raise TraceValidationError("Trace must be a JSON object")

    events = data.get("events", [])

    if not isinstance(events, list):
        raise TraceValidationError("'events' must be a list")

    if len(events) > MAX_EVENTS:
        raise TraceValidationError(
            f"Trace has {len(events)} events — limit is {MAX_EVENTS}. "
            "Split into smaller traces for analysis."
        )

    for i, event in enumerate(events):
        if not isinstance(event, dict):
            raise TraceValidationError(f"Event at index {i} must be an object")

        # Prevent memory exhaustion from huge output fields
        output_str = str(event.get("output", ""))
        if len(output_str) > MAX_STRING_LENGTH:
            raise TraceValidationError(
                f"Event {i} output exceeds {MAX_STRING_LENGTH} character limit. "
                "Truncate large outputs before analysis."
            )

        input_str = str(event.get("input", ""))
        if len(input_str) > MAX_STRING_LENGTH:
            raise TraceValidationError(
                f"Event {i} input exceeds {MAX_STRING_LENGTH} character limit."
            )

        # Required fields
        if "step" not in event:
            raise TraceValidationError(f"Event {i} missing required field 'step'")
        if "type" not in event:
            raise TraceValidationError(f"Event {i} missing required field 'type'")
        if "success" not in event:
            raise TraceValidationError(f"Event {i} missing required field 'success'")

        # Valid event types
        valid_types = {"tool_call", "llm_call", "retrieval", "error"}
        if event["type"] not in valid_types:
            raise TraceValidationError(
                f"Event {i} has invalid type '{event['type']}'. "
                f"Must be one of: {valid_types}"
            )
