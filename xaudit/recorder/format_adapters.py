"""Format adapters translate native trace formats into AgentTrace.

Rules:
- Adapters are dumb translators. No semantic interpretation here.
- All validation happens before adapters are called (validators.py).
- If a field is missing, use a safe default. Never crash on missing optional fields.
- Semantic decisions (what counts as failure, what counts as retrieval)
  belong in detectors, not here.
"""

import time
from xaudit.recorder.trace_recorder import AgentEvent, AgentTrace
from xaudit.utils.validators import validate_raw_trace


class RawJSONAdapter:
    """Handles the xaudit native raw JSON format."""

    def adapt(self, data: dict) -> AgentTrace:
        validate_raw_trace(data)
        events = []
        for raw_event in data["events"]:
            event = AgentEvent(
                step_index=int(raw_event["step"]),
                timestamp=float(raw_event.get("timestamp", time.time())),
                event_type=raw_event["type"],
                tool_name=raw_event.get("tool"),
                input=raw_event.get("input", {}),
                output=raw_event.get("output", {}),
                success=bool(raw_event["success"]),
                latency_ms=float(raw_event.get("latency_ms", 0.0)),
                metadata=raw_event.get("metadata", {}),
            )
            events.append(event)

        events.sort(key=lambda e: e.step_index)

        return AgentTrace(
            trace_id=str(data.get("trace_id", "unknown")),
            agent_name=str(data.get("agent_name", "unknown")),
            task=str(data.get("task", "")),
            events=events,
        )


class LangSmithAdapter:
    """Handles LangSmith export format.

    LangSmith exports traces as nested runs with run_type, inputs, outputs,
    start_time, end_time, error fields.
    """

    def adapt(self, data: dict) -> AgentTrace:
        runs = data.get("runs", [data])  # single run or list of runs

        events = []
        for i, run in enumerate(runs):
            run_type = run.get("run_type", "chain")

            # Map LangSmith run types to xaudit event types
            if run_type == "tool":
                event_type = "tool_call"
            elif run_type == "retriever":
                event_type = "retrieval"
            elif run_type == "llm":
                event_type = "llm_call"
            else:
                event_type = "tool_call"

            # Compute latency from timestamps
            start = run.get("start_time")
            end = run.get("end_time")
            latency_ms = 0.0
            if start and end:
                try:
                    from datetime import datetime
                    fmt = "%Y-%m-%dT%H:%M:%S.%fZ"
                    latency_ms = (
                        datetime.strptime(end, fmt) - datetime.strptime(start, fmt)
                    ).total_seconds() * 1000
                except Exception:
                    latency_ms = 0.0

            success = run.get("error") is None

            # Truncate large outputs before creating event
            output = run.get("outputs", {})
            if len(str(output)) > 50_000:
                output = {"_truncated": True, "preview": str(output)[:500]}

            event = AgentEvent(
                step_index=i,
                timestamp=time.time(),
                event_type=event_type,
                tool_name=run.get("name"),
                input=run.get("inputs", {}),
                output=output,
                success=success,
                latency_ms=latency_ms,
                metadata={"run_id": run.get("id", "")},
            )
            events.append(event)

        return AgentTrace(
            trace_id=str(data.get("id", "langsmith_trace")),
            agent_name=str(data.get("name", "langsmith_agent")),
            task=str(data.get("inputs", {}).get("input", "")),
            events=events,
        )


class LangfuseAdapter:
    """Handles Langfuse export format.

    Langfuse exports traces with an observations[] array.
    Each observation has type, input, output, startTime, endTime, level fields.
    """

    def adapt(self, data: dict) -> AgentTrace:
        observations = data.get("observations", [])

        events = []
        for i, obs in enumerate(observations):
            obs_type = obs.get("type", "").lower()

            if obs_type in ("tool", "span"):
                event_type = "tool_call"
            elif obs_type == "generation":
                event_type = "llm_call"
            else:
                event_type = "tool_call"

            # Langfuse uses level field for error state
            success = obs.get("level", "DEFAULT") not in ("ERROR", "WARNING")

            # Latency from startTime / endTime
            latency_ms = 0.0
            try:
                from datetime import datetime
                start = obs.get("startTime", "")
                end = obs.get("endTime", "")
                if start and end:
                    fmt = "%Y-%m-%dT%H:%M:%S.%fZ"
                    latency_ms = (
                        datetime.strptime(end, fmt) - datetime.strptime(start, fmt)
                    ).total_seconds() * 1000
            except Exception:
                latency_ms = 0.0

            output = obs.get("output", {}) or {}
            if len(str(output)) > 50_000:
                output = {"_truncated": True, "preview": str(output)[:500]}

            event = AgentEvent(
                step_index=i,
                timestamp=time.time(),
                event_type=event_type,
                tool_name=obs.get("name"),
                input=obs.get("input", {}) or {},
                output=output,
                success=success,
                latency_ms=latency_ms,
                metadata={"observation_id": obs.get("id", "")},
            )
            events.append(event)

        return AgentTrace(
            trace_id=str(data.get("id", "langfuse_trace")),
            agent_name=str(data.get("name", "langfuse_agent")),
            task=str(data.get("input", "")),
            events=events,
        )


def get_adapter(fmt: str):
    adapters = {
        "raw": RawJSONAdapter(),
        "langsmith": LangSmithAdapter(),
        "langfuse": LangfuseAdapter(),
    }
    if fmt not in adapters:
        raise ValueError(f"Unknown format '{fmt}'. Use: raw, langsmith, langfuse")
    return adapters[fmt]
