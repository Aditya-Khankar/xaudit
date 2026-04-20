"""Shared fixtures for all tests."""
import pytest
from xaudit.recorder.trace_recorder import AgentEvent, AgentTrace


def make_event(step: int, event_type="tool_call", tool="web_search",
               success=True, output=None) -> AgentEvent:
    return AgentEvent(
        step_index=step,
        timestamp=float(step),
        event_type=event_type,
        tool_name=tool,
        input={"query": f"query_{step}"},
        output=output or {"results": [f"result_{step}"]},
        success=success,
        latency_ms=100.0,
    )


def make_trace(events: list[AgentEvent], task="test task") -> AgentTrace:
    return AgentTrace(
        trace_id="test_trace",
        agent_name="test_agent",
        task=task,
        events=events,
    )
