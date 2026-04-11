import json
from tests.conftest import make_event, make_trace
from cognidrift.recorder.format_adapters import RawJSONAdapter, LangSmithAdapter, LangfuseAdapter, get_adapter
from cognidrift.utils.format_detect import detect_format
import pytest


def test_raw_adapter():
    """Raw JSON adapter converts correctly."""
    data = {
        "trace_id": "test_001",
        "agent_name": "test_agent",
        "task": "test task",
        "events": [
            {
                "step": 0,
                "type": "tool_call",
                "tool": "web_search",
                "input": {"q": "test"},
                "output": {"r": "result"},
                "success": True,
                "latency_ms": 100,
            },
            {
                "step": 1,
                "type": "llm_call",
                "tool": None,
                "input": {},
                "output": {"text": "answer"},
                "success": True,
            },
        ],
    }
    adapter = RawJSONAdapter()
    trace = adapter.adapt(data)
    assert trace.trace_id == "test_001"
    assert trace.total_steps == 2
    assert trace.events[0].event_type == "tool_call"
    assert trace.events[1].event_type == "llm_call"


def test_format_detection_raw():
    assert detect_format({"events": []}) == "raw"


def test_format_detection_langsmith():
    assert detect_format({"run_type": "chain"}) == "langsmith"
    assert detect_format({"runs": []}) == "langsmith"


def test_format_detection_langfuse():
    assert detect_format({"observations": []}) == "langfuse"


def test_format_detection_unknown():
    with pytest.raises(ValueError, match="Could not detect"):
        detect_format({"unknown_key": True})


def test_get_adapter_unknown():
    with pytest.raises(ValueError, match="Unknown format"):
        get_adapter("nonexistent")
