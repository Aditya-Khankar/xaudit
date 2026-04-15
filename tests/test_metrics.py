from tests.conftest import make_event, make_trace
from xaudit.metrics.efficiency import compute_efficiency
from xaudit.metrics.exploration_score import compute_exploration_score
from xaudit.metrics.recovery_time import compute_recovery_time


def test_efficiency():
    events = [
        make_event(0, "tool_call", "web_search", success=True),
        make_event(1, "tool_call", "web_search", success=True),
        make_event(2, "tool_call", "web_search", success=False),
        make_event(3, "llm_call", None, success=True),
    ]
    trace = make_trace(events)
    result = compute_efficiency(trace)
    assert result["value"] == 0.5  # 2 successful tool events / 4 total
    assert result["goal_advancing_actions"] == 2


def test_exploration_score():
    events = [
        make_event(0, "tool_call", "web_search"),
        make_event(1, "tool_call", "arxiv_search"),
        make_event(2, "tool_call", "calculator"),
        make_event(3, "tool_call", "web_search"),
    ]
    trace = make_trace(events)
    result = compute_exploration_score(trace)
    assert result["unique_tools_used"] == 3
    assert result["value"] > 0


def test_recovery_time_no_failures():
    events = [make_event(i, success=True) for i in range(5)]
    trace = make_trace(events)
    result = compute_recovery_time(trace)
    assert result["failure_events"] == 0
    assert result["value"] == 0


def test_recovery_time_with_failures():
    events = [
        make_event(0, success=True),
        make_event(1, success=False),
        make_event(2, success=False),
        make_event(3, success=True),  # recovery after 2 steps
    ]
    trace = make_trace(events)
    result = compute_recovery_time(trace)
    assert result["failure_events"] == 2
    assert result["avg_steps_to_recovery"] > 0
