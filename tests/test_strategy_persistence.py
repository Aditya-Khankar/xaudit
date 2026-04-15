from tests.conftest import make_event, make_trace
from xaudit.detectors.strategy_persistence import StrategyPersistenceDetector


def test_strategy_persistence_detected():
    """Agent fails at step 2 but doesn't change strategy until step 20 — sunk cost."""
    events = [
        make_event(0, "tool_call", "code_executor"),
        make_event(1, "tool_call", "code_executor"),
        make_event(2, "tool_call", "code_executor", success=False),
        make_event(3, "tool_call", "code_executor"),
        make_event(4, "tool_call", "code_executor", success=False),
        make_event(5, "tool_call", "code_executor"),
        make_event(6, "tool_call", "code_executor", success=False),
        make_event(7, "tool_call", "code_executor"),
        make_event(8, "tool_call", "code_executor"),
        make_event(9, "tool_call", "code_executor", success=False),
        make_event(10, "tool_call", "code_executor"),
        make_event(11, "tool_call", "code_executor"),
        make_event(12, "tool_call", "code_executor"),
        make_event(13, "tool_call", "code_executor"),
        make_event(14, "tool_call", "code_executor"),
        make_event(15, "tool_call", "code_executor"),
        make_event(16, "tool_call", "code_executor"),
        make_event(17, "tool_call", "code_executor"),
        make_event(18, "tool_call", "code_executor"),
        make_event(19, "tool_call", "code_executor"),
        # Finally pivots to a new tool
        make_event(20, "tool_call", "log_analyzer"),
        make_event(21, "llm_call", None),
    ]
    trace = make_trace(events)
    result = StrategyPersistenceDetector().detect(trace)
    assert result.detected is True
    assert result.score >= result.threshold


def test_strategy_persistence_not_detected():
    """Agent pivots quickly after failure — no sunk cost."""
    events = [
        make_event(0, "tool_call", "code_executor"),
        make_event(1, "tool_call", "code_executor"),
        make_event(2, "tool_call", "code_executor", success=False),
        # Quick pivot
        make_event(3, "tool_call", "log_analyzer"),
        make_event(4, "tool_call", "documentation_search"),
        make_event(5, "llm_call", None),
        make_event(6, "llm_call", None),
    ]
    trace = make_trace(events)
    result = StrategyPersistenceDetector().detect(trace)
    assert result.detected is False


def test_strategy_persistence_no_failures():
    """No failures in trace — sunk cost not applicable."""
    events = [
        make_event(0, "tool_call", "web_search"),
        make_event(1, "tool_call", "document_reader"),
        make_event(2, "tool_call", "web_search"),
        make_event(3, "llm_call", None),
    ]
    trace = make_trace(events)
    result = StrategyPersistenceDetector().detect(trace)
    assert result.detected is False
    assert result.score == 0.0
