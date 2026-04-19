from tests.conftest import make_event, make_trace
from xaudit.detectors.cyclic_redundancy import CyclicRedundancy, lz76_complexity

def test_lz76_empty_sequence():
    """Empty input → 0.0"""
    assert lz76_complexity([]) == 0.0

def test_lz76_single_element():
    """Single element → 1.0"""
    assert lz76_complexity(["A"]) == 1.0

def test_lz76_perfectly_repetitive():
    """Same action 20 times → low complexity"""
    result = lz76_complexity(["search"] * 20)
    assert 0.05 < result < 0.35

def test_lz76_maximally_diverse():
    """20 unique actions → high complexity"""
    result = lz76_complexity([f"tool_{i}" for i in range(20)])
    assert 0.75 < result <= 1.0  # Note: Can reach exactly 1.0 due to max cap

def test_lz76_periodic_pattern():
    """Alternating AB pattern → moderate complexity"""
    result = lz76_complexity(["A", "B"] * 10)
    assert 0.2 < result < 0.5

def test_lz76_real_agent_discrimination():
    """Looping agent must have lower complexity than exploring agent"""
    looping = ["search", "search", "search", "calculator", "search", "search"]
    exploring = ["search", "calculator", "wikipedia", "datetime", "python", "search"]
    assert lz76_complexity(looping) < lz76_complexity(exploring)

def test_cyclic_redundancy_repetitive_detected():
    """Full detector: repetitive trace → detected=True"""
    # Create trace with 20 events all using "search"
    events = [make_event(i, "tool_call", "search") for i in range(20)]
    trace = make_trace(events)
    result = CyclicRedundancy().detect(trace)
    assert result.detected is True
    assert result.score > 0.65

def test_cyclic_redundancy_diverse_not_detected():
    """Full detector: diverse trace → detected=False"""
    # Create trace with 20 events using 20 different tools
    events = [make_event(i, "tool_call", f"tool_{i}") for i in range(20)]
    trace = make_trace(events)
    result = CyclicRedundancy().detect(trace)
    assert result.detected is False
    assert result.score < 0.35

def test_cyclic_redundancy_empty_trace():
    """Empty trace → safe defaults"""
    trace = make_trace([])
    result = CyclicRedundancy().detect(trace)
    assert result.score == 0.0
    assert result.detected is False

def test_loop_not_detected():
    """Varied tool usage — no loop. (Old ported test)"""
    events = [
        make_event(0, "tool_call", "web_search"),
        make_event(1, "tool_call", "arxiv_search"),
        make_event(2, "tool_call", "calculator"),
        make_event(3, "tool_call", "document_reader"),
        make_event(4, "tool_call", "financial_data"),
        make_event(5, "tool_call", "citation_finder"),
        make_event(6, "tool_call", "web_search"),
        make_event(7, "tool_call", "calculator"),
    ]
    trace = make_trace(events)
    result = CyclicRedundancy().detect(trace)
    assert result.detected is False
