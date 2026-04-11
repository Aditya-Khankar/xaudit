from tests.conftest import make_event, make_trace
from cognidrift.detectors.loop_detector import LoopDetector


def test_loop_detected():
    """Clear repeating A-B-A-B pattern — should detect loop."""
    events = [
        make_event(0, "tool_call", "web_search"),
        make_event(1, "tool_call", "document_reader"),
        make_event(2, "tool_call", "web_search"),
        make_event(3, "tool_call", "document_reader"),
        make_event(4, "tool_call", "web_search"),
        make_event(5, "tool_call", "document_reader"),
        make_event(6, "tool_call", "web_search"),
        make_event(7, "tool_call", "document_reader"),
        make_event(8, "tool_call", "web_search"),
        make_event(9, "tool_call", "document_reader"),
        make_event(10, "tool_call", "web_search"),
        make_event(11, "tool_call", "document_reader"),
    ]
    trace = make_trace(events)
    result = LoopDetector().detect(trace)
    assert result.detected is True
    assert result.score >= result.threshold
    assert result.evidence["dominant_lag"] == 2


def test_loop_not_detected():
    """Varied tool usage — no loop."""
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
    result = LoopDetector().detect(trace)
    assert result.detected is False


def test_loop_constant_sequence():
    """All same tool — constant sequence, not a detectable loop."""
    events = [
        make_event(i, "tool_call", "web_search")
        for i in range(10)
    ]
    trace = make_trace(events)
    result = LoopDetector().detect(trace)
    assert result.detected is False
    assert result.score == 0.0


def test_loop_too_short():
    """Trace shorter than 6 events — insufficient."""
    events = [
        make_event(0, "tool_call", "web_search"),
        make_event(1, "tool_call", "document_reader"),
    ]
    trace = make_trace(events)
    result = LoopDetector().detect(trace)
    assert result.detected is False
    assert result.score == 0.0
