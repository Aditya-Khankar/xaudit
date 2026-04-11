from tests.conftest import make_event, make_trace
from cognidrift.detectors.confirmation import ConfirmationDetector


def test_confirmation_detected():
    """Tool diversity narrows from 4 tools to 1 — should detect."""
    events = [
        # First half: diverse tools
        make_event(0, "tool_call", "web_search"),
        make_event(1, "tool_call", "arxiv_search"),
        make_event(2, "tool_call", "citation_finder"),
        make_event(3, "tool_call", "document_reader"),
        make_event(4, "tool_call", "web_search"),
        make_event(5, "tool_call", "arxiv_search"),
        # Second half: only web_search
        make_event(6, "tool_call", "web_search"),
        make_event(7, "tool_call", "web_search"),
        make_event(8, "tool_call", "web_search"),
        make_event(9, "tool_call", "web_search"),
        make_event(10, "tool_call", "web_search"),
        make_event(11, "tool_call", "web_search"),
    ]
    trace = make_trace(events)
    result = ConfirmationDetector().detect(trace)
    assert result.detected is True
    assert result.score >= result.threshold


def test_confirmation_not_detected():
    """Stable tool diversity — should not detect."""
    events = [
        make_event(0, "tool_call", "web_search"),
        make_event(1, "tool_call", "arxiv_search"),
        make_event(2, "tool_call", "document_reader"),
        make_event(3, "tool_call", "calculator"),
        make_event(4, "tool_call", "web_search"),
        make_event(5, "tool_call", "arxiv_search"),
        make_event(6, "tool_call", "document_reader"),
        make_event(7, "tool_call", "calculator"),
        make_event(8, "tool_call", "web_search"),
        make_event(9, "tool_call", "arxiv_search"),
    ]
    trace = make_trace(events)
    result = ConfirmationDetector().detect(trace)
    assert result.detected is False


def test_confirmation_insufficient_events():
    """Too few tool events — insufficient for analysis."""
    events = [
        make_event(0, "tool_call", "web_search"),
        make_event(1, "tool_call", "arxiv_search"),
    ]
    trace = make_trace(events)
    result = ConfirmationDetector().detect(trace)
    assert result.detected is False
    assert result.score == 0.0
