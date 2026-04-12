from tests.conftest import make_event, make_trace
from cognidrift.detectors.degradation import DegradationDetector


def test_degradation_detected():
    """Efficiency drops from 100% to 0% midway — should detect degradation."""
    events = (
        # First 10 steps: all succeed
        [make_event(i, "tool_call", "web_search", success=True) for i in range(10)]
        # Next 10 steps: all fail
        + [make_event(i, "tool_call", "web_search", success=False) for i in range(10, 20)]
    )
    trace = make_trace(events)
    result = DegradationDetector().detect(trace)
    assert result.detected is True
    assert result.score >= result.threshold


def test_degradation_not_detected():
    """Consistent success throughout — no degradation."""
    events = [make_event(i, "tool_call", "web_search", success=True) for i in range(20)]
    trace = make_trace(events)
    result = DegradationDetector().detect(trace)
    assert result.detected is False


def test_degradation_too_short():
    """Trace too short for analysis."""
    events = [make_event(i, "tool_call", "web_search") for i in range(4)]
    trace = make_trace(events)
    result = DegradationDetector().detect(trace)
    assert result.detected is False
    assert result.score == 0.0


def test_degradation_complete_failure():
    """All events fail — complete degradation."""
    events = [make_event(i, "tool_call", "web_search", success=False) for i in range(20)]
    trace = make_trace(events)
    result = DegradationDetector().detect(trace)
    assert result.detected is True
    assert result.score == 1.0
