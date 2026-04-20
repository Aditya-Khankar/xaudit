from tests.conftest import make_event, make_trace
from xaudit.detectors.query_entropy_collapse import QueryEntropyCollapseDetector


def test_query_entropy_collapse_detected():
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
    result = QueryEntropyCollapseDetector().detect(trace)
    assert result.detected is True
    assert result.score >= result.threshold


def test_query_entropy_collapse_not_detected():
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
    result = QueryEntropyCollapseDetector().detect(trace)
    assert result.detected is False


def test_query_entropy_collapse_insufficient_events():
    """Too few tool events — insufficient for analysis."""
    events = [
        make_event(0, "tool_call", "web_search"),
        make_event(1, "tool_call", "arxiv_search"),
    ]
    trace = make_trace(events)
    result = QueryEntropyCollapseDetector().detect(trace)
    assert result.detected is False
    assert result.score == 0.0


def test_renyi_entropy_uniform_distribution():
    """4 different tools used equally → entropy close to log₂(4) = 2.0"""
    from xaudit.detectors.query_entropy_collapse import renyi_entropy_alpha2
    result = renyi_entropy_alpha2({"search": 5, "calc": 5, "wiki": 5, "python": 5})
    assert abs(result - 2.0) < 0.01

def test_renyi_entropy_single_tool():
    """One tool dominates completely → entropy = 0.0"""
    from xaudit.detectors.query_entropy_collapse import renyi_entropy_alpha2
    result = renyi_entropy_alpha2({"search": 20})
    assert result == 0.0

def test_renyi_entropy_two_tools_unequal():
    """Two tools, one dominant → entropy between 0 and 1"""
    from xaudit.detectors.query_entropy_collapse import renyi_entropy_alpha2
    result = renyi_entropy_alpha2({"search": 15, "calc": 5})
    assert 0.5 < result < 1.0

def test_renyi_entropy_empty():
    """No tools → entropy = 0.0"""
    from xaudit.detectors.query_entropy_collapse import renyi_entropy_alpha2
    result = renyi_entropy_alpha2({})
    assert result == 0.0

def test_gradual_collapse_detected():
    """Diverse tools early → single tool late → collapse detected"""
    # Build trace: first 10 steps use 4 different tools
    # Last 10 steps use only "search"
    events = [
        make_event(0, "tool_call", "search"),
        make_event(1, "tool_call", "calc"),
        make_event(2, "tool_call", "wiki"),
        make_event(3, "tool_call", "python"),
        make_event(4, "tool_call", "search"),
        make_event(5, "tool_call", "calc"),
        make_event(6, "tool_call", "wiki"),
        make_event(7, "tool_call", "python"),
        make_event(8, "tool_call", "search"),
        make_event(9, "tool_call", "calc"),
        # Next 10 steps only search
        make_event(10, "tool_call", "search"),
        make_event(11, "tool_call", "search"),
        make_event(12, "tool_call", "search"),
        make_event(13, "tool_call", "search"),
        make_event(14, "tool_call", "search"),
        make_event(15, "tool_call", "search"),
        make_event(16, "tool_call", "search"),
        make_event(17, "tool_call", "search"),
        make_event(18, "tool_call", "search"),
        make_event(19, "tool_call", "search"),
    ]
    trace = make_trace(events)
    result = QueryEntropyCollapseDetector().detect(trace)
    assert result.detected is True
    assert result.score > 0.4

def test_expanding_exploration_not_detected():
    """Single tool early → diverse tools late → NOT collapse"""
    # Build trace: first 10 steps only "search"
    # Last 10 steps use 4 different tools
    events = [
        make_event(i, "tool_call", "search") for i in range(10)
    ] + [
        make_event(10, "tool_call", "search"),
        make_event(11, "tool_call", "calc"),
        make_event(12, "tool_call", "wiki"),
        make_event(13, "tool_call", "python"),
        make_event(14, "tool_call", "search"),
        make_event(15, "tool_call", "calc"),
        make_event(16, "tool_call", "wiki"),
        make_event(17, "tool_call", "python"),
        make_event(18, "tool_call", "search"),
        make_event(19, "tool_call", "calc"),
    ]
    trace = make_trace(events)
    result = QueryEntropyCollapseDetector().detect(trace)
    assert result.detected is False

def test_stable_diversity_not_detected():
    """Consistent tool diversity throughout → no collapse"""
    # Build trace: 20 steps cycling through 4 tools evenly
    events = []
    tools = ["search", "calc", "wiki", "python"]
    for i in range(20):
        events.append(make_event(i, "tool_call", tools[i % 4]))
    trace = make_trace(events)
    result = QueryEntropyCollapseDetector().detect(trace)
    assert result.detected is False

def test_empty_trace_safe():
    """Empty trace → safe defaults"""
    trace = make_trace([])
    result = QueryEntropyCollapseDetector().detect(trace)
    assert result.score == 0.0
    assert result.detected is False
