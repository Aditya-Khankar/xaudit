from tests.conftest import make_event, make_trace
from cognidrift.detectors.anchoring import AnchoringDetector


def test_anchoring_detected():
    """Trace where answer closely matches first retrieval — should detect."""
    first_content = "Transformer attention mechanisms use query key value operations for context-aware representation learning in neural networks"
    later_content = "LSTM recurrent units use gating mechanisms to handle long-range dependencies in sequential data processing tasks"

    events = [
        make_event(0, "retrieval", "arxiv", output={"text": first_content}),
        make_event(1, "tool_call", "web_search"),
        make_event(2, "tool_call", "web_search"),
        make_event(3, "retrieval", "document_reader", output={"text": later_content}),
        make_event(4, "tool_call", "web_search"),
        # LLM output mirrors first retrieval
        make_event(5, "llm_call", None, output={"text": first_content + " applied to language modeling tasks"}),
    ]
    trace = make_trace(events)
    result = AnchoringDetector().detect(trace)
    assert result.detected is True
    assert result.score >= result.threshold


def test_anchoring_not_detected():
    """Trace with balanced retrieval usage — should not detect."""
    events = [
        make_event(0, "retrieval", "arxiv", output={"text": "apple orange banana fruit salad"}),
        make_event(1, "retrieval", "web", output={"text": "car truck vehicle transport engine"}),
        make_event(2, "retrieval", "docs", output={"text": "cloud rain weather temperature humidity"}),
        make_event(3, "llm_call", None, output={"text": "cloud rain weather temperature humidity forecast"}),
    ]
    trace = make_trace(events)
    result = AnchoringDetector().detect(trace)
    assert result.detected is False


def test_anchoring_insufficient_retrievals():
    """Single retrieval — insufficient for anchoring detection."""
    events = [
        make_event(0, "retrieval", "arxiv", output={"text": "some content here"}),
        make_event(1, "llm_call", None, output={"text": "some content here"}),
    ]
    trace = make_trace(events)
    result = AnchoringDetector().detect(trace)
    assert result.detected is False
    assert result.score == 0.0
