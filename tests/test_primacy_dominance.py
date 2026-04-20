from tests.conftest import make_event, make_trace
from xaudit.detectors.primacy_dominance import PrimacyDominanceDetector


def test_primacy_dominance_detected():
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
    result = PrimacyDominanceDetector().detect(trace)
    assert result.detected is True
    assert result.score >= result.threshold


def test_primacy_dominance_not_detected():
    """Trace with balanced retrieval usage — should not detect."""
    events = [
        make_event(0, "retrieval", "arxiv", output={"text": "apple orange banana fruit salad"}),
        make_event(1, "retrieval", "web", output={"text": "car truck vehicle transport engine"}),
        make_event(2, "retrieval", "docs", output={"text": "cloud rain weather temperature humidity"}),
        make_event(3, "llm_call", None, output={"text": "cloud rain weather temperature humidity forecast"}),
    ]
    trace = make_trace(events)
    result = PrimacyDominanceDetector().detect(trace)
    assert result.detected is False


def test_primacy_dominance_insufficient_retrievals():
    """Single retrieval — insufficient for primacy_dominance detection."""
    events = [
        make_event(0, "retrieval", "arxiv", output={"text": "some content here"}),
        make_event(1, "llm_call", None, output={"text": "some content here"}),
    ]
    trace = make_trace(events)
    result = PrimacyDominanceDetector().detect(trace)
    assert result.detected is False
    assert result.score == 0.0


def test_extreme_primacy():
    """All weight on first retrieval → high score"""
    first_content = "Transformer attention mechanisms use query key value operations for context-aware representation learning in neural networks"
    later_content1 = "LSTM recurrent units use gating mechanisms to handle long-range dependencies in sequential data processing tasks"
    later_content2 = "CNNs use convolutional layers to extract spatial features from image and video datasets"

    events = [
        make_event(0, "retrieval", "arxiv", output={"text": first_content}),
        make_event(1, "retrieval", "document_reader", output={"text": later_content1}),
        make_event(2, "retrieval", "document_reader", output={"text": later_content2}),
        make_event(3, "llm_call", None, output={"text": first_content + " applied to language modeling tasks"}),
    ]
    trace = make_trace(events)
    result = PrimacyDominanceDetector().detect(trace)
    assert result.score > 0.5
    assert result.detected is True


def test_uniform_usage():
    """Equal usage of all retrievals → score near zero"""
    content1 = "Transformer attention mechanisms use query key value operations."
    content2 = "LSTM recurrent units use gating mechanisms."
    content3 = "CNNs use convolutional layers."

    events = [
        make_event(0, "retrieval", "arxiv", output={"text": content1}),
        make_event(1, "retrieval", "arxiv", output={"text": content2}),
        make_event(2, "retrieval", "arxiv", output={"text": content3}),
        make_event(3, "llm_call", None, output={"text": content1 + " " + content2 + " " + content3}),
    ]
    trace = make_trace(events)
    result = PrimacyDominanceDetector().detect(trace)
    assert result.score < 0.15
    assert result.detected is False


def test_moderate_primacy():
    """70% from first retrieval → moderate score"""
    first_content = "Transformer attention mechanisms use query key value operations for context-aware representation learning."
    later_content1 = "LSTM recurrent units use gating mechanisms."
    later_content2 = "CNNs use convolutional layers."

    events = [
        make_event(0, "retrieval", "arxiv", output={"text": first_content}),
        make_event(1, "retrieval", "document_reader", output={"text": later_content1}),
        make_event(2, "retrieval", "document_reader", output={"text": later_content2}),
        make_event(3, "llm_call", None, output={"text": first_content + " And some mention of LSTM recurrent units."}),
    ]
    trace = make_trace(events)
    result = PrimacyDominanceDetector().detect(trace)
    assert 0.2 < result.score < 0.6


def test_no_retrievals():
    """Empty trace → safe default"""
    events = [
        make_event(0, "llm_call", None, output={"text": "some content here"}),
    ]
    trace = make_trace(events)
    result = PrimacyDominanceDetector().detect(trace)
    assert result.score == 0.0
    assert result.detected is False


def test_single_retrieval():
    """One retrieval → cannot detect primacy"""
    events = [
        make_event(0, "retrieval", "arxiv", output={"text": "some content here"}),
        make_event(1, "llm_call", None, output={"text": "some content here"}),
    ]
    trace = make_trace(events)
    result = PrimacyDominanceDetector().detect(trace)
    assert result.score == 0.0
    assert result.detected is False
