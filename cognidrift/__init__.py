"""cognidrift — behavioral auditing for autonomous agents.

Quick start:
    from cognidrift import analyze_trace

    with open("trace.json") as f:
        trace_dict = json.load(f)

    report = analyze_trace(trace_dict)
    print(report["overall_rationality_score"])
"""

from cognidrift.recorder.format_adapters import get_adapter, RawJSONAdapter
from cognidrift.utils.format_detect import detect_format
from cognidrift.report.builder import build_report
import json
import tempfile
import os


def analyze_trace(trace_dict: dict, fmt: str = "auto") -> dict:
    """Analyze an agent trace dict and return a behavior report.

    Args:
        trace_dict: Agent trace as a Python dict (loaded from JSON).
        fmt: Trace format — "auto", "raw", "langsmith", or "langfuse".
             Defaults to auto-detection.

    Returns:
        dict: Complete behavior report with detector scores, metrics,
              and overall rationality score.

    Example:
        import json
        from cognidrift import analyze_trace

        with open("my_trace.json") as f:
            trace = json.load(f)

        report = analyze_trace(trace)
        print(f"Rationality: {report['overall_rationality_score']}")
        print(f"Biases: {report['biases_detected']}")
    """
    resolved_fmt = fmt if fmt != "auto" else detect_format(trace_dict)
    adapter = get_adapter(resolved_fmt)
    trace = adapter.adapt(trace_dict)

    with tempfile.TemporaryDirectory() as tmp:
        report = build_report(trace, tmp)

    return report


__version__ = "0.1.0"
__all__ = ["analyze_trace"]
