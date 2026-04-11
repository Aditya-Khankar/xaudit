"""cognidrift — Behavioral auditing for autonomous agents.

Public API:
    from cognidrift import analyze_trace
    report = analyze_trace(trace_dict)
"""

from cognidrift.recorder.format_adapters import get_adapter
from cognidrift.report.builder import build_report
from cognidrift.utils.format_detect import detect_format

__version__ = "0.1.0"


def analyze_trace(trace_dict: dict, output_dir: str = "./") -> dict:
    """Analyze a trace dict for behavioral bias patterns.

    Accepts traces in raw, LangSmith, or Langfuse format.
    Auto-detects the format and runs all 5 behavioral detectors
    plus 3 metrics.

    Args:
        trace_dict: Trace data as a Python dict.
        output_dir: Directory to write behavior_report.json.

    Returns:
        Report dict with detector results, metrics, and
        overall rationality score.

    Example:
        from cognidrift import analyze_trace
        report = analyze_trace(langsmith_export)
        if report["overall_rationality_score"] < 0.5:
            print("Agent reasoning quality is low")
    """
    fmt = detect_format(trace_dict)
    adapter = get_adapter(fmt)
    trace = adapter.adapt(trace_dict)
    return build_report(trace, output_dir)
