"""Assembles behavior_report.json from all detector and metric results."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from xaudit.recorder.trace_recorder import AgentTrace
from xaudit.detectors.primacy_dominance import PrimacyDominanceDetector
from xaudit.detectors.query_entropy_collapse import QueryEntropyCollapseDetector
from xaudit.detectors.strategy_persistence import StrategyPersistenceDetector
from xaudit.detectors.cyclic_redundancy import CyclicRedundancyDetector
from xaudit.detectors.context_decay import ContextDecayDetector
from xaudit.metrics.efficiency import compute_efficiency
from xaudit.metrics.exploration_score import compute_exploration_score
from xaudit.metrics.recovery_time import compute_recovery_time
from xaudit.utils.paths import safe_output_path, ensure_output_dir

logger = logging.getLogger("xaudit")

DETECTORS = [
    PrimacyDominanceDetector(),
    QueryEntropyCollapseDetector(),
    StrategyPersistenceDetector(),
    CyclicRedundancyDetector(),
    ContextDecayDetector(),
]


MAX_STEPS = 100


def build_report(trace: AgentTrace, output_dir: str) -> dict:
    """Run all detectors and metrics, assemble report, write to disk."""
    from xaudit.themes import load_config
    config = load_config()
    custom_thresholds = config.get("thresholds", {})

    # Run detectors
    if trace.total_steps > MAX_STEPS:
        logger.warning(
            f"Trace {trace.trace_id} exceeds {MAX_STEPS} steps ({trace.total_steps}). "
            "Analysis may take longer due to complex behavioral patterns."
        )

    detector_results = {}
    detected_scores = []
    patterns_detected = []

    for detector in DETECTORS:
        # Apply custom threshold if set
        if detector.name in custom_thresholds:
            detector.threshold = custom_thresholds[detector.name]
            logger.info(f"Using custom threshold for {detector.name}: {detector.threshold}")

        logger.info(f"Running detector: {detector.name} v{detector.version}")
        result = detector.detect(trace)
        detector_results[result.detector_name] = result.to_dict()
        if result.detected:
            patterns_detected.append(result.detector_name)
            detected_scores.append(result.score)
            logger.info(f"  {detector.name}: DETECTED (score={result.score:.3f})")
        else:
            logger.info(f"  {detector.name}: not detected (score={result.score:.3f})")

    # Overall rationality: combines behavioral pattern penalty with functional success
    # Formula: rationality = (1.0 - pattern_penalty) * success_rate
    # This ensures agents that fail every step score near 0 regardless of patterns
    n_detectors = len(DETECTORS)
    if detected_scores:
        penalty = sum(detected_scores) / n_detectors
    else:
        penalty = 0.0

    # Success rate: fraction of events that succeeded
    total_events = len(trace.events)
    if total_events > 0:
        success_count = sum(1 for e in trace.events if e.success)
        success_rate = success_count / total_events
    else:
        success_rate = 1.0  # empty trace — no failures

    overall_rationality = round(max(0.0, min(1.0, (1.0 - penalty) * success_rate)), 4)

    # Compute metrics
    logger.info("Computing metrics...")
    metrics = {
        "efficiency": compute_efficiency(trace),
        "exploration_score": compute_exploration_score(trace),
        "recovery_time": compute_recovery_time(trace),
    }

    report = {
        "trace_id": trace.trace_id,
        "agent_name": trace.agent_name,
        "task": trace.task,
        "total_steps": trace.total_steps,
        "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
        "overall_rationality_score": overall_rationality,
        "patterns_detected": patterns_detected,
        "detectors": detector_results,
        "metrics": metrics,
    }

    # Write report
    abs_output = ensure_output_dir(output_dir)
    report_path = safe_output_path(abs_output, "behavior_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"Report written to {report_path}")

    return report
