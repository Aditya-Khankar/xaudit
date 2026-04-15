"""Detects action loops — agent repeating identical or near-identical
action sequences via autocorrelation analysis."""

import numpy as np
from xaudit.detectors.base import BaseDetector, DetectorResult
from xaudit.recorder.trace_recorder import AgentTrace


class CyclicRedundancy(BaseDetector):
    name = "loop_detection"
    version = "1.0.0"
    threshold = 0.65

    def detect(self, trace: AgentTrace) -> DetectorResult:
        if trace.total_steps < 6:
            return self._insufficient("Trace too short for loop detection (need 6+ steps).")

        # Encode tool sequence as integer array
        tool_vocab: dict[str, int] = {}
        counter = 0
        sequence = []
        for event in trace.events:
            tool = event.tool_name or event.event_type
            if tool not in tool_vocab:
                tool_vocab[tool] = counter
                counter += 1
            sequence.append(tool_vocab[tool])

        seq = np.array(sequence, dtype=float)
        seq_centered = seq - seq.mean()

        # Constant sequence — all same tool — no variance, no loop signal
        if seq_centered.std() == 0:
            return DetectorResult(
                detector_name=self.name,
                detected=False,
                score=0.0,
                threshold=self.threshold,
                evidence={
                    "reason": "constant sequence (single tool used throughout)",
                    "sequence_length": len(sequence),
                    "unique_tools": 1,
                },
                interpretation=(
                    "Agent used only one tool type throughout. "
                    "Cannot compute autocorrelation on constant sequence."
                ),
                detector_version=self.version,
            )

        # Autocorrelation
        autocorr_full = np.correlate(seq_centered, seq_centered, mode="full")
        autocorr = autocorr_full[len(autocorr_full) // 2:]

        # Normalize by zero-lag value
        zero_lag = autocorr[0]
        if zero_lag == 0:
            return self._insufficient("Zero-lag autocorrelation is zero — cannot normalize.")

        autocorr_norm = autocorr / zero_lag

        # Check lags 2–10 (skip lag 0 = self, lag 1 = trivial adjacency)
        max_lag = min(11, len(autocorr_norm))
        lag_values = np.abs(autocorr_norm[2:max_lag])

        if len(lag_values) == 0:
            return self._insufficient("Sequence too short for lag analysis.")

        max_autocorr = float(np.max(lag_values))
        dominant_lag = int(np.argmax(lag_values)) + 2  # offset for lag 2 start

        score = self._clamp(max_autocorr)

        return DetectorResult(
            detector_name=self.name,
            detected=score >= self.threshold,
            score=score,
            threshold=self.threshold,
            evidence={
                "max_autocorrelation": round(max_autocorr, 3),
                "dominant_lag": dominant_lag,
                "sequence_length": len(sequence),
                "unique_tools": len(tool_vocab),
                "tool_vocabulary": tool_vocab,
            },
            interpretation=(
                f"Peak autocorrelation {max_autocorr:.2f} at lag {dominant_lag}. "
                + (
                    f"Repetitive loop pattern detected — action repeats every ~{dominant_lag} steps."
                    if score >= self.threshold
                    else "No significant action loop detected."
                )
            ),
            detector_version=self.version,
        )
