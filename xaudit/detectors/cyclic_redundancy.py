"""
Cyclic Redundancy Detector

Detects repeated action patterns using Lempel-Ziv complexity from
algorithmic information theory. Low complexity indicates compressible
(repetitive) behavior; high complexity indicates diverse exploration.

Method: LZ76 substring counting with n/log₂(n) normalization.
        Complexity is inverted to produce a redundancy score:
        high redundancy = repetitive loops detected.
Field:  Algorithmic information theory, Kolmogorov complexity.
Reference: Ziv & Lempel (1976), "On the Complexity of Finite Sequences"
"""

import math
from xaudit.detectors.base import BaseDetector, DetectorResult
from xaudit.recorder.trace_recorder import AgentTrace


def lz76_complexity(sequence: list[str]) -> float:
    """
    Compute normalized Lempel-Ziv complexity (LZ76).

    Measures algorithmic compressibility of an action sequence.
    Low complexity = repetitive/looping behavior.
    High complexity = diverse/exploratory behavior.

    Algorithm (LZ76):
        1. Convert sequence to a single string with unique separator (handled intrinsically if strings)
        2. Initialize: complexity_count = 1, start scanning from index 1
        3. At each position, find the longest substring starting here
           that has already appeared earlier in the sequence
        4. When a new (unseen) substring is found, increment complexity_count
        5. Advance past the matched portion + 1 new symbol
        6. Continue until end of sequence

    Normalization:
        For sequence length n, theoretical max complexity ≈ n / log₂(n)
        Return: complexity_count / (n / log₂(n))
        Capped at 1.0

    Args:
        sequence: list of action/tool names (strings)

    Returns:
        float: 0.0 (empty) to ~1.0 (maximally complex)
    """
    if len(sequence) == 0:
        return 0.0
    if len(sequence) == 1:
        return 1.0

    n = len(sequence)
    i = 1
    complexity_count = 1

    while i < n:
        # Find longest match of sequence[i:i+length] in sequence
        max_match_len = 0
        for length in range(1, n - i + 1):
            # Current substring to match
            current = sequence[i:i + length]
            # Search in all positions before i
            found = False
            for j in range(0, i):
                if sequence[j:j + length] == current:
                    found = True
                    break
            if found:
                max_match_len = length
            else:
                break

        i += max_match_len + 1
        if i <= n:
            complexity_count += 1

    # Normalize
    if n <= 1:
        return 1.0
    theoretical_max = n / math.log2(n)
    normalized = complexity_count / theoretical_max
    return min(normalized, 1.0)


class CyclicRedundancy(BaseDetector):
    name = "cyclic_redundancy"
    version = "1.0.0"
    threshold = 0.65

    def detect(self, trace: AgentTrace) -> DetectorResult:
        # Extract ordered list of tools
        tool_events = [
            e for e in trace.events
            if e.event_type in ("tool_call", "retrieval")
        ]

        if len(tool_events) == 0:
            return self._insufficient("No tool events found.")

        if len(tool_events) == 1:
            return self._insufficient("Single tool event — cannot detect pattern with 1 event.")

        sequence = [(e.tool_name or e.event_type) for e in tool_events]

        # Compute complexity using LZ76
        lz_complexity = lz76_complexity(sequence)
        
        # Redundancy is the inverse of complexity
        redundancy_score = 1.0 - lz_complexity
        
        detected = redundancy_score > self.threshold

        unique_tools = len(set(sequence))

        interpretation_suffix = (
            "Repetitive loop pattern detected."
            if detected
            else "Diverse exploratory pattern, no looping behavior."
        )

        complexity_phrases = int(lz_complexity * (len(sequence) / math.log2(len(sequence)))) if len(sequence) > 1 else 1

        return DetectorResult(
            detector_name=self.name,
            detected=detected,
            score=redundancy_score,
            threshold=self.threshold,
            evidence={
                "lz_complexity": float(round(lz_complexity, 4)),
                "sequence_length": len(sequence),
                "unique_actions": unique_tools,
                "complexity_phrases": complexity_phrases
            },
            interpretation=f"Redundancy score: {redundancy_score:.3f}. {interpretation_suffix}",
            detector_version=self.version,
        )
