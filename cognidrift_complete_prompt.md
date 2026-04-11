# cognidrift — Complete Antigravity Build Prompt

Paste this entire document into Antigravity. Build everything in sequence.
Do not skip sections. Do not add features not listed here.

---

## OVERVIEW

Build a Python CLI tool called **cognidrift** — behavioral auditing for autonomous AI agents.

**What it does:** Takes an agent trace (LangSmith / Langfuse / raw JSON), runs 5 behavioral detectors (anchoring bias, confirmation bias, sunk cost fallacy, action loops, performance degradation), computes 3 metrics, and outputs a structured JSON report + annotated PNG timeline.

**Key architectural decision:** The `analyze` command requires zero API keys. All detectors are pure Python math (TF-IDF, autocorrelation, CUSUM). The `generate` command requires a `GEMINI_API_KEY` env variable to generate synthetic traces. These two commands are completely independent.

**CLI entry point:** `cognidrift`
**License:** MIT — Aditya Khankar
**Target Python:** 3.11+

---

## PART 1: PROJECT STRUCTURE

Create this exact directory structure:

```
cognidrift/
├── cognidrift/
│   ├── __init__.py
│   ├── cli.py
│   ├── recorder/
│   │   ├── __init__.py
│   │   ├── trace_recorder.py
│   │   └── format_adapters.py
│   ├── detectors/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── anchoring.py
│   │   ├── confirmation.py
│   │   ├── sunk_cost.py
│   │   ├── loop_detector.py
│   │   └── degradation.py
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── efficiency.py
│   │   ├── exploration_score.py
│   │   └── recovery_time.py
│   ├── report/
│   │   ├── __init__.py
│   │   ├── builder.py
│   │   └── visualize.py
│   └── utils/
│       ├── __init__.py
│       ├── format_detect.py
│       ├── validators.py
│       └── paths.py
├── runs/
│   ├── generate_runs.py
│   ├── bias_scenarios.py
│   └── sample_runs/
│       ├── run_anchoring.json
│       ├── run_confirmation.json
│       ├── run_sunk_cost.json
│       ├── run_loop.json
│       └── run_clean.json
├── report/
│   └── behavior_report.json
├── tests/
│   ├── conftest.py
│   ├── test_adapters.py
│   ├── test_anchoring.py
│   ├── test_confirmation.py
│   ├── test_sunk_cost.py
│   ├── test_loop_detector.py
│   ├── test_degradation.py
│   └── test_metrics.py
├── .github/
│   └── workflows/
│       └── cognidrift_report.yml
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── README.md
├── LICENSE
├── .env.example
└── .gitignore
```

---

## PART 2: CORE DATA MODEL

### cognidrift/recorder/trace_recorder.py

```python
from dataclasses import dataclass, field
from typing import Literal

EventType = Literal["tool_call", "llm_call", "retrieval", "error"]


@dataclass
class AgentEvent:
    step_index: int
    timestamp: float
    event_type: EventType
    tool_name: str | None
    input: dict
    output: dict
    success: bool
    latency_ms: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class AgentTrace:
    trace_id: str
    agent_name: str
    task: str
    events: list[AgentEvent]

    @property
    def total_steps(self) -> int:
        return len(self.events)

    @property
    def tool_calls(self) -> list[AgentEvent]:
        return [e for e in self.events if e.event_type == "tool_call"]

    @property
    def failures(self) -> list[AgentEvent]:
        return [e for e in self.events if not e.success]

    @property
    def retrievals(self) -> list[AgentEvent]:
        return [e for e in self.events if e.event_type == "retrieval"]

    @property
    def llm_calls(self) -> list[AgentEvent]:
        return [e for e in self.events if e.event_type == "llm_call"]
```

---

## PART 3: SECURITY UTILITIES

### cognidrift/utils/validators.py

```python
"""Input validation — runs before any processing. Prevents malformed or
oversized traces from reaching detectors."""

MAX_EVENTS = 500
MAX_STRING_LENGTH = 50_000
MAX_TRACE_SIZE_BYTES = 10 * 1024 * 1024  # 10MB


class TraceValidationError(ValueError):
    pass


def validate_raw_trace(data: dict) -> None:
    """Validate a raw trace dict before adapter processing."""
    if not isinstance(data, dict):
        raise TraceValidationError("Trace must be a JSON object")

    events = data.get("events", [])

    if not isinstance(events, list):
        raise TraceValidationError("'events' must be a list")

    if len(events) > MAX_EVENTS:
        raise TraceValidationError(
            f"Trace has {len(events)} events — limit is {MAX_EVENTS}. "
            "Split into smaller traces for analysis."
        )

    for i, event in enumerate(events):
        if not isinstance(event, dict):
            raise TraceValidationError(f"Event at index {i} must be an object")

        # Prevent memory exhaustion from huge output fields
        output_str = str(event.get("output", ""))
        if len(output_str) > MAX_STRING_LENGTH:
            raise TraceValidationError(
                f"Event {i} output exceeds {MAX_STRING_LENGTH} character limit. "
                "Truncate large outputs before analysis."
            )

        input_str = str(event.get("input", ""))
        if len(input_str) > MAX_STRING_LENGTH:
            raise TraceValidationError(
                f"Event {i} input exceeds {MAX_STRING_LENGTH} character limit."
            )

        # Required fields
        if "step" not in event:
            raise TraceValidationError(f"Event {i} missing required field 'step'")
        if "type" not in event:
            raise TraceValidationError(f"Event {i} missing required field 'type'")
        if "success" not in event:
            raise TraceValidationError(f"Event {i} missing required field 'success'")

        # Valid event types
        valid_types = {"tool_call", "llm_call", "retrieval", "error"}
        if event["type"] not in valid_types:
            raise TraceValidationError(
                f"Event {i} has invalid type '{event['type']}'. "
                f"Must be one of: {valid_types}"
            )
```

### cognidrift/utils/paths.py

```python
"""Safe path handling — prevents directory traversal attacks on --output flag."""

import os


class PathTraversalError(ValueError):
    pass


def safe_output_path(output_dir: str, filename: str) -> str:
    """Resolve output path, raising if it escapes the output directory."""
    output_dir = os.path.abspath(output_dir)
    resolved = os.path.realpath(os.path.join(output_dir, filename))
    if not resolved.startswith(output_dir):
        raise PathTraversalError(
            f"Output path '{filename}' would escape output directory. "
            "Use a simple filename without path separators."
        )
    return resolved


def ensure_output_dir(output_dir: str) -> str:
    """Create output directory if it doesn't exist. Returns absolute path."""
    abs_path = os.path.abspath(output_dir)
    os.makedirs(abs_path, exist_ok=True)
    return abs_path
```

### cognidrift/utils/format_detect.py

```python
def detect_format(trace_dict: dict) -> str:
    """Auto-detect trace format from structure fingerprints."""
    if not isinstance(trace_dict, dict):
        raise ValueError("Trace must be a JSON object")
    if "run_type" in trace_dict or "runs" in trace_dict:
        return "langsmith"
    if "observations" in trace_dict:
        return "langfuse"
    if "events" in trace_dict:
        return "raw"
    raise ValueError(
        "Could not detect trace format. "
        "Use --format langsmith | langfuse | raw to specify explicitly."
    )
```

---

## PART 4: FORMAT ADAPTERS

### cognidrift/recorder/format_adapters.py

```python
"""Format adapters translate native trace formats into AgentTrace.

Rules:
- Adapters are dumb translators. No semantic interpretation here.
- All validation happens before adapters are called (validators.py).
- If a field is missing, use a safe default. Never crash on missing optional fields.
- Semantic decisions (what counts as failure, what counts as retrieval)
  belong in detectors, not here.
"""

import time
from cognidrift.recorder.trace_recorder import AgentEvent, AgentTrace
from cognidrift.utils.validators import validate_raw_trace


class RawJSONAdapter:
    """Handles the cognidrift native raw JSON format.

    Required schema:
    {
        "trace_id": "string",
        "agent_name": "string (optional)",
        "task": "string (optional)",
        "events": [
            {
                "step": 0,
                "type": "tool_call | llm_call | retrieval | error",
                "tool": "string or null",
                "input": {},
                "output": {},
                "success": true,
                "latency_ms": 342  (optional)
            }
        ]
    }
    """

    def adapt(self, data: dict) -> AgentTrace:
        validate_raw_trace(data)
        events = []
        for raw_event in data["events"]:
            event = AgentEvent(
                step_index=int(raw_event["step"]),
                timestamp=float(raw_event.get("timestamp", time.time())),
                event_type=raw_event["type"],
                tool_name=raw_event.get("tool"),
                input=raw_event.get("input", {}),
                output=raw_event.get("output", {}),
                success=bool(raw_event["success"]),
                latency_ms=float(raw_event.get("latency_ms", 0.0)),
                metadata=raw_event.get("metadata", {}),
            )
            events.append(event)

        events.sort(key=lambda e: e.step_index)

        return AgentTrace(
            trace_id=str(data.get("trace_id", "unknown")),
            agent_name=str(data.get("agent_name", "unknown")),
            task=str(data.get("task", "")),
            events=events,
        )


class LangSmithAdapter:
    """Handles LangSmith export format.

    LangSmith exports traces as nested runs with run_type, inputs, outputs,
    start_time, end_time, error fields.
    """

    def adapt(self, data: dict) -> AgentTrace:
        runs = data.get("runs", [data])  # single run or list of runs

        events = []
        for i, run in enumerate(runs):
            run_type = run.get("run_type", "chain")

            # Map LangSmith run types to cognidrift event types
            if run_type == "tool":
                event_type = "tool_call"
            elif run_type == "retriever":
                event_type = "retrieval"
            elif run_type == "llm":
                event_type = "llm_call"
            else:
                event_type = "tool_call"

            # Compute latency from timestamps
            start = run.get("start_time")
            end = run.get("end_time")
            latency_ms = 0.0
            if start and end:
                try:
                    from datetime import datetime
                    fmt = "%Y-%m-%dT%H:%M:%S.%fZ"
                    latency_ms = (
                        datetime.strptime(end, fmt) - datetime.strptime(start, fmt)
                    ).total_seconds() * 1000
                except Exception:
                    latency_ms = 0.0

            success = run.get("error") is None

            # Truncate large outputs before creating event
            output = run.get("outputs", {})
            if len(str(output)) > 50_000:
                output = {"_truncated": True, "preview": str(output)[:500]}

            event = AgentEvent(
                step_index=i,
                timestamp=time.time(),
                event_type=event_type,
                tool_name=run.get("name"),
                input=run.get("inputs", {}),
                output=output,
                success=success,
                latency_ms=latency_ms,
                metadata={"run_id": run.get("id", "")},
            )
            events.append(event)

        return AgentTrace(
            trace_id=str(data.get("id", "langsmith_trace")),
            agent_name=str(data.get("name", "langsmith_agent")),
            task=str(data.get("inputs", {}).get("input", "")),
            events=events,
        )


class LangfuseAdapter:
    """Handles Langfuse export format.

    Langfuse exports traces with an observations[] array.
    Each observation has type, input, output, startTime, endTime, level fields.
    """

    def adapt(self, data: dict) -> AgentTrace:
        observations = data.get("observations", [])

        events = []
        for i, obs in enumerate(observations):
            obs_type = obs.get("type", "").lower()

            if obs_type in ("tool", "span"):
                event_type = "tool_call"
            elif obs_type == "generation":
                event_type = "llm_call"
            else:
                event_type = "tool_call"

            # Langfuse uses level field for error state
            success = obs.get("level", "DEFAULT") not in ("ERROR", "WARNING")

            # Latency from startTime / endTime
            latency_ms = 0.0
            try:
                from datetime import datetime
                start = obs.get("startTime", "")
                end = obs.get("endTime", "")
                if start and end:
                    fmt = "%Y-%m-%dT%H:%M:%S.%fZ"
                    latency_ms = (
                        datetime.strptime(end, fmt) - datetime.strptime(start, fmt)
                    ).total_seconds() * 1000
            except Exception:
                latency_ms = 0.0

            output = obs.get("output", {}) or {}
            if len(str(output)) > 50_000:
                output = {"_truncated": True, "preview": str(output)[:500]}

            event = AgentEvent(
                step_index=i,
                timestamp=time.time(),
                event_type=event_type,
                tool_name=obs.get("name"),
                input=obs.get("input", {}) or {},
                output=output,
                success=success,
                latency_ms=latency_ms,
                metadata={"observation_id": obs.get("id", "")},
            )
            events.append(event)

        return AgentTrace(
            trace_id=str(data.get("id", "langfuse_trace")),
            agent_name=str(data.get("name", "langfuse_agent")),
            task=str(data.get("input", "")),
            events=events,
        )


def get_adapter(fmt: str):
    adapters = {
        "raw": RawJSONAdapter(),
        "langsmith": LangSmithAdapter(),
        "langfuse": LangfuseAdapter(),
    }
    if fmt not in adapters:
        raise ValueError(f"Unknown format '{fmt}'. Use: raw, langsmith, langfuse")
    return adapters[fmt]
```

---

## PART 5: DETECTORS

### cognidrift/detectors/base.py

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from cognidrift.recorder.trace_recorder import AgentTrace


@dataclass
class DetectorResult:
    detector_name: str
    detected: bool
    score: float        # always 0.0–1.0
    threshold: float
    evidence: dict
    interpretation: str

    def to_dict(self) -> dict:
        return {
            "detected": self.detected,
            "score": round(self.score, 4),
            "threshold": self.threshold,
            "evidence": self.evidence,
            "interpretation": self.interpretation,
        }


class BaseDetector(ABC):
    name: str
    threshold: float

    @abstractmethod
    def detect(self, trace: AgentTrace) -> DetectorResult:
        pass

    def _clamp(self, value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _insufficient(self, reason: str) -> DetectorResult:
        """Standard result for traces that can't be evaluated."""
        return DetectorResult(
            detector_name=self.name,
            detected=False,
            score=0.0,
            threshold=self.threshold,
            evidence={"reason": reason},
            interpretation=reason,
        )
```

### cognidrift/detectors/anchoring.py

```python
"""Detects first-retrieval dominance — agent over-relies on its first
retrieval result regardless of contradicting information found later."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from cognidrift.detectors.base import BaseDetector, DetectorResult
from cognidrift.recorder.trace_recorder import AgentTrace


class AnchoringDetector(BaseDetector):
    name = "anchoring"
    threshold = 0.60

    def detect(self, trace: AgentTrace) -> DetectorResult:
        retrievals = trace.retrievals
        llm_calls = trace.llm_calls

        # Need at least 2 retrievals and 1 LLM output to compute
        if len(retrievals) < 2:
            return self._insufficient(
                "Less than 2 retrieval events — anchoring not computable."
            )
        if not llm_calls:
            return self._insufficient("No LLM output events found.")

        # Use final LLM call as the answer
        final_output = str(llm_calls[-1].output)
        if not final_output.strip():
            return self._insufficient("Final LLM output is empty.")

        # Build text corpus: all retrieval outputs + final answer
        retrieval_texts = []
        for r in retrievals:
            text = str(r.output).strip()
            if text:
                retrieval_texts.append(text)

        if len(retrieval_texts) < 2:
            return self._insufficient(
                "Less than 2 non-empty retrieval outputs."
            )

        docs = retrieval_texts + [final_output]

        try:
            vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
            tfidf_matrix = vectorizer.fit_transform(docs)
        except ValueError:
            # TF-IDF fails on very short or empty documents
            return self._insufficient(
                "Retrieval outputs too short for TF-IDF analysis."
            )

        answer_vec = tfidf_matrix[-1]
        first_sim = float(
            cosine_similarity(tfidf_matrix[0], answer_vec)[0][0]
        )
        later_sims = [
            float(cosine_similarity(tfidf_matrix[i], answer_vec)[0][0])
            for i in range(1, len(retrieval_texts))
        ]
        avg_later_sim = float(np.mean(later_sims)) if later_sims else 0.0

        score = self._clamp(first_sim)

        return DetectorResult(
            detector_name=self.name,
            detected=score >= self.threshold,
            score=score,
            threshold=self.threshold,
            evidence={
                "first_retrieval_similarity_to_answer": round(first_sim, 3),
                "avg_later_retrieval_similarity": round(avg_later_sim, 3),
                "first_retrieval_step": retrievals[0].step_index,
                "answer_step": llm_calls[-1].step_index,
                "total_retrievals": len(retrievals),
            },
            interpretation=(
                f"Answer was {score:.0%} similar to first retrieval. "
                f"Later retrievals averaged {avg_later_sim:.0%} similarity. "
                + (
                    "Anchoring pattern detected."
                    if score >= self.threshold
                    else "No significant anchoring detected."
                )
            ),
        )
```

### cognidrift/detectors/confirmation.py

```python
"""Detects confirmation bias — agent narrows tool usage over time,
seeking confirmation rather than exploring alternative information sources."""

import numpy as np
from cognidrift.detectors.base import BaseDetector, DetectorResult
from cognidrift.recorder.trace_recorder import AgentTrace


class ConfirmationDetector(BaseDetector):
    name = "confirmation_bias"
    threshold = 0.60
    window_size = 5

    def detect(self, trace: AgentTrace) -> DetectorResult:
        tool_events = [
            e for e in trace.events
            if e.event_type in ("tool_call", "retrieval")
        ]

        if len(tool_events) < self.window_size * 2:
            return self._insufficient(
                f"Need at least {self.window_size * 2} tool events for confirmation analysis."
            )

        # Compute unique tool count in rolling windows
        diversity_values = []
        for i in range(len(tool_events) - self.window_size + 1):
            window = tool_events[i: i + self.window_size]
            unique = len(set(e.tool_name or e.event_type for e in window))
            diversity_values.append(unique)

        if len(diversity_values) < 2:
            return self._insufficient("Insufficient windows for diversity trend.")

        # Linear regression slope on diversity over time
        x = np.arange(len(diversity_values), dtype=float)
        y = np.array(diversity_values, dtype=float)
        slope = float(np.polyfit(x, y, 1)[0])

        # Negative slope = narrowing = confirmation bias
        # Normalize: slope of -1.0 per window maps to score of ~1.0
        score = self._clamp(-slope / max(diversity_values[0], 1))

        # First half vs second half unique tools
        mid = len(tool_events) // 2
        first_half_tools = set(e.tool_name or e.event_type for e in tool_events[:mid])
        second_half_tools = set(e.tool_name or e.event_type for e in tool_events[mid:])

        # Tool usage distribution
        usage = {}
        for e in tool_events:
            key = e.tool_name or e.event_type
            usage[key] = usage.get(key, 0) + 1

        return DetectorResult(
            detector_name=self.name,
            detected=score >= self.threshold,
            score=score,
            threshold=self.threshold,
            evidence={
                "tool_diversity_slope": round(slope, 4),
                "unique_tools_first_half": len(first_half_tools),
                "unique_tools_second_half": len(second_half_tools),
                "diversity_values": [round(v, 2) for v in diversity_values],
                "tool_usage_distribution": usage,
            },
            interpretation=(
                f"Tool diversity slope: {slope:.3f} per window. "
                + (
                    f"Narrowed from {len(first_half_tools)} to {len(second_half_tools)} unique tools. "
                    "Confirmation bias pattern detected."
                    if score >= self.threshold
                    else "Tool diversity relatively stable — no confirmation bias detected."
                )
            ),
        )
```

### cognidrift/detectors/sunk_cost.py

```python
"""Detects sunk cost fallacy — agent continuing a failing strategy
past the rational pivot point instead of adapting."""

from cognidrift.detectors.base import BaseDetector, DetectorResult
from cognidrift.recorder.trace_recorder import AgentTrace


class SunkCostDetector(BaseDetector):
    name = "sunk_cost"
    threshold = 0.50

    def detect(self, trace: AgentTrace) -> DetectorResult:
        if not trace.failures:
            return self._insufficient("No failure events in trace — sunk cost not applicable.")

        if trace.total_steps < 6:
            return self._insufficient("Trace too short for sunk cost analysis.")

        # First failure step
        first_failure_step = trace.failures[0].step_index
        failure_step_indices = [e.step_index for e in trace.failures]

        # Tools used before first failure
        tools_before_failure = set(
            e.tool_name or e.event_type
            for e in trace.events
            if e.step_index < first_failure_step
               and e.event_type in ("tool_call", "retrieval")
        )

        # Strategy change = first step after failure using a NEW tool
        strategy_change_step = None
        for event in trace.events:
            if event.step_index <= first_failure_step:
                continue
            if event.event_type not in ("tool_call", "retrieval"):
                continue
            tool = event.tool_name or event.event_type
            if tool not in tools_before_failure:
                strategy_change_step = event.step_index
                break

        # If agent never pivoted — maximum sunk cost
        if strategy_change_step is None:
            steps_after_failure = trace.total_steps - first_failure_step
            score = 1.0
            interpretation = (
                f"Agent never changed strategy after first failure at step {first_failure_step}. "
                f"Continued same approach for {steps_after_failure} remaining steps."
            )
        else:
            steps_after_failure = strategy_change_step - first_failure_step
            score = self._clamp(steps_after_failure / max(trace.total_steps, 1))
            interpretation = (
                f"First failure at step {first_failure_step}. "
                f"Strategy changed at step {strategy_change_step} "
                f"({steps_after_failure} steps later). "
                + (
                    "Significant sunk cost — slow to adapt."
                    if score >= self.threshold
                    else "Reasonable adaptation speed."
                )
            )

        return DetectorResult(
            detector_name=self.name,
            detected=score >= self.threshold,
            score=score,
            threshold=self.threshold,
            evidence={
                "first_failure_step": first_failure_step,
                "strategy_change_step": strategy_change_step,
                "steps_after_failure": steps_after_failure
                if strategy_change_step
                else trace.total_steps - first_failure_step,
                "total_steps": trace.total_steps,
                "failure_signals": failure_step_indices,
                "never_pivoted": strategy_change_step is None,
            },
            interpretation=interpretation,
        )
```

### cognidrift/detectors/loop_detector.py

```python
"""Detects action loops — agent repeating identical or near-identical
action sequences via autocorrelation analysis."""

import numpy as np
from cognidrift.detectors.base import BaseDetector, DetectorResult
from cognidrift.recorder.trace_recorder import AgentTrace


class LoopDetector(BaseDetector):
    name = "loop_detection"
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
        )
```

### cognidrift/detectors/degradation.py

```python
"""Detects performance degradation — agent efficiency declining over
the course of a session, detected via CUSUM changepoint analysis."""

import numpy as np
from cognidrift.detectors.base import BaseDetector, DetectorResult
from cognidrift.recorder.trace_recorder import AgentTrace


class DegradationDetector(BaseDetector):
    name = "degradation"
    threshold = 0.40
    window_size = 5

    def detect(self, trace: AgentTrace) -> DetectorResult:
        if trace.total_steps < self.window_size * 2:
            return self._insufficient(
                f"Need at least {self.window_size * 2} steps for degradation analysis."
            )

        # Efficiency per rolling window: success rate
        events = trace.events
        efficiencies = []
        for i in range(0, len(events) - self.window_size + 1, self.window_size):
            chunk = events[i: i + self.window_size]
            eff = sum(1 for e in chunk if e.success) / len(chunk)
            efficiencies.append(eff)

        if len(efficiencies) < 2:
            return self._insufficient("Insufficient windows for degradation analysis.")

        eff_array = np.array(efficiencies, dtype=float)

        # Guard: all-zero efficiency (complete failure throughout)
        if eff_array.max() == 0.0:
            return DetectorResult(
                detector_name=self.name,
                detected=True,
                score=1.0,
                threshold=self.threshold,
                evidence={
                    "changepoint_detected": True,
                    "changepoint_step": 0,
                    "efficiency_trend": "complete_failure",
                    "cusum_max": 1.0,
                    "efficiency_values": efficiencies,
                },
                interpretation="Agent had 0% success rate across all windows — complete degradation.",
            )

        # Guard: all-one efficiency (perfect throughout — no degradation possible)
        if eff_array.min() == 1.0:
            return self._insufficient(
                "Agent had 100% success rate throughout — no degradation."
            )

        # CUSUM: cumulative sum control chart
        # Detects when efficiency drops below expected level
        target = float(eff_array.mean())
        std = float(eff_array.std())
        k = 0.5 * std if std > 0 else 0.1  # allowance parameter

        cusum = 0.0
        cusum_values = []
        for eff in efficiencies:
            # Lower CUSUM: detects downward shift (degradation)
            cusum = max(0.0, cusum + (target - eff) - k)
            cusum_values.append(cusum)

        max_cusum = float(max(cusum_values))

        # Normalize score: divide by efficiency range
        eff_range = eff_array.max() - eff_array.min()
        score = self._clamp(max_cusum / (eff_range + 1e-6))

        changepoint_detected = score >= self.threshold
        changepoint_window = int(np.argmax(cusum_values)) if changepoint_detected else None
        changepoint_step = (
            changepoint_window * self.window_size if changepoint_window is not None else None
        )

        return DetectorResult(
            detector_name=self.name,
            detected=changepoint_detected,
            score=score,
            threshold=self.threshold,
            evidence={
                "changepoint_detected": changepoint_detected,
                "changepoint_step": changepoint_step,
                "efficiency_trend": "declining" if changepoint_detected else "stable",
                "cusum_max": round(max_cusum, 3),
                "efficiency_values": [round(e, 3) for e in efficiencies],
                "baseline_efficiency": round(target, 3),
            },
            interpretation=(
                f"Efficiency changepoint at step {changepoint_step}. "
                f"Baseline was {target:.0%}."
                if changepoint_detected
                else f"Agent efficiency stable at ~{target:.0%} throughout."
            ),
        )
```

---

## PART 6: METRICS

### cognidrift/metrics/efficiency.py

```python
from cognidrift.recorder.trace_recorder import AgentTrace


def compute_efficiency(trace: AgentTrace) -> dict:
    active_events = [
        e for e in trace.events
        if e.event_type in ("tool_call", "retrieval")
    ]
    goal_advancing = [e for e in active_events if e.success]
    total = trace.total_steps

    value = len(goal_advancing) / total if total > 0 else 0.0

    return {
        "value": round(value, 4),
        "goal_advancing_actions": len(goal_advancing),
        "total_actions": total,
        "interpretation": f"{value:.0%} of actions directly advanced the goal.",
    }
```

### cognidrift/metrics/exploration_score.py

```python
from cognidrift.recorder.trace_recorder import AgentTrace


def compute_exploration_score(trace: AgentTrace) -> dict:
    tool_events = [
        e for e in trace.events
        if e.event_type in ("tool_call", "retrieval")
    ]

    usage: dict[str, int] = {}
    for e in tool_events:
        key = e.tool_name or e.event_type
        usage[key] = usage.get(key, 0) + 1

    unique_used = len(usage)
    # Infer total available as max(unique used, 6) unless we know otherwise
    total_available = max(unique_used, 6)

    value = unique_used / total_available if total_available > 0 else 0.0

    return {
        "value": round(value, 4),
        "unique_tools_used": unique_used,
        "total_tools_available": total_available,
        "tool_usage_distribution": usage,
        "interpretation": f"Used {unique_used} of ~{total_available} available tools.",
    }
```

### cognidrift/metrics/recovery_time.py

```python
from cognidrift.recorder.trace_recorder import AgentTrace


def compute_recovery_time(trace: AgentTrace) -> dict:
    events = trace.events
    failure_indices = [e.step_index for e in trace.failures]

    if not failure_indices:
        return {
            "value": 0,
            "unit": "steps",
            "failure_events": 0,
            "avg_steps_to_recovery": 0,
            "per_failure_recovery": [],
            "interpretation": "No failure events — recovery time not applicable.",
        }

    per_failure = []
    for fail_step in failure_indices:
        # Find next success after this failure
        recovery = None
        for e in events:
            if e.step_index > fail_step and e.success:
                recovery = e.step_index - fail_step
                break
        per_failure.append(recovery)

    # Filter out None (no recovery found after some failures)
    recovered = [r for r in per_failure if r is not None]
    avg = sum(recovered) / len(recovered) if recovered else 0

    return {
        "value": round(avg, 2),
        "unit": "steps",
        "failure_events": len(failure_indices),
        "avg_steps_to_recovery": round(avg, 2),
        "per_failure_recovery": per_failure,
        "interpretation": (
            f"Average {avg:.1f} steps to recover from each failure. "
            f"{len(failure_indices) - len(recovered)} failures had no recovery."
            if failure_indices
            else "No failures detected."
        ),
    }
```

---

## PART 7: REPORT BUILDER

### cognidrift/report/builder.py

```python
"""Assembles behavior_report.json from all detector and metric results."""

import json
from datetime import datetime, timezone
from pathlib import Path

from cognidrift.recorder.trace_recorder import AgentTrace
from cognidrift.detectors.anchoring import AnchoringDetector
from cognidrift.detectors.confirmation import ConfirmationDetector
from cognidrift.detectors.sunk_cost import SunkCostDetector
from cognidrift.detectors.loop_detector import LoopDetector
from cognidrift.detectors.degradation import DegradationDetector
from cognidrift.metrics.efficiency import compute_efficiency
from cognidrift.metrics.exploration_score import compute_exploration_score
from cognidrift.metrics.recovery_time import compute_recovery_time
from cognidrift.utils.paths import safe_output_path, ensure_output_dir


DETECTORS = [
    AnchoringDetector(),
    ConfirmationDetector(),
    SunkCostDetector(),
    LoopDetector(),
    DegradationDetector(),
]


def build_report(trace: AgentTrace, output_dir: str) -> dict:
    """Run all detectors and metrics, assemble report, write to disk."""

    # Run detectors
    detector_results = {}
    detected_scores = []
    biases_detected = []

    for detector in DETECTORS:
        result = detector.detect(trace)
        detector_results[result.detector_name] = result.to_dict()
        if result.detected:
            biases_detected.append(result.detector_name)
            detected_scores.append(result.score)

    # Overall rationality: 1.0 - mean of detected bias scores
    if detected_scores:
        overall_rationality = round(1.0 - (sum(detected_scores) / len(detected_scores)), 4)
    else:
        overall_rationality = 1.0

    # Compute metrics
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
        "biases_detected": biases_detected,
        "detectors": detector_results,
        "metrics": metrics,
    }

    # Write report
    abs_output = ensure_output_dir(output_dir)
    report_path = safe_output_path(abs_output, "behavior_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    return report
```

---

## PART 8: VISUALIZATION

### cognidrift/report/visualize.py

```python
"""Generates behavioral_timeline.png — annotated horizontal timeline
of agent actions with bias detection markers."""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np

from cognidrift.recorder.trace_recorder import AgentTrace
from cognidrift.utils.paths import safe_output_path, ensure_output_dir


EVENT_Y = {
    "tool_call": 3,
    "llm_call": 2,
    "retrieval": 1,
    "error": 0,
}

EVENT_LABELS = {3: "tool call", 2: "llm call", 1: "retrieval", 0: "error"}


def generate_timeline(trace: AgentTrace, report: dict, output_dir: str) -> str:
    """Generate and save behavioral_timeline.png. Returns file path."""

    fig, ax = plt.subplots(figsize=(12, 4), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")

    # Plot each event as a dot
    for event in trace.events:
        y = EVENT_Y.get(event.event_type, 0)
        color = "#22c55e" if event.success else "#ef4444"  # green / red
        size = 60 if event.success else 80
        ax.scatter(event.step_index, y, c=color, s=size, zorder=5, alpha=0.85)

    # Sunk cost shaded region
    sunk_evidence = report["detectors"].get("sunk_cost", {}).get("evidence", {})
    if report["detectors"].get("sunk_cost", {}).get("detected"):
        fail_step = sunk_evidence.get("first_failure_step")
        change_step = sunk_evidence.get("strategy_change_step") or trace.total_steps
        if fail_step is not None:
            ax.axvspan(fail_step, change_step, alpha=0.12, color="#ef4444", label="_nolegend_")
            ax.annotate(
                f"SUNK COST\n{change_step - fail_step} steps",
                xy=((fail_step + change_step) / 2, 3.6),
                ha="center", fontsize=7, color="#b91c1c", fontweight="bold",
            )

    # Anchoring shaded region
    anchor_evidence = report["detectors"].get("anchoring", {}).get("evidence", {})
    if report["detectors"].get("anchoring", {}).get("detected"):
        first_retrieval = anchor_evidence.get("first_retrieval_step", 0)
        answer_step = anchor_evidence.get("answer_step", trace.total_steps)
        ax.axvspan(first_retrieval, answer_step, alpha=0.10, color="#f59e0b", label="_nolegend_")
        ax.annotate(
            "ANCHORING",
            xy=(first_retrieval + 0.5, 3.6),
            ha="left", fontsize=7, color="#92400e", fontweight="bold",
        )

    # Vertical markers
    dashed_lines = []
    if sunk_evidence.get("first_failure_step") is not None:
        ax.axvline(sunk_evidence["first_failure_step"], color="#ef4444",
                   linestyle="--", linewidth=1, alpha=0.6)
    if sunk_evidence.get("strategy_change_step") is not None:
        ax.axvline(sunk_evidence["strategy_change_step"], color="#3b82f6",
                   linestyle="--", linewidth=1, alpha=0.6)
    degrad_evidence = report["detectors"].get("degradation", {}).get("evidence", {})
    if degrad_evidence.get("changepoint_step") is not None:
        ax.axvline(degrad_evidence["changepoint_step"], color="#8b5cf6",
                   linestyle="--", linewidth=1, alpha=0.6)

    # Axes formatting
    ax.set_yticks(list(EVENT_Y.values()))
    ax.set_yticklabels([EVENT_LABELS[v] for v in EVENT_Y.values()], fontsize=9)
    ax.set_xlabel("step index", fontsize=9)
    ax.set_xlim(-0.5, trace.total_steps + 0.5)
    ax.set_ylim(-0.5, 4.2)
    ax.grid(axis="x", linestyle=":", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    # Rationality score annotation
    score = report.get("overall_rationality_score", 1.0)
    score_color = "#16a34a" if score > 0.7 else "#d97706" if score > 0.4 else "#dc2626"
    ax.text(
        0.99, 0.95,
        f"rationality: {score:.2f}",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=9, color=score_color, fontweight="bold",
    )

    plt.tight_layout()

    abs_output = ensure_output_dir(output_dir)
    out_path = safe_output_path(abs_output, "behavioral_timeline.png")
    plt.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close()
    return out_path
```

---

## PART 9: CLI

### cognidrift/cli.py

```python
"""CLI entry point for cognidrift.

analyze command: zero API keys required.
generate command: requires GEMINI_API_KEY environment variable.
"""

import json
import os
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich import print as rprint

from cognidrift.recorder.format_adapters import get_adapter
from cognidrift.utils.format_detect import detect_format
from cognidrift.utils.validators import TraceValidationError
from cognidrift.utils.paths import PathTraversalError
from cognidrift.report.builder import build_report
from cognidrift.report.visualize import generate_timeline

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="cognidrift")
def main():
    """cognidrift — behavioral auditing for autonomous agents."""
    pass


@main.command()
@click.option("--trace", "trace_path", required=True,
              help="Path to trace file (.json) or directory of trace files.")
@click.option("--format", "fmt", default="auto",
              type=click.Choice(["auto", "langsmith", "langfuse", "raw"]),
              help="Trace format. Default: auto-detect.")
@click.option("--output", "output_dir", default=".",
              help="Output directory for report and visualization. Default: current directory.")
@click.option("--no-viz", is_flag=True, default=False,
              help="Skip PNG timeline generation.")
@click.option("--json-only", is_flag=True, default=False,
              help="Skip terminal output, write JSON only.")
def analyze(trace_path, fmt, output_dir, no_viz, json_only):
    """Analyze an agent trace for behavioral bias patterns.

    Does not require any API key.

    Example:
        cognidrift analyze --trace ./runs/sample_runs/run_anchoring.json
    """
    path = Path(trace_path)

    # Collect trace files
    if path.is_dir():
        trace_files = list(path.glob("*.json"))
        if not trace_files:
            console.print(f"[red]No JSON files found in {trace_path}[/red]")
            sys.exit(1)
    elif path.is_file():
        trace_files = [path]
    else:
        console.print(f"[red]Path not found: {trace_path}[/red]")
        sys.exit(1)

    for trace_file in trace_files:
        if not json_only:
            console.print(f"\n[bold]Analyzing:[/bold] {trace_file.name}")

        try:
            with open(trace_file) as f:
                raw = json.load(f)
        except json.JSONDecodeError as e:
            console.print(f"[red]Invalid JSON in {trace_file.name}: {e}[/red]")
            continue

        # Format detection
        try:
            resolved_fmt = fmt if fmt != "auto" else detect_format(raw)
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            continue

        # Adapt to AgentTrace
        try:
            adapter = get_adapter(resolved_fmt)
            trace = adapter.adapt(raw)
        except (TraceValidationError, ValueError) as e:
            console.print(f"[red]Trace validation failed: {e}[/red]")
            continue

        if not json_only:
            with console.status("Running behavioral detectors..."):
                try:
                    report = build_report(trace, output_dir)
                except PathTraversalError as e:
                    console.print(f"[red]{e}[/red]")
                    continue
        else:
            try:
                report = build_report(trace, output_dir)
            except PathTraversalError as e:
                console.print(f"[red]{e}[/red]")
                continue

        if not json_only:
            _print_results(report)

        if not no_viz:
            try:
                png_path = generate_timeline(trace, report, output_dir)
                if not json_only:
                    console.print(f"[green]Timeline saved:[/green] {png_path}")
            except Exception as e:
                console.print(f"[yellow]Visualization failed (non-fatal): {e}[/yellow]")

        if not json_only:
            console.print(f"[green]Report saved:[/green] {output_dir}/behavior_report.json")


def _print_results(report: dict) -> None:
    """Print a rich table of detector results to terminal."""
    score = report["overall_rationality_score"]
    score_color = "green" if score > 0.7 else "yellow" if score > 0.4 else "red"

    console.print(
        f"\nOverall rationality score: [{score_color}]{score:.2f}[/{score_color}]"
    )

    if report["biases_detected"]:
        console.print(f"Biases detected: [red]{', '.join(report['biases_detected'])}[/red]")
    else:
        console.print("[green]No biases detected.[/green]")

    table = Table(show_header=True, header_style="bold", box=None)
    table.add_column("Detector", style="dim", width=22)
    table.add_column("Score", justify="right", width=8)
    table.add_column("Threshold", justify="right", width=10)
    table.add_column("Detected", justify="center", width=10)

    for name, result in report["detectors"].items():
        detected = result["detected"]
        color = "red" if detected else "green"
        table.add_row(
            name,
            f"{result['score']:.3f}",
            f"{result['threshold']:.2f}",
            f"[{color}]{'YES' if detected else 'no'}[/{color}]",
        )

    console.print(table)


@main.command()
@click.option("--scenarios", default="all",
              help="Comma-separated bias scenarios to generate. Options: anchoring, confirmation, sunk_cost, loop, degradation, clean, all")
@click.option("--output", "output_dir", default="./runs/",
              help="Output directory for generated traces.")
def generate(scenarios, output_dir):
    """Generate synthetic agent traces using LangGraph + Gemini.

    Requires GEMINI_API_KEY environment variable.
    The analyze command does NOT require this — it works on any trace file.

    Example:
        export GEMINI_API_KEY=your_key_here
        cognidrift generate --scenarios anchoring,sunk_cost
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        console.print(
            "[red]GEMINI_API_KEY environment variable not set.[/red]\n"
            "Set it with: export GEMINI_API_KEY=your_key_here\n\n"
            "Note: The [bold]analyze[/bold] command does not require any API key.\n"
            "You can analyze existing traces without generating new ones."
        )
        sys.exit(1)

    try:
        from runs.generate_runs import run_generation
        run_generation(scenarios=scenarios, output_dir=output_dir, api_key=api_key)
    except Exception:
        # Never expose raw exceptions — may contain key fragments
        console.print(
            "[red]Generation failed.[/red]\n"
            "Check that GEMINI_API_KEY is valid and has quota remaining."
        )
        sys.exit(1)
```

---

## PART 10: SAMPLE RUNS

Create 5 handcrafted JSON files in `runs/sample_runs/`. These must be runnable with `cognidrift analyze` immediately, zero API key, zero setup.

### run_anchoring.json
20 events. Research task. 3 retrieval events — retrieval at step 0 has output content about "transformer attention mechanisms" which closely matches the final LLM output at step 19. Retrievals at steps 7 and 12 return different content about "LSTM architectures" and "RNN variants". Agent ignores later retrievals.

Structure events so:
- Steps 0–4: initial search + first retrieval (output: detailed transformer attention content)
- Steps 5–11: more searches + second retrieval (output: LSTM content)
- Steps 12–16: third retrieval (output: RNN content) + analysis
- Steps 17–19: LLM calls, final output closely mirrors step-0 retrieval content
- All events success=true (anchoring is a silent bias — no errors)

### run_confirmation.json
18 events. Task: "Find evidence that transformers outperform RNNs". Starts with 4 different tools (web_search, document_reader, arxiv_search, citation_finder). By step 10, only web_search is used. Tool diversity clearly narrows from 4 → 1 tool type in the second half.

### run_sunk_cost.json
24 events. Debugging task. First failure at step 5 (tool: code_executor, success: false). Agent continues using code_executor with minor variations until step 19. New tool (log_analyzer) only introduced at step 20. Events 5–19 all have mixed success but same tool strategy.

### run_loop.json
22 events. Open-ended research task. Tool sequence follows a clear repeating pattern: [web_search, document_reader, web_search, document_reader, ...] with period 2–3 steps. High autocorrelation guaranteed by this structure.

### run_clean.json
15 events. Straightforward data retrieval task. 4 different tools used throughout. No failures. Efficiency high. No loops. This is the baseline — all detectors should return detected=false.

**All 5 files must strictly follow the raw JSON schema:**
```json
{
  "trace_id": "string",
  "agent_name": "string",
  "task": "string",
  "events": [
    {
      "step": 0,
      "type": "tool_call | llm_call | retrieval | error",
      "tool": "string or null",
      "input": {},
      "output": {},
      "success": true,
      "latency_ms": 200
    }
  ]
}
```

---

## PART 11: GENERATE RUNS (requires Gemini API key)

### runs/generate_runs.py

```python
"""Generates synthetic agent traces using LangGraph + Gemini API.
Called by `cognidrift generate` command. Never called by analyze.
Never import from CLI — this is invoked only when the user explicitly
runs the generate command with a valid API key.
"""

import json
import os
from pathlib import Path


def run_generation(scenarios: str, output_dir: str, api_key: str) -> None:
    """Generate trace files for specified scenarios."""
    import google.generativeai as genai
    from langgraph.graph import StateGraph, END
    from runs.bias_scenarios import SCENARIOS

    genai.configure(api_key=api_key)

    target_scenarios = (
        list(SCENARIOS.keys())
        if scenarios == "all"
        else [s.strip() for s in scenarios.split(",")]
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for scenario_name in target_scenarios:
        if scenario_name not in SCENARIOS:
            print(f"Unknown scenario: {scenario_name}. Skipping.")
            continue

        scenario = SCENARIOS[scenario_name]
        print(f"Generating: {scenario_name}...")

        try:
            trace = _run_agent(scenario, genai, api_key)
            out_path = Path(output_dir) / f"generated_{scenario_name}.json"
            with open(out_path, "w") as f:
                json.dump(trace, f, indent=2)
            print(f"  Saved: {out_path}")
        except Exception:
            # Never expose raw exception — may contain API key fragments
            print(f"  Failed to generate {scenario_name}. Check API quota.")


def _run_agent(scenario: dict, genai, api_key: str) -> dict:
    """Run a LangGraph agent on a scenario and capture the trace."""
    import time

    events = []
    step = 0

    model = genai.GenerativeModel("gemini-1.5-flash")
    tools = scenario.get("tools", ["web_search", "document_reader"])
    task = scenario["task"]

    # Simple sequential agent: LLM decides tool, tool executes, repeat
    messages = [{"role": "user", "parts": [task]}]

    for _ in range(scenario.get("max_steps", 15)):
        start = time.time()
        try:
            response = model.generate_content(messages)
            llm_output = response.text if response.text else ""
        except Exception:
            raise  # Re-raise — caller handles without exposing key

        latency = (time.time() - start) * 1000

        # Record LLM call
        events.append({
            "step": step,
            "type": "llm_call",
            "tool": None,
            "input": {"messages": len(messages)},
            "output": {"text": llm_output[:500]},  # truncate for safety
            "success": True,
            "latency_ms": round(latency, 1),
        })
        step += 1

        # Simulate tool call based on scenario logic
        tool_name = _pick_tool(scenario, step, tools)
        tool_success = _tool_succeeds(scenario, step)
        tool_output = _simulate_tool_output(tool_name, task, tool_success)

        events.append({
            "step": step,
            "type": "tool_call" if tool_name != "retrieval" else "retrieval",
            "tool": tool_name,
            "input": {"query": task[:100]},
            "output": tool_output,
            "success": tool_success,
            "latency_ms": round(200 + step * 10, 1),
        })
        step += 1

        messages.append({"role": "model", "parts": [llm_output]})

        if step >= scenario.get("max_steps", 15):
            break

    return {
        "trace_id": f"generated_{scenario['name']}_{int(time.time())}",
        "agent_name": "gemini_langgraph_agent",
        "task": task,
        "events": events,
    }


def _pick_tool(scenario: dict, step: int, tools: list) -> str:
    """Scenario-specific tool selection logic."""
    bias = scenario.get("bias_type")
    if bias == "loop" and tools:
        return tools[step % len(tools)]
    if bias == "confirmation":
        # Narrow to first tool after step 8
        return tools[0] if step > 8 and tools else (tools[step % len(tools)] if tools else "web_search")
    return tools[step % len(tools)] if tools else "web_search"


def _tool_succeeds(scenario: dict, step: int) -> bool:
    """Scenario-specific success logic."""
    fail_after = scenario.get("fail_after_step")
    if fail_after and step > fail_after:
        # Fail rate increases after threshold
        import random
        return random.random() > 0.6
    return True


def _simulate_tool_output(tool: str, task: str, success: bool) -> dict:
    if not success:
        return {"error": "Tool returned no results", "results": []}
    return {"results": [f"Simulated output from {tool} for task: {task[:50]}..."]}
```

### runs/bias_scenarios.py

```python
"""Task definitions designed to trigger each behavioral bias type."""

SCENARIOS = {
    "anchoring": {
        "name": "anchoring",
        "bias_type": "anchoring",
        "task": "Research the current best practices for RAG (retrieval-augmented generation) evaluation. Find the most important metrics and methods.",
        "tools": ["web_search", "document_reader", "arxiv_search"],
        "max_steps": 20,
    },
    "confirmation": {
        "name": "confirmation",
        "bias_type": "confirmation",
        "task": "Find evidence that transformer models outperform recurrent neural networks on natural language processing tasks.",
        "tools": ["web_search", "arxiv_search", "citation_finder", "document_reader"],
        "max_steps": 18,
    },
    "sunk_cost": {
        "name": "sunk_cost",
        "bias_type": "sunk_cost",
        "task": "Debug why this Python function returns None instead of the expected dictionary: def process_data(x): result = transform(x) return result.get('output')",
        "tools": ["code_executor", "log_analyzer", "documentation_search"],
        "fail_after_step": 5,
        "max_steps": 24,
    },
    "loop": {
        "name": "loop",
        "bias_type": "loop",
        "task": "Compile a comprehensive list of all papers published on agent evaluation benchmarks in 2024.",
        "tools": ["web_search", "document_reader"],
        "max_steps": 22,
    },
    "degradation": {
        "name": "degradation",
        "bias_type": "degradation",
        "task": "Create a 15-step research agenda covering all major open problems in large language model evaluation.",
        "tools": ["web_search", "document_reader", "arxiv_search", "note_taker"],
        "max_steps": 30,
    },
    "clean": {
        "name": "clean",
        "bias_type": None,
        "task": "Find the current exchange rate between USD and EUR and summarize recent trends.",
        "tools": ["web_search", "financial_data", "calculator"],
        "max_steps": 12,
    },
}
```

---

## PART 12: TESTS

### tests/conftest.py

```python
"""Shared fixtures for all tests."""
import pytest
from cognidrift.recorder.trace_recorder import AgentEvent, AgentTrace


def make_event(step: int, event_type="tool_call", tool="web_search",
               success=True, output=None) -> AgentEvent:
    return AgentEvent(
        step_index=step,
        timestamp=float(step),
        event_type=event_type,
        tool_name=tool,
        input={"query": f"query_{step}"},
        output=output or {"results": [f"result_{step}"]},
        success=success,
        latency_ms=100.0,
    )


def make_trace(events: list[AgentEvent], task="test task") -> AgentTrace:
    return AgentTrace(
        trace_id="test_trace",
        agent_name="test_agent",
        task=task,
        events=events,
    )
```

### tests/test_anchoring.py

```python
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
```

Write similar test files for confirmation, sunk_cost, loop_detector, and degradation — each with:
- One test that constructs a trace guaranteed to trigger the bias (detected=True)
- One test that constructs a clean trace (detected=False)
- One test for the edge case (empty trace, constant sequence, no failures, etc.)

---

## PART 13: GITHUB ACTIONS

### .github/workflows/cognidrift_report.yml

```yaml
name: cognidrift behavioral report
on:
  push:
    branches: [main]

jobs:
  analyze:
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install cognidrift
        run: pip install -e .
      - name: Run behavioral analysis on sample trace
        run: |
          mkdir -p report
          cognidrift analyze \
            --trace ./runs/sample_runs/run_anchoring.json \
            --output ./report/ \
            --no-viz
      - name: Commit updated report
        run: |
          git config user.email "action@github.com"
          git config user.name "cognidrift-bot"
          git add ./report/behavior_report.json
          git diff --cached --quiet || git commit -m "chore: update behavioral report [skip ci]"
          git push
```

---

## PART 14: CONFIGURATION FILES

### pyproject.toml

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "cognidrift"
version = "0.1.0"
description = "Behavioral auditing for autonomous agents"
authors = [{name = "Aditya Khankar"}]
license = {text = "MIT"}
requires-python = ">=3.11"
readme = "README.md"
keywords = ["ai", "agents", "evaluation", "behavioral-economics", "llm"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
dependencies = [
    "langchain-core>=0.1.0",
    "langgraph>=0.1.0",
    "google-generativeai>=0.4.0",
    "numpy>=1.26.0",
    "scipy>=1.11.0",
    "scikit-learn>=1.3.0",
    "pydantic>=2.0.0",
    "click>=8.1.0",
    "rich>=13.0.0",
    "matplotlib>=3.8.0",
]

[project.scripts]
cognidrift = "cognidrift.cli:main"

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
]
```

### requirements.txt (pinned for reproducibility)

```
langchain-core==0.1.52
langgraph==0.1.19
google-generativeai==0.5.4
numpy==1.26.4
scipy==1.13.0
scikit-learn==1.4.2
pydantic==2.7.1
click==8.1.7
rich==13.7.1
matplotlib==3.8.4
```

### requirements-dev.txt

```
-r requirements.txt
pytest==7.4.4
pytest-cov==4.1.0
black==24.4.2
ruff==0.4.3
```

### .gitignore

```
.env
__pycache__/
*.pyc
*.pyo
.pytest_cache/
dist/
build/
*.egg-info/
.coverage
htmlcov/
behavioral_timeline.png
generated_*.json
.DS_Store
```

### .env.example

```bash
# Copy this to .env and fill in your key
# Only required for: cognidrift generate
# NOT required for: cognidrift analyze

GEMINI_API_KEY=your_gemini_api_key_here
```

### LICENSE

```
MIT License

Copyright (c) 2026 Aditya Khankar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## PART 15: README.md

Write the README following these rules exactly.

**Content:**

```markdown
# cognidrift

Behavioral auditing for autonomous agents.

## What it detects

cognidrift analyzes agent traces and surfaces irrational decision patterns — 
anchoring on first retrieval results, seeking confirmation over exploration, 
continuing failed strategies past the rational pivot point, repeating action 
loops, and degrading performance across a session. Each pattern is scored 0–1 
with evidence and an overall rationality score.

## Quickstart

pip install cognidrift

cognidrift analyze --trace ./runs/sample_runs/run_anchoring.json

No API key required for analysis.

## Output

[behavioral_timeline.png embedded here]

[behavior_report.json excerpt here — show anchoring detector result only]

## Supported formats

- Raw JSON (native format — schema below)
- LangSmith export
- Langfuse export

cognidrift analyze --trace ./trace.json              # auto-detect format
cognidrift analyze --trace ./trace.json --format langsmith

## Raw JSON schema

{
  "trace_id": "string",
  "agent_name": "string",
  "task": "string",
  "events": [
    {
      "step": 0,
      "type": "tool_call | llm_call | retrieval | error",
      "tool": "string or null",
      "input": {},
      "output": {},
      "success": true,
      "latency_ms": 200
    }
  ]
}

## Generate synthetic traces

Requires GEMINI_API_KEY.

export GEMINI_API_KEY=your_key_here
cognidrift generate --scenarios anchoring,sunk_cost --output ./runs/

## Detectors

| Detector | Detects | Method |
|---|---|---|
| Anchoring | First-retrieval dominance | TF-IDF cosine similarity |
| Confirmation bias | Tool diversity decay | Linear regression slope |
| Sunk cost | Steps after failure before pivot | Step counter |
| Loop detection | Repeated action sequences | Autocorrelation |
| Degradation | Efficiency decline | CUSUM changepoint |

## Architecture

Traces are loaded and validated, adapted from native formats into a canonical
AgentTrace schema, then passed through five independent behavioral detectors.
Metrics are computed separately. All results are assembled into behavior_report.json.
The visualization layer reads the report and the trace to generate the annotated timeline.

## License

MIT — Aditya Khankar
```

**README rules (enforce strictly):**
- Never write "student project", "hackathon", "university", "NITK", "college", "I built"
- Never write "this is my first" or any self-referential framing
- No unverifiable superlatives: "first", "best", "only", "most powerful"
- No motivational language
- Write in imperative or third person — like documentation for a tool someone else built

---

## PART 16: GIT PUSH

1. Create repo at github.com/Aditya-Khankar/cognidrift if not exists
2. Initialize git in project root
3. Run `git status` — verify .env is NOT listed (must be in .gitignore)
4. `git add .`
5. Run `git status` again — confirm only intended files staged
6. Commit: `feat: initial cognidrift release — behavioral auditing for autonomous agents`
7. Push to main branch

---

## DONE CHECKLIST

Before marking complete, verify every item:

- [ ] `cognidrift analyze --trace ./runs/sample_runs/run_anchoring.json` runs end-to-end
- [ ] `cognidrift analyze --trace ./runs/sample_runs/` analyzes all 5 sample runs
- [ ] `behavior_report.json` generated with all 5 detector results and 3 metrics
- [ ] `behavioral_timeline.png` generated
- [ ] All 5 sample runs produce valid non-empty detector output
- [ ] `cognidrift generate` exits with clear error if GEMINI_API_KEY not set
- [ ] All tests pass (`pytest tests/`)
- [ ] `.env` is in `.gitignore` and NOT committed
- [ ] GitHub Actions workflow only triggers on `branches: [main]`
- [ ] README contains no student/institutional framing
- [ ] LICENSE is MIT under Aditya Khankar, year 2026
- [ ] `pyproject.toml` entry point is `cognidrift = "cognidrift.cli:main"`
- [ ] Pushed to github.com/Aditya-Khankar/cognidrift
