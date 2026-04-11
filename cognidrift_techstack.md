# cognidrift — Tech Stack & Architecture

**Version:** 1.0  
**Author:** Aditya Khankar

---

## 1. Language & runtime

| Choice | Reason |
|---|---|
| Python 3.11+ | Type hints with `Self`, `match` statements, faster runtime. Standard for ML tooling. |
| No async | All operations are batch/offline. Async adds complexity with zero benefit here. |

---

## 2. Dependencies (full)

### Core pipeline
```
langchain-core>=0.1.0       # Base abstractions — messages, tools, runnables
langgraph>=0.1.0             # Agent orchestration — nodes, edges, state graph
google-generativeai>=0.4.0   # Gemini API — free tier, function calling
```

### Numerical computation (detectors)
```
numpy>=1.26.0               # Autocorrelation, array ops, CUSUM computation
scipy>=1.11.0               # signal.correlate, stats — statistical functions
scikit-learn>=1.3.0         # TfidfVectorizer, cosine_similarity — anchoring detector
```

### Data validation
```
pydantic>=2.0.0             # Trace schema validation, AgentEvent dataclass
```

### CLI
```
click>=8.1.0                # CLI framework — commands, options, argument parsing
rich>=13.0.0                # Terminal output — tables, progress bars, colored scores
```

### Visualization
```
matplotlib>=3.8.0           # Behavioral timeline PNG generation
```

### Dev/test
```
pytest>=7.4.0               # Test runner
pytest-cov>=4.1.0           # Coverage reporting
black>=23.0.0               # Formatter
ruff>=0.1.0                 # Linter
```

### Full requirements.txt
```
langchain-core>=0.1.0
langgraph>=0.1.0
google-generativeai>=0.4.0
numpy>=1.26.0
scipy>=1.11.0
scikit-learn>=1.3.0
pydantic>=2.0.0
click>=8.1.0
rich>=13.0.0
matplotlib>=3.8.0
pytest>=7.4.0
pytest-cov>=4.1.0
black>=23.0.0
ruff>=0.1.0
```

---

## 3. Project structure (complete)

```
cognidrift/
│
├── cognidrift/                 # Main package
│   ├── __init__.py
│   ├── cli.py                  # Click entry point
│   │
│   ├── recorder/
│   │   ├── __init__.py
│   │   ├── trace_recorder.py   # AgentEvent dataclass + session recording
│   │   └── format_adapters.py  # LangSmith / Langfuse / Raw JSON → AgentEvent
│   │
│   ├── detectors/
│   │   ├── __init__.py
│   │   ├── base.py             # BaseDetector abstract class
│   │   ├── anchoring.py        # TF-IDF similarity, first-retrieval dominance
│   │   ├── confirmation.py     # Tool diversity decay + confidence proxy
│   │   ├── sunk_cost.py        # Steps-after-failure counter
│   │   ├── loop_detector.py    # Autocorrelation-based repetition
│   │   └── degradation.py      # CUSUM changepoint detection
│   │
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── efficiency.py       # Goal-advancing / total actions
│   │   ├── exploration_score.py # Tool diversity over time
│   │   └── recovery_time.py    # Steps from failure to correction
│   │
│   ├── report/
│   │   ├── __init__.py
│   │   ├── builder.py          # Assembles behavior_report.json from detector outputs
│   │   └── visualize.py        # Matplotlib timeline PNG
│   │
│   └── utils/
│       ├── __init__.py
│       └── format_detect.py    # Auto-detect trace format
│
├── runs/
│   ├── generate_runs.py        # LangGraph + Gemini agent, generates traces
│   ├── bias_scenarios.py       # Task definitions that trigger each bias
│   └── sample_runs/            # 5 pre-generated traces (committed to repo)
│       ├── run_anchoring.json
│       ├── run_confirmation.json
│       ├── run_sunk_cost.json
│       ├── run_loop.json
│       └── run_clean.json      # Baseline: no biases detected
│
├── report/
│   └── behavior_report.json    # Auto-updated by GitHub Actions on every push
│
├── tests/
│   ├── test_adapters.py
│   ├── test_anchoring.py
│   ├── test_confirmation.py
│   ├── test_sunk_cost.py
│   ├── test_loop_detector.py
│   ├── test_degradation.py
│   └── test_metrics.py
│
├── .github/
│   └── workflows/
│       └── cognidrift_report.yml
│
├── pyproject.toml              # Package config + entry point
├── requirements.txt
├── README.md
├── LICENSE                     # MIT — Aditya Khankar
└── .gitignore
```

---

## 4. Core data model

```python
# cognidrift/recorder/trace_recorder.py

from dataclasses import dataclass, field
from typing import Literal

EventType = Literal["tool_call", "llm_call", "retrieval", "error"]

@dataclass
class AgentEvent:
    step_index: int
    timestamp: float          # Unix timestamp
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
```

---

## 5. BaseDetector interface

```python
# cognidrift/detectors/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from cognidrift.recorder.trace_recorder import AgentTrace

@dataclass
class DetectorResult:
    detector_name: str
    detected: bool
    score: float              # 0.0 to 1.0
    threshold: float
    evidence: dict
    interpretation: str

class BaseDetector(ABC):
    name: str
    threshold: float
    
    @abstractmethod
    def detect(self, trace: AgentTrace) -> DetectorResult:
        """Run detection on a trace. Returns DetectorResult."""
        pass
    
    def _clamp(self, value: float) -> float:
        return max(0.0, min(1.0, value))
```

---

## 6. Detector implementations

### anchoring.py
```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from cognidrift.detectors.base import BaseDetector, DetectorResult
from cognidrift.recorder.trace_recorder import AgentTrace

class AnchoringDetector(BaseDetector):
    name = "anchoring"
    threshold = 0.60
    
    def detect(self, trace: AgentTrace) -> DetectorResult:
        retrievals = [e for e in trace.events if e.event_type == "retrieval"]
        if len(retrievals) < 2:
            return DetectorResult(
                detector_name=self.name,
                detected=False,
                score=0.0,
                threshold=self.threshold,
                evidence={"reason": "insufficient retrievals"},
                interpretation="Less than 2 retrieval events — anchoring not computable."
            )
        
        # Get final LLM output
        llm_events = [e for e in trace.events if e.event_type == "llm_call"]
        if not llm_events:
            return DetectorResult(
                detector_name=self.name, detected=False, score=0.0,
                threshold=self.threshold,
                evidence={"reason": "no llm_call events found"},
                interpretation="No LLM output found."
            )
        
        final_output = str(llm_events[-1].output)
        
        # TF-IDF similarity: first retrieval vs answer, later retrievals vs answer
        docs = [str(r.output) for r in retrievals] + [final_output]
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(docs)
        
        answer_vec = tfidf_matrix[-1]
        first_sim = cosine_similarity(tfidf_matrix[0], answer_vec)[0][0]
        later_sims = [
            cosine_similarity(tfidf_matrix[i], answer_vec)[0][0]
            for i in range(1, len(retrievals))
        ]
        avg_later_sim = np.mean(later_sims) if later_sims else 0.0
        
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
                "answer_step": llm_events[-1].step_index
            },
            interpretation=(
                f"Agent answer was {score:.0%} similar to first retrieval. "
                f"Later retrievals averaged {avg_later_sim:.0%} similarity."
            )
        )
```

### loop_detector.py
```python
import numpy as np
from cognidrift.detectors.base import BaseDetector, DetectorResult
from cognidrift.recorder.trace_recorder import AgentTrace

class LoopDetector(BaseDetector):
    name = "loop_detection"
    threshold = 0.65
    
    def detect(self, trace: AgentTrace) -> DetectorResult:
        # Encode tool sequence as integers
        tool_vocab = {}
        counter = 0
        sequence = []
        for event in trace.events:
            tool = event.tool_name or event.event_type
            if tool not in tool_vocab:
                tool_vocab[tool] = counter
                counter += 1
            sequence.append(tool_vocab[tool])
        
        if len(sequence) < 6:
            return DetectorResult(
                detector_name=self.name, detected=False, score=0.0,
                threshold=self.threshold,
                evidence={"reason": "sequence too short"},
                interpretation="Trace too short for loop detection."
            )
        
        # Autocorrelation
        seq = np.array(sequence, dtype=float)
        seq -= seq.mean()
        autocorr = np.correlate(seq, seq, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr /= autocorr[0] if autocorr[0] != 0 else 1
        
        # Check lags 2–10 (ignore lag 0 = self-correlation)
        lag_values = autocorr[2:min(11, len(autocorr))]
        max_autocorr = float(np.max(np.abs(lag_values))) if len(lag_values) > 0 else 0.0
        dominant_lag = int(np.argmax(np.abs(lag_values))) + 2 if len(lag_values) > 0 else None
        
        score = self._clamp(max_autocorr)
        
        return DetectorResult(
            detector_name=self.name,
            detected=score >= self.threshold,
            score=score,
            threshold=self.threshold,
            evidence={
                "max_autocorrelation": round(max_autocorr, 3),
                "dominant_lag": dominant_lag,
                "sequence_length": len(sequence)
            },
            interpretation=(
                f"Action sequence autocorrelation: {max_autocorr:.2f} at lag {dominant_lag}. "
                + ("Repetitive loop pattern detected." if score >= self.threshold else "No significant loop detected.")
            )
        )
```

### degradation.py (CUSUM)
```python
import numpy as np
from cognidrift.detectors.base import BaseDetector, DetectorResult
from cognidrift.recorder.trace_recorder import AgentTrace

class DegradationDetector(BaseDetector):
    name = "degradation"
    threshold = 0.40
    
    def detect(self, trace: AgentTrace) -> DetectorResult:
        # Compute efficiency in rolling windows of 5 steps
        window = 5
        events = trace.events
        
        if len(events) < window * 2:
            return DetectorResult(
                detector_name=self.name, detected=False, score=0.0,
                threshold=self.threshold,
                evidence={"reason": "trace too short"},
                interpretation="Trace too short for degradation detection."
            )
        
        # Efficiency per window: success rate
        efficiencies = []
        for i in range(0, len(events) - window + 1, window):
            chunk = events[i:i+window]
            eff = sum(1 for e in chunk if e.success) / len(chunk)
            efficiencies.append(eff)
        
        # CUSUM on efficiency
        target = np.mean(efficiencies)
        k = 0.5 * np.std(efficiencies)  # allowance
        cusum = 0.0
        cusum_values = []
        for eff in efficiencies:
            cusum = max(0.0, cusum + (target - eff) - k)
            cusum_values.append(cusum)
        
        max_cusum = float(max(cusum_values)) if cusum_values else 0.0
        # Normalize to 0-1 range
        score = self._clamp(max_cusum / (max(efficiencies) - min(efficiencies) + 1e-6))
        
        changepoint_detected = score >= self.threshold
        changepoint_window = int(np.argmax(cusum_values)) if changepoint_detected else None
        changepoint_step = changepoint_window * window if changepoint_window is not None else None
        
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
                "efficiency_values": [round(e, 3) for e in efficiencies]
            },
            interpretation=(
                f"Efficiency changepoint detected at step {changepoint_step}."
                if changepoint_detected else
                "Agent efficiency remained stable throughout the session."
            )
        )
```

---

## 7. pyproject.toml

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
dev = ["pytest>=7.4.0", "pytest-cov>=4.1.0", "black>=23.0.0", "ruff>=0.1.0"]

[tool.black]
line-length = 88
target-version = ["py311"]

[tool.ruff]
line-length = 88
select = ["E", "F", "I"]
```

---

## 8. Environment variables

```bash
# .env (never commit this)
GEMINI_API_KEY=your_key_here

# Load in generate_runs.py:
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.environ["GEMINI_API_KEY"]
```

`.gitignore` must include:
```
.env
__pycache__/
*.pyc
.pytest_cache/
dist/
*.egg-info/
.coverage
behavioral_timeline.png  # generated, not committed (except sample)
```

---

## 9. README structure

```markdown
# cognidrift

Behavioral auditing for autonomous agents.

## What it detects
[3-sentence explanation of the 5 biases, no jargon]

## Quickstart
pip install cognidrift
cognidrift analyze --trace ./runs/sample_runs/run_anchoring.json

## Output
[embed behavioral_timeline.png from report/]
[embed truncated behavior_report.json snippet]

## Supported formats
- Raw JSON (schema below)
- LangSmith export
- Langfuse export

## Raw JSON schema
[schema block]

## Generate your own runs
[instructions — requires GEMINI_API_KEY]

## Architecture
[brief module overview]

## License
MIT — Aditya Khankar
```

Rules: no "student project", no "I built this for", no unverifiable superlatives. Write like a builder shipping a tool, not a student submitting a project.
