# XAudit

> Your agent completed the task. Did it reason well?

Detect primacy dominance, query entropy collapse, strategy persistence, cyclic redundancy, and context decay in any autonomous agent trace.

![Python](https://img.shields.io/badge/python-3.11+-blue?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)
![Formats](https://img.shields.io/badge/formats-LangSmith%20%7C%20Langfuse%20%7C%20Raw%20JSON-orange?style=flat-square)

```bash
$ xaudit demo

Overall rationality score: 0.84
Patterns detected: primacy_dominance
  → Answer was 93% similar to first retrieval. Later retrievals averaged 9% similarity.

Detector                  Score   Threshold   Detected
primacy_dominance         0.793   0.35        YES
query_entropy_collapse    0.000   0.40        no
strategy_persistence      0.000   0.50        no
cyclic_redundancy         0.000   0.65        no
context_decay             0.000   0.40        no

Timeline saved: ./demo_output/behavioral_timeline.png
Report saved:   ./demo_output/behavior_report.json
```

---

## Quickstart

**Requirements:** Python 3.11+

```bash
git clone https://github.com/Aditya-Khankar/xaudit
cd xaudit
pip install -e .
xaudit demo
```

No API key or configuration required. Results in under 60 seconds.

> Having trouble? See [INSTALL.md](INSTALL.md) for platform-specific instructions.

---

## 🎨 Themes & Logging

XAudit is built for a premium developer experience. It ships with two visual themes and professional stylized logging.

```bash
xaudit demo --theme amber       # warm gold on dark (default)
xaudit demo --theme nebula      # purple & plain text
```

**Save your preferred theme:**

```bash
xaudit config set theme amber
```

**Premium Logging:** Run any command with `--debug` to enable high-fidelity, stylized logs powered by `rich`.

---

## ⚡ High-Performance Batch Analysis

XAudit features a **Dual-Mode analysis engine** designed for both speed and clarity:

- **Smooth Mode:** For 1–5 traces, analysis is sequential with live status indicators.
- **Parallel Mode:** For batches of >5 traces, XAudit automatically uses **multi-core parallel processing** via `ProcessPoolExecutor`.

Batch analysis is roughly **5–10x faster** on multi-core systems.

---

## 🧪 Researcher Controls

Fine-tune the auditing sensitivity globally to match your specific agent architecture:

```bash
# Set primacy dominance sensitivity to 30%
xaudit config set primacy_dominance.threshold 0.30

# View all custom settings
xaudit config get
```

---

## Why XAudit?

Current agent observability tools like LangSmith and Langfuse primarily track *whether* an agent crashed.

**XAudit** analyzes *how* your agent reasoned — and whether that reasoning remained rational throughout the task.

An agent that completes a task by anchoring on its first retrieval — ignoring 18 steps of contradicting evidence — is failing silently. XAudit surfaces these invisible reasoning failures.

---

## What It Detects

| Detector | What It Finds | Method | Mathematical Field |
|---|---|---|---|
| **Primacy Dominance** | Over-reliance on first retrieval | Wasserstein-1 Distance | Optimal transport theory |
| **Query Entropy Collapse** | Tool diversity narrowing over time | Rényi Entropy (α=2) | Information theory |
| **Strategy Persistence** | Continuing failed strategies past pivot point | Failure event tracking | Empirical analysis |
| **Cyclic Redundancy** | Repeated action sequences without progress | Lempel-Ziv Complexity | Algorithmic information theory |
| **Context Decay** | Reasoning quality declining mid-session | CUSUM changepoint | Statistical process control |

XAudit applies disambiguation logic to reduce false positives. Each detector is aware of its own limitations — see [Detector Limitations](#detector-limitations) for details.

---

## Mathematical Approach

XAudit uses deterministic mathematical analysis instead of LLM-as-a-judge evaluation:

- ✅ **Zero API cost** — no LLM calls during analysis
- ✅ **Reproducible** — same trace produces identical scores every time
- ✅ **Fast** — parallelizable across CPU cores
- ✅ **Framework-agnostic** — works on any agent trace format

Methods drawn from **optimal transport theory**, **information theory**, **algorithmic complexity theory**, and **statistical process control**.

No probabilistic system judging another probabilistic system. Mathematical proxies that detect patterns, not opinions.

---

## Try Your Own Traces

```bash
xaudit analyze --trace ./my_trace.json
xaudit analyze --trace ./traces/
xaudit analyze --trace ./trace.json --format langsmith
xaudit analyze --trace ./trace.json --output ./results/
```

---

## Output

XAudit generates two primary artifacts in your output directory:

**`behavior_report.json`** — Detailed results including scores, evidence, and mathematical interpretations.

```json
{
  "overall_rationality_score": 0.815,
  "biases_detected": ["primacy_dominance"],
  "detectors": {
    "primacy_dominance": {
      "detected": true,
      "score": 0.793,
      "threshold": 0.35,
      "evidence": {
        "wasserstein_distance": 0.793,
        "weight_distribution": [0.85, 0.05, 0.05, 0.05]
      },
      "interpretation": "Agent heavily weighted first retrieval. Wasserstein distance from uniform: 0.79. Primacy dominance detected."
    }
  }
}
```

**`behavioral_timeline.png`** — A visual timeline annotated with pattern markers and behavioral zone highlights.

---

## Supported Formats

| Format | Usage |
|---|---|
| Raw JSON (native) | `xaudit analyze --trace ./trace.json` |
| LangSmith export | `xaudit analyze --trace ./trace.json --format langsmith` |
| Langfuse export | `xaudit analyze --trace ./trace.json --format langfuse` |

Format is auto-detected by default. Use `--format` to manually override.

**Raw JSON schema:**

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

## Python API

```python
import json
from xaudit import analyze_trace

with open("my_trace.json") as f:
    trace = json.load(f)

report = analyze_trace(trace)
print(f"Rationality: {report['overall_rationality_score']}")
print(f"Patterns: {report['biases_detected']}")

if report["overall_rationality_score"] < 0.5:
    print("Agent reasoning quality is low")
```

---

## How the Demo Works

The demo analyzes a pre-packaged research agent trace. In this scenario, the agent was tasked with researching RAG evaluation best practices.

- **Step 0:** Retrieved a paper on transformer attention.
- **Steps 1–18:** Retrieved papers on LSTMs, RNNs, and other architectures.
- **Step 19:** Wrote a final answer — 93% similar to step 0, completely ignoring 18 steps of new data.

XAudit flags this as **primacy dominance**. Although the agent "completed" the task without crashing, its reasoning process was flawed.

---

## Generate Synthetic Traces

Only required for developer testing or scenario generation.

```bash
pip install -e ".[generate]"
cp .env.example .env
# Open .env and insert your Gemini API key
```

> [!TIP]
> Get a free key at **[Google AI Studio](https://aistudio.google.com/apikey)**. Having trouble? See the troubleshooting guide in [INSTALL.md](INSTALL.md#troubleshooting-the-generator).

```bash
xaudit generate --scenarios primacy_dominance,strategy_persistence --output ./runs/
```

Available scenarios: `primacy_dominance`, `query_entropy_collapse`, `strategy_persistence`, `cyclic_redundancy`, `context_decay`, `clean`, `all`.

---

## Detector Limitations

XAudit uses mathematical proxies to detect behavioral patterns. We prioritize transparency about where these heuristics might vary:

| Detector | Known Limitation | Disambiguation Applied |
|---|---|---|
| Primacy Dominance | Similarity metrics cannot distinguish copying from criticizing | Stance divergence scoring |
| Query Entropy Collapse | Deep dives sometimes naturally narrow tool usage | Success-rate gating |
| Strategy Persistence | Rational retries can resemble persistence | Input variation analysis |
| Cyclic Redundancy | Structured iteration (e.g., scraping) can appear periodic | Output entropy gating |
| Context Decay | Task difficulty can mimic performance decay | Baseline-relative CUSUM |

XAudit provides heuristic signals, not definitive psychological verdicts.

---

## Architecture

```
trace.json → format_detect → adapter → AgentTrace
                                            ↓
                               ┌────────────┼────────────┐
                               ↓            ↓            ↓
                          detectors      metrics    visualize.py
                               ↓            ↓            ↓
                      behavior_report.json      behavioral_timeline.png
```

---

## Roadmap

**v0.2.0** (planned):

- Kaplan-Meier survival analysis for strategy lifespan modeling
- Bayesian Online Changepoint Detection (BOCPD) for real-time degradation alerts
- Recurrence Quantification Analysis (RQA) for behavioral phase space analysis
- Granger causality for tool interaction dependencies

---

## Ready to Contribute?

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions and guidelines on adding new detectors or format adapters.

Found a bug? [Open an issue](https://github.com/Aditya-Khankar/xaudit/issues).

---

## License

MIT — Aditya Khankar

---

## About

Behavioral auditing for autonomous agents — detects primacy dominance, query entropy collapse, strategy persistence, cyclic redundancy, and context decay in agent traces.

**Topics:** `agents` `autonomous-agents` `ai-safety` `llm-evaluation` `reasoning-quality` `information-theory` `optimal-transport` `langchain` `langgraph` `agent-evals`
