# Cognidrift

> Your agent completed the task. Did it reason well?

Detect anchoring bias, confirmation bias, sunk cost fallacy, action loops, and performance degradation in any autonomous agent trace. 

![Python](https://img.shields.io/badge/python-3.11+-blue?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)
![Formats](https://img.shields.io/badge/formats-LangSmith%20%7C%20Langfuse%20%7C%20Raw%20JSON-orange?style=flat-square)

```bash
$ cognidrift demo

Overall rationality score: 0.81
Biases detected: anchoring
  → Answer was 93% similar to first retrieval. Later retrievals averaged 9% similarity.

Detector            Score   Threshold   Detected
anchoring           0.926   0.60        YES
confirmation_bias   0.000   0.60        no
sunk_cost           0.000   0.50        no
loop_detection      0.242   0.65        no
degradation         0.000   0.40        no

Timeline saved: ./demo_output/behavioral_timeline.png
Report saved:   ./demo_output/behavior_report.json
```

---

## Quickstart

**Requirements:** Python 3.11+

```bash
git clone https://github.com/Aditya-Khankar/cognidrift
cd cognidrift
pip install -e .
cognidrift demo
```

No API key or configuration required. Results in under 60 seconds.

> Having trouble? See [INSTALL.md](INSTALL.md) for platform-specific instructions.

---

---

## 🎨 Themes & Logging

Cognidrift is built for a premium developer experience. It ships with two visual themes and professional stylized logging.

```bash
cognidrift demo --theme amber       # warm gold on dark (default)
cognidrift demo --theme nebula      # legendary purple & plain text
```

**Save your preferred theme:**

```bash
cognidrift config set theme amber
```

**Premium Logging:**
Run any command with `--debug` to enable high-fidelity, stylized logs powered by `rich`.

---

## ⚡ High-Performance Batch Analysis

Cognidrift features a **Dual-Mode analysis engine** designed for both speed and clarity:

- **Smooth Mode:** For 1–5 traces, analysis is sequential with live status indicators.
- **Parallel Mode:** For batches of >5 traces, Cognidrift automatically uses **multi-core parallel processing** via `ProcessPoolExecutor`.

Batch analysis is roughly **5–10x faster** on multi-core systems.

---

## 🧪 Researcher Controls

Fine-tune the auditing sensitivity globally to match your specific agent architecture:

```bash
# Set anchoring sensitivity to 55%
cognidrift config set anchoring.threshold 0.55

# View all custom settings
cognidrift config get
```

## Why Cognidrift?

Current agent observability tools like LangSmith and Langfuse primarily track *whether* an agent crashed.  
**Cognidrift** analyzes *how* your agent reasoned—and whether that reasoning remained rational throughout the task.

An agent that completes a task by "anchoring" on its first result—ignoring 15 steps of contradicting evidence—is failing silently. Cognidrift surfaces these invisible reasoning failures.

---

## What it detects?

| Bias | Description | Method |
|---|---|---|
| **Anchoring** | Over-relies on first retrieval; ignores later evidence | TF-IDF cosine similarity |
| **Confirmation bias** | Tool usage narrows — seeking confirmation over exploration | Linear regression slope |
| **Sunk cost** | Persists with failing strategy past rational pivot point | Step counter + input variation |
| **Loop detection** | Repeating action sequences with no forward progress | Autocorrelation |
| **Degradation** | Efficiency declining mid-session — context window breakdown | CUSUM changepoint |

Cognidrift applies disambiguation logic to reduce false positives. Each detector is aware of its own limitations—see [Detector limitations](#detector-limitations) for details.

---

## Try your own traces

```bash
cognidrift analyze --trace ./my_trace.json
cognidrift analyze --trace ./traces/
cognidrift analyze --trace ./trace.json --format langsmith
cognidrift analyze --trace ./trace.json --output ./results/
```

---

## Output

Cognidrift generates two primary artifacts in your output directory:

**`behavior_report.json`** — Detailed results including scores, evidence, and mathematical interpretations.

```json
{
  "overall_rationality_score": 0.815,
  "biases_detected": ["anchoring"],
  "detectors": {
    "anchoring": {
      "detected": true,
      "score": 0.926,
      "threshold": 0.60,
      "evidence": {
        "first_retrieval_similarity_to_answer": 0.926,
        "avg_later_retrieval_similarity": 0.088
      },
      "interpretation": "Answer was 93% similar to first retrieval. Later retrievals averaged 9% similarity. Anchoring pattern detected."
    }
  }
}
```

**`behavioral_timeline.png`** — A visual timeline annotated with bias markers and behavioral zone highlights.

---

## Supported formats

| Format | Usage |
|---|---|
| Raw JSON (native) | `cognidrift analyze --trace ./trace.json` |
| LangSmith export | `cognidrift analyze --trace ./trace.json --format langsmith` |
| Langfuse export | `cognidrift analyze --trace ./trace.json --format langfuse` |

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
from cognidrift import analyze_trace

with open("my_trace.json") as f:
    trace = json.load(f)

report = analyze_trace(trace)
print(f"Rationality: {report['overall_rationality_score']}")
print(f"Biases: {report['biases_detected']}")

if report["overall_rationality_score"] < 0.5:
    print("Agent reasoning quality is low")
```

---

## How the demo works?

The demo analyzes a pre-packaged research agent trace. In this scenario, the agent was tasked with researching RAG evaluation best practices.

- **Step 0:** Retrieved a paper on transformer attention.
- **Steps 1–18:** Retrieved papers on LSTMs, RNNs, and other architectures.
- **Step 19:** Wrote a final answer — 93% similar to step 0, completely ignoring 18 steps of new data.

Cognidrift flags this as anchoring. Although the agent "completed" the task without crashing, its reasoning process was flawed.

---

## Generate synthetic traces

Only required for developer testing or scenario generation.

```bash
pip install -e ".[generate]"
cp .env.example .env
```

Open `.env` and insert your Gemini API key. 

> [!TIP]
> Get a free key at **[Google AI Studio](https://aistudio.google.com/apikey)**. Having trouble? See the troubleshooting guide in [INSTALL.md](INSTALL.md#troubleshooting-the-generator).

```bash
cognidrift generate --scenarios anchoring,sunk_cost --output ./runs/
```

Available scenarios: `anchoring`, `confirmation`, `sunk_cost`, `loop`, `degradation`, `clean`, `all`.

---

## Detector limitations

Cognidrift uses mathematical proxies to detect behavioral patterns. We prioritize transparency about where these heuristics might vary:

| Detector | Known Limitation | Disambiguation Applied |
|---|---|---|
| Anchoring | TF-IDF can't distinguish copying from criticizing | Stance divergence scoring |
| Confirmation bias | Deep dives sometimes naturally narrow tool usage | Success-rate gating |
| Sunk cost | Rational retries can look like sunk cost | Input variation analysis |
| Loop detection | Structured iteration (e.g., scraping) can look periodic | Output entropy gating |
| Degradation | Task difficulty can mimic performance decay | Baseline-relative CUSUM |

Cognidrift provides heuristic signals, not definitive psychological verdicts.

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

## Ready to contribute?

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions and guidelines on adding new detectors or format adapters.

Found a bug? [Open an issue](https://github.com/Aditya-Khankar/cognidrift/issues).

---

## License

MIT — Aditya Khankar
