# cognidrift

Behavioral auditing for autonomous agents.

## What it detects

cognidrift analyzes agent traces and surfaces irrational decision patterns —
anchoring on first retrieval results, seeking confirmation over exploration,
continuing failed strategies past the rational pivot point, repeating action
loops, and degrading performance across a session. Each pattern is scored 0–1
with evidence and an overall rationality score.

## Quickstart

```bash
pip install cognidrift
cognidrift demo  # One command, instant output
```

No API key required for analysis.

## Try your own traces

```bash
cognidrift analyze --trace ./my_trace.json
cognidrift analyze --trace ./traces/              # analyze a directory
cognidrift analyze --trace ./trace.json --format langsmith
cognidrift analyze --trace ./trace.json --output ./results/ --debug
```

## Output

Running `cognidrift demo` or `cognidrift analyze` produces:

- **behavior_report.json** — structured results with per-detector scores, evidence, and interpretations
- **behavioral_timeline.png** — annotated visual timeline with bias markers

Example report excerpt:

```json
{
  "overall_rationality_score": 0.22,
  "biases_detected": ["anchoring"],
  "detectors": {
    "anchoring": {
      "detected": true,
      "score": 0.78,
      "threshold": 0.60,
      "version": "1.0.0",
      "evidence": {
        "first_retrieval_similarity_to_answer": 0.78,
        "avg_later_retrieval_similarity": 0.31
      },
      "interpretation": "Answer was 78% similar to first retrieval. Later retrievals averaged 31% similarity. Anchoring pattern detected."
    }
  }
}
```

## Supported formats

- Raw JSON (native format — schema below)
- LangSmith export
- Langfuse export

Format is auto-detected. Override with `--format langsmith | langfuse | raw`.

## Raw JSON schema

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

## Python API

```python
from cognidrift import analyze_trace

report = analyze_trace(trace_dict)
if report["overall_rationality_score"] < 0.5:
    print("Agent reasoning quality is low")
```

## Generate synthetic traces

Requires `GEMINI_API_KEY`. Install with generation dependencies:

```bash
pip install cognidrift[generate]
export GEMINI_API_KEY=your_key_here
cognidrift generate --scenarios anchoring,sunk_cost --output ./runs/
```

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

```
trace.json → format_detect → adapter → AgentTrace
                                           ↓
                              ┌────────────┼────────────┐
                              ↓            ↓            ↓
                         detectors      metrics    visualize.py
                              ↓            ↓            ↓
                         behavior_report.json    behavioral_timeline.png
```

## License

MIT — Aditya Khankar
