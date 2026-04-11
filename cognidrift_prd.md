# cognidrift — Product Requirements Document

**Version:** 1.0  
**Author:** Aditya Khankar  
**Last updated:** April 2026  
**Status:** Active build

---

## 1. Problem statement

Autonomous AI agents fail in two distinct ways:

**Visible failures** — the agent errors out, returns nothing, or crashes. These are caught by existing observability tools (LangSmith, Langfuse).

**Invisible failures** — the agent completes the task but reasons irrationally. It anchors on its first retrieval and ignores contradicting evidence. It seeks confirmation instead of exploration. It continues a failing strategy 20 steps past the rational pivot point. These are not caught by anything. No tool exists that surfaces them.

The cost of invisible failures is higher than visible ones. A crashed agent is obvious. An agent that confidently produces anchored, biased output gets deployed.

**cognidrift solves the second category.**

---

## 2. Target users

**Primary:** Founders and engineers at companies building autonomous agent systems.

**Specific profile:**
- Building on LangGraph, AutoGen, CrewAI, or custom agent frameworks
- Already using LangSmith or Langfuse for observability (or willing to instrument traces)
- Deploying agents for research, customer service, data analysis, or workflow automation
- Concerned about agent reliability at scale — not just task completion but reasoning quality

**Secondary:** ML researchers working on agent evaluation benchmarks.

---

## 3. Core value proposition

"Your agent completed the task. cognidrift tells you whether it reasoned to get there — or got lucky."

Three things cognidrift surfaces that no existing tool does:
1. **Anchoring bias** — did the agent over-rely on its first retrieval result?
2. **Confirmation bias** — did the agent narrow its tool usage to confirm rather than explore?
3. **Sunk cost fallacy** — how many steps did the agent waste after the first failure signal before pivoting?

Output: a quantified behavioral score (0–1 per bias type) + a visual timeline with annotated pattern markers.

---

## 4. Non-goals (v1.0)

- No web dashboard or frontend UI
- No real-time monitoring (batch analysis only)
- No agent framework integration beyond LangGraph for trace generation
- No automatic remediation or agent correction
- No multi-agent system analysis
- No fine-tuning or model-level intervention

These are Phase 2+. v1.0 is purely: take a trace → output behavioral scores + visualization.

---

## 5. Success criteria

v1.0 is complete when:
- [ ] `cognidrift analyze --trace ./run.json` runs end-to-end on all three input formats
- [ ] All 5 detectors produce scores between 0–1 with evidence JSON
- [ ] All 3 metrics produce values
- [ ] `behavior_report.json` is valid and human-readable
- [ ] `visualize.py` produces a PNG timeline with bias annotations
- [ ] 5 sample traces included in `sample_runs/` — runnable without any API key
- [ ] GitHub Actions workflow commits behavior_report.json on every push
- [ ] README passes the Singapore engineer test (no student framing, no unverifiable claims)
- [ ] MIT License filed under Aditya Khankar

---

## 6. User flow

```
1. User has an agent trace (LangSmith export / Langfuse export / raw JSON)

2. User runs:
   cognidrift analyze --trace ./my_run.json

3. cognidrift outputs:
   - Live terminal output (Rich): per-detector scores as they compute
   - behavior_report.json: full structured results
   - behavioral_timeline.png: annotated visual timeline

4. User reads the report:
   - Which biases were detected
   - At which step index they occurred
   - The evidence that triggered the detection
   - Overall rationality score (0–1)
```

---

## 7. behavior_report.json schema

```json
{
  "trace_id": "run_001",
  "agent_name": "research_agent",
  "task": "Find recent papers on RAG evaluation",
  "total_steps": 24,
  "analysis_timestamp": "2026-04-11T09:00:00Z",
  "overall_rationality_score": 0.61,
  "biases_detected": ["anchoring", "sunk_cost"],
  "detectors": {
    "anchoring": {
      "detected": true,
      "score": 0.78,
      "threshold": 0.60,
      "evidence": {
        "first_retrieval_similarity_to_answer": 0.78,
        "later_retrieval_similarity_to_answer": 0.31,
        "first_retrieval_step": 0,
        "answer_step": 23
      },
      "interpretation": "Agent answer was 78% similar to first retrieval result. Later retrievals were largely ignored."
    },
    "confirmation_bias": {
      "detected": false,
      "score": 0.34,
      "threshold": 0.60,
      "evidence": {
        "tool_diversity_slope": -0.12,
        "unique_tools_first_half": 4,
        "unique_tools_second_half": 3,
        "diversity_decay_rate": 0.25
      },
      "interpretation": "Mild tool diversity reduction observed but below detection threshold."
    },
    "sunk_cost": {
      "detected": true,
      "score": 0.71,
      "threshold": 0.50,
      "evidence": {
        "first_failure_step": 8,
        "strategy_change_step": 19,
        "steps_after_failure": 11,
        "total_steps": 24,
        "failure_signals": [8, 10, 13, 15]
      },
      "interpretation": "Agent continued failed search strategy for 11 steps after first failure signal."
    },
    "loop_detection": {
      "detected": false,
      "score": 0.22,
      "threshold": 0.65,
      "evidence": {
        "max_autocorrelation": 0.22,
        "dominant_lag": null,
        "repeated_sequences": []
      },
      "interpretation": "No significant action sequence repetition detected."
    },
    "degradation": {
      "detected": false,
      "score": 0.18,
      "threshold": 0.40,
      "evidence": {
        "changepoint_detected": false,
        "efficiency_trend": "stable",
        "cusum_max": 0.18
      },
      "interpretation": "Agent efficiency remained stable throughout the session."
    }
  },
  "metrics": {
    "efficiency": {
      "value": 0.58,
      "goal_advancing_actions": 14,
      "total_actions": 24,
      "interpretation": "58% of actions directly advanced the goal."
    },
    "exploration_score": {
      "value": 0.67,
      "unique_tools_used": 4,
      "total_tools_available": 6,
      "tool_usage_distribution": {
        "web_search": 12,
        "document_reader": 6,
        "calculator": 3,
        "code_executor": 3
      }
    },
    "recovery_time": {
      "value": 11,
      "unit": "steps",
      "failure_events": 4,
      "avg_steps_to_recovery": 2.75
    }
  }
}
```

---

## 8. Visualization spec (`behavioral_timeline.png`)

**Type:** Horizontal timeline, one row per event type, annotations above/below for detected patterns.

**Layout:**
- X-axis: step index (0 to N)
- Y-axis: event types (tool_call, llm_call, retrieval, error)
- Color coding: green = success, red = error/failure, amber = warning
- Vertical dashed lines at: first failure step, strategy change step, changepoint (if degradation detected)
- Shaded regions: anchoring zone (step 0 to answer), sunk cost zone (failure to pivot)
- Annotations: bias labels ("ANCHORING DETECTED", "SUNK COST: 11 STEPS") at detected regions

**Output:** `behavioral_timeline.png` in the same directory as the input trace.

**Size:** 1200x400px for clean embedding in README and Twitter.

---

## 9. CLI specification

```bash
# Basic usage
cognidrift analyze --trace ./run.json

# Specify format explicitly
cognidrift analyze --trace ./run.json --format langsmith
cognidrift analyze --trace ./run.json --format langfuse
cognidrift analyze --trace ./run.json --format raw

# Output options
cognidrift analyze --trace ./run.json --output ./results/
cognidrift analyze --trace ./run.json --no-viz      # skip PNG generation
cognidrift analyze --trace ./run.json --json-only   # skip terminal output

# Batch analysis
cognidrift analyze --traces ./sample_runs/          # analyze all traces in directory

# Generate sample runs (requires Gemini API key)
cognidrift generate --scenarios all --output ./runs/
cognidrift generate --scenarios anchoring,sunk_cost --output ./runs/

# Info
cognidrift --version
cognidrift --help
```

---

## 10. Format auto-detection logic

```python
def detect_format(trace_dict: dict) -> str:
    if "run_type" in trace_dict or "runs" in trace_dict:
        return "langsmith"
    if "observations" in trace_dict:
        return "langfuse"
    if "events" in trace_dict:
        return "raw"
    raise FormatDetectionError(
        "Could not detect trace format. Use --format to specify explicitly."
    )
```

---

## 11. Raw JSON trace schema (canonical input format)

```json
{
  "trace_id": "string (required)",
  "agent_name": "string (optional)",
  "task": "string (optional — improves interpretation output)",
  "events": [
    {
      "step": "integer (required, 0-indexed)",
      "type": "string (required) — one of: tool_call, llm_call, retrieval, error",
      "tool": "string (required for tool_call/retrieval, null otherwise)",
      "input": "object (required)",
      "output": "object (required)",
      "success": "boolean (required)",
      "latency_ms": "number (optional)"
    }
  ]
}
```

---

## 12. Bias scenario tasks (for generate_runs.py)

Each scenario is a task designed to reliably trigger a specific bias in a LangGraph agent.

| Bias | Task | Why it triggers it |
|---|---|---|
| Anchoring | "Research the current state of RAG evaluation — find the best methods" | First web search result dominates; agent rarely contradicts it |
| Confirmation | "Find evidence that transformer models outperform RNNs on NLP tasks" | Confirmatory framing narrows tool usage |
| Sunk cost | "Debug why this code returns None — here is the code: [deliberately ambiguous]" | First strategy (print debugging) fails; agent persists |
| Loop | "Summarize all papers published on agent evals in 2024" | Open-ended task with no clear terminal condition → repetition |
| Degradation | "Plan a 10-step research agenda on LLM evaluation" (long horizon task) | Performance degrades as context window fills |

---

## 13. GitHub Actions workflow spec

File: `.github/workflows/cognidrift_report.yml`

```yaml
name: cognidrift behavioral report
on: [push]
jobs:
  analyze:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -e .
      - run: cognidrift analyze --traces ./runs/sample_runs/ --output ./report/ --no-viz
      - run: |
          git config user.email "action@github.com"
          git config user.name "cognidrift-bot"
          git add ./report/behavior_report.json
          git diff --cached --quiet || git commit -m "chore: update behavioral report [skip ci]"
          git push
```

No API key needed — sample runs are pre-generated and committed.
