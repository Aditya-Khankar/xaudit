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
