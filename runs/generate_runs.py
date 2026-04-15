"""Generates synthetic agent traces using LangGraph + Gemini SDK (v2).
Called by `xaudit generate` command. Never called by analyze.
"""

import json
import os
import time
from pathlib import Path


def run_generation(scenarios: str, output_dir: str, api_key: str) -> None:
    """Generate trace files for specified scenarios using the new google-genai SDK."""
    from google import genai
    from runs.bias_scenarios import SCENARIOS

    # Initialize the modern Gemini Client
    client = genai.Client(api_key=api_key)

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
            trace = _run_agent(scenario, client)
            out_path = Path(output_dir) / f"generated_{scenario_name}.json"
            with open(out_path, "w") as f:
                json.dump(trace, f, indent=2)
            print(f"  Saved: {out_path}")
        except Exception as e:
            # We provide helpful guidance for common API issues without exposing the raw key-containing trace
            err_msg = str(e).lower()
            if "quota" in err_msg or "429" in err_msg:
                print(f"  [!] Failed to generate {scenario_name}: API quota exceeded.")
            elif "permission" in err_msg or "403" in err_msg or "not found" in err_msg:
                print(f"  [!] Failed to generate {scenario_name}: API key permissions or model access issue.")
            else:
                print(f"  [!] Failed to generate {scenario_name}. Please check your GEMINI_API_KEY.")


def _run_agent(scenario: dict, client) -> dict:
    """Run a simulated agent on a scenario and capture the trace."""
    events = []
    step = 0

    task = scenario["task"]
    tools = scenario.get("tools", ["web_search", "document_reader"])
    
    # We use gemini-2.0-flash as the best-performing small model for this key
    model_name = "models/gemini-2.0-flash"

    # Simple sequential agent: LLM decides, tools execute
    contents = [task]

    for _ in range(scenario.get("max_steps", 15)):
        start = time.time()
        try:
            # Use the new SDK's generate_content
            response = client.models.generate_content(
                model=model_name,
                contents=contents
            )
            llm_output = response.text if response.text else ""
        except Exception:
            raise

        latency = (time.time() - start) * 1000

        # Record LLM call
        events.append({
            "step": step,
            "type": "llm_call",
            "tool": None,
            "input": {"messages": len(contents)},
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

        contents.append(llm_output)

        if step >= scenario.get("max_steps", 15):
            break

    return {
        "trace_id": f"generated_{scenario['name']}_{int(time.time())}",
        "agent_name": "gemini_2_agent",
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
