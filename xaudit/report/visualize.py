"""Generates behavioral_timeline.png.

Design principles:
- Anchor source event gets a distinct gold marker (diamond shape)
- Bias zone shading is precise, not full-chart
- Insight text rendered below chart, not inside it
- Score displayed using the corrected formula value from report
- Clean white background, 1200x500px for Twitter/README embedding
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np

from xaudit.recorder.trace_recorder import AgentTrace
from xaudit.utils.paths import safe_output_path, ensure_output_dir

EVENT_Y = {
    "tool_call": 2,
    "llm_call": 1,
    "retrieval": 3,
    "error": 0,
}
EVENT_LABELS = {3: "retrieval", 2: "tool call", 1: "llm call", 0: "error"}
LANE_COLORS = {
    "tool_call": "#378ADD",
    "llm_call": "#7F77DD",
    "retrieval": "#1D9E75",
    "error": "#E24B4A",
}


def generate_timeline(trace: AgentTrace, report: dict, output_dir: str) -> str:
    fig, (ax_main, ax_info) = plt.subplots(
        2, 1,
        figsize=(12, 5),
        dpi=150,
        gridspec_kw={"height_ratios": [3.5, 1]},
    )
    fig.patch.set_facecolor("white")
    ax_main.set_facecolor("#fafafa")
    ax_info.set_facecolor("white")
    ax_info.axis("off")

    # --- Bias zone shading (precise, not full-chart) ---
    primacy_dominance = report["detectors"].get("primacy_dominance", {})
    sunk = report["detectors"].get("strategy_persistence", {})
    degrad = report["detectors"].get("context_decay", {})

    anchor_source_step = None

    if primacy_dominance.get("detected"):
        ev = primacy_dominance.get("evidence", {})
        first_step = ev.get("first_retrieval_step", 0)
        answer_step = ev.get("answer_step", trace.total_steps)
        anchor_source_step = first_step
        ax_main.axvspan(
            first_step, answer_step,
            alpha=0.07, color="#E24B4A", zorder=0
        )
        ax_main.text(
            first_step + 0.3,
            3.55,
            f"primacy_dominance zone (steps {first_step}–{answer_step})",
            fontsize=8, color="#A32D2D", fontweight="bold",
        )

    if sunk.get("detected"):
        ev = sunk.get("evidence", {})
        fail_step = ev.get("first_failure_step")
        change_step = ev.get("strategy_change_step") or trace.total_steps
        if fail_step is not None:
            ax_main.axvspan(fail_step, change_step, alpha=0.08, color="#E24B4A", zorder=0)
            ax_main.text(
                (fail_step + change_step) / 2, 3.55,
                f"sunk cost ({change_step - fail_step} steps)",
                fontsize=8, color="#A32D2D", fontweight="bold", ha="center",
            )

    if degrad.get("detected"):
        ev = degrad.get("evidence", {})
        cp = ev.get("changepoint_step")
        if cp is not None:
            ax_main.axvline(cp, color="#7F77DD", linestyle="--", linewidth=1, alpha=0.7)
            ax_main.text(
                cp + 0.2, 3.55,
                f"context_decay at step {cp}",
                fontsize=8, color="#534AB7", fontweight="bold",
            )

    # --- Plot events ---
    for event in trace.events:
        y = EVENT_Y.get(event.event_type, 0)
        color = LANE_COLORS.get(event.event_type, "#888780")
        fail_color = "#E24B4A"

        is_anchor_source = (
            event.event_type == "retrieval"
            and anchor_source_step is not None
            and event.step_index == anchor_source_step
        )

        if is_anchor_source:
            # Gold diamond — anchor source
            ax_main.scatter(
                event.step_index, y,
                marker="D",
                s=100,
                c="#F59E0B",
                edgecolors="#B45309",
                linewidths=1.5,
                zorder=6,
            )
            ax_main.annotate(
                "anchor\nsource",
                xy=(event.step_index, y),
                xytext=(event.step_index + 0.4, y + 0.35),
                fontsize=7,
                color="#92400E",
                fontweight="bold",
            )
        elif not event.success:
            ax_main.scatter(
                event.step_index, y,
                marker="X",
                s=80,
                c=fail_color,
                edgecolors="#991B1B",
                linewidths=1,
                zorder=5,
            )
        else:
            ax_main.scatter(
                event.step_index, y,
                marker="o",
                s=55,
                c=color,
                edgecolors=color,
                linewidths=0,
                alpha=0.85,
                zorder=5,
            )

    # --- Axes formatting ---
    ax_main.set_yticks(list(EVENT_LABELS.keys()))
    ax_main.set_yticklabels(
        [EVENT_LABELS[v] for v in EVENT_LABELS.keys()],
        fontsize=9, color="#5F5E5A"
    )
    ax_main.set_xlabel("step index", fontsize=9, color="#5F5E5A")
    ax_main.set_xlim(-0.8, trace.total_steps + 0.5)
    ax_main.set_ylim(-0.6, 4.0)
    ax_main.tick_params(colors="#888780", labelsize=8)
    ax_main.grid(axis="x", linestyle=":", alpha=0.25, color="#888780")
    ax_main.spines["top"].set_visible(False)
    ax_main.spines["right"].set_visible(False)
    ax_main.spines["left"].set_color("#D3D1C7")
    ax_main.spines["bottom"].set_color("#D3D1C7")

    # --- Rationality score (top right, corrected value) ---
    score = report.get("overall_rationality_score", 1.0)
    score_color = "#16a34a" if score > 0.7 else "#d97706" if score > 0.4 else "#dc2626"
    ax_main.text(
        0.99, 0.97,
        f"rationality  {score:.2f}",
        transform=ax_main.transAxes,
        ha="right", va="top",
        fontsize=10, color=score_color, fontweight="bold",
    )

    # --- Legend ---
    legend_elements = [
        mlines.Line2D([0], [0], marker="D", color="w", markerfacecolor="#F59E0B",
                      markeredgecolor="#B45309", markersize=8, label="anchor source"),
        mlines.Line2D([0], [0], marker="o", color="w", markerfacecolor="#1D9E75",
                      markersize=7, label="retrieval"),
        mlines.Line2D([0], [0], marker="o", color="w", markerfacecolor="#378ADD",
                      markersize=7, label="tool call"),
        mlines.Line2D([0], [0], marker="o", color="w", markerfacecolor="#7F77DD",
                      markersize=7, label="llm call"),
        mlines.Line2D([0], [0], marker="X", color="w", markerfacecolor="#E24B4A",
                      markersize=7, label="failure"),
        mpatches.Patch(facecolor="#E24B4A", alpha=0.15, label="bias zone"),
    ]
    ax_main.legend(
        handles=legend_elements,
        loc="lower right",
        fontsize=8,
        framealpha=0.9,
        edgecolor="#D3D1C7",
        ncol=3,
    )

    # --- Insight strip (below chart) ---
    biases = report.get("biases_detected", [])
    metrics = report.get("metrics", {})
    efficiency = metrics.get("efficiency", {}).get("value", 0)

    if biases:
        insight_parts = []
        if primacy_dominance.get("detected"):
            ev = primacy_dominance.get("evidence", {})
            first_sim = ev.get("first_retrieval_similarity_to_answer", 0)
            later_sim = ev.get("avg_later_retrieval_similarity", 0)
            insight_parts.append(
                f"primacy_dominance: answer {first_sim:.0%} similar to step-0 retrieval "
                f"vs {later_sim:.0%} for later retrievals"
            )
        if sunk.get("detected"):
            ev = sunk.get("evidence", {})
            steps = ev.get("steps_after_failure", 0)
            insight_parts.append(f"sunk cost: {steps} steps after first failure before pivot")
        insight_text = "  ·  ".join(insight_parts)
    else:
        insight_text = "no biases detected — agent reasoning within normal parameters"

    ax_info.text(
        0.01, 0.75,
        insight_text,
        transform=ax_info.transAxes,
        fontsize=8.5,
        color="#5F5E5A",
        va="top",
    )
    ax_info.text(
        0.01, 0.25,
        f"efficiency {efficiency:.0%}  ·  "
        f"{trace.total_steps} steps  ·  "
        f"{report.get('agent_name', 'agent')}  ·  "
        f"xaudit v0.1.0",
        transform=ax_info.transAxes,
        fontsize=7.5,
        color="#B4B2A9",
        va="top",
    )

    plt.tight_layout(h_pad=0.5)

    abs_output = ensure_output_dir(output_dir)
    out_path = safe_output_path(abs_output, "behavioral_timeline.png")
    plt.savefig(out_path, bbox_inches="tight", facecolor="white", dpi=150)
    plt.close()
    return out_path
