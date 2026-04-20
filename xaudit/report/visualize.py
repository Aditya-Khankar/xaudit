"""Behavioral timeline visualization for agent trace analysis."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np

from xaudit.recorder.trace_recorder import AgentTrace
from xaudit.utils.paths import safe_output_path, ensure_output_dir

# ── Lane configuration ─────────────────────────────────────────────────────────

LANE_ORDER = ["retrieval", "tool_call", "llm_call", "error"]

LANE_DISPLAY = {
    "tool_call": "Tool call",
    "llm_call":  "LLM call",
    "retrieval": "Retrieval",
    "error":     "Error",
}

LANE_COLORS = {
    "tool_call": "#378ADD",
    "llm_call":  "#7F77DD",
    "retrieval": "#1D9E75",
    "error":     "#E24B4A",
}


# ── Main function ──────────────────────────────────────────────────────────────

def generate_timeline(trace: AgentTrace, report: dict, output_dir: str) -> str:
    """Generate behavioral_timeline.png. Returns the output file path."""

    # ── Build active lanes from actual events ──────────────────────────────────
    active_types = set(e.event_type for e in trace.events)
    active_lanes = [t for t in LANE_ORDER if t in active_types]
    EVENT_Y = {lane: i for i, lane in enumerate(reversed(active_lanes))}
    n_lanes = len(active_lanes)

    # ── Figure layout ──────────────────────────────────────────────────────────
    # Detect how many insight lines we'll need
    detectors = report.get("detectors", {})
    patterns_detected = report.get("patterns_detected", [])
    n_insight_lines = max(len(patterns_detected), 1)

    # Scale info panel height with number of insight lines
    info_height = 0.8 + (n_insight_lines * 0.25)

    # dynamic width based on number of events to prevent label overlap
    width = max(16.0, len(trace.events) * 0.8)
    fig, (ax_main, ax_info) = plt.subplots(
        2, 1,
        figsize=(width, 8.0 + info_height),
        dpi=150,
        gridspec_kw={"height_ratios": [3.5, info_height]},
    )
    fig.patch.set_facecolor("white")
    ax_main.set_facecolor("#fafafa")
    ax_info.set_facecolor("white")
    ax_info.axis("off")

    # ── Extract detector evidence ──────────────────────────────────────────────
    primacy   = detectors.get("primacy_dominance", {})
    strategy  = detectors.get("strategy_persistence", {})
    context   = detectors.get("context_decay", {})
    cyclic    = detectors.get("cyclic_redundancy", {})
    entropy   = detectors.get("query_entropy_collapse", {})

    anchor_source_step = None

    # ── Pattern zone shading ───────────────────────────────────────────────────
    if primacy.get("detected"):
        ev = primacy.get("evidence", {})
        first_step  = ev.get("first_retrieval_step", 0)
        answer_step = ev.get("answer_step", trace.total_steps)
        anchor_source_step = first_step

        ax_main.axvspan(
            first_step, answer_step,
            alpha=0.07, color="#E24B4A", zorder=0
        )
        ax_main.text(
            (first_step + answer_step) / 2,
            n_lanes - 0.25,
            f"pattern zone",
            fontsize=7, color="#A32D2D", fontstyle="italic",
            ha="center", va="top",
        )

    if strategy.get("detected"):
        ev = strategy.get("evidence", {})
        fail_step   = ev.get("first_failure_step")
        change_step = ev.get("strategy_change_step") or trace.total_steps
        if fail_step is not None:
            ax_main.axvspan(
                fail_step, change_step,
                alpha=0.08, color="#F59E0B", zorder=0
            )
            ax_main.text(
                (fail_step + change_step) / 2,
                n_lanes - 0.25,
                f"persistence zone",
                fontsize=7, color="#92400E", fontstyle="italic",
                ha="center", va="top",
            )

    if context.get("detected"):
        ev = context.get("evidence", {})
        cp = ev.get("changepoint_step")
        if cp is not None:
            ax_main.axvline(
                cp, color="#7F77DD",
                linestyle="--", linewidth=1, alpha=0.7
            )
            ax_main.text(
                cp + 0.3,
                n_lanes - 0.25,
                f"decay at {cp}",
                fontsize=7, color="#534AB7", fontstyle="italic",
            )

    # ── Plot events ────────────────────────────────────────────────────────────
    for event in trace.events:
        y = EVENT_Y.get(event.event_type, 0)
        color = LANE_COLORS.get(event.event_type, "#888780")

        is_anchor = (
            event.event_type == "retrieval"
            and anchor_source_step is not None
            and event.step_index == anchor_source_step
        )

        if is_anchor:
            # Gold diamond — anchor source
            ax_main.scatter(
                event.step_index, y,
                marker="D", s=110,
                c="#F59E0B", edgecolors="#B45309",
                linewidths=1.5, zorder=6,
            )
            pass # Removed redundant annotation that overlaps with pattern zone

        elif not event.success:
            ax_main.scatter(
                event.step_index, y,
                marker="X", s=80,
                c="#E24B4A", edgecolors="#991B1B",
                linewidths=1, zorder=5,
            )

        else:
            ax_main.scatter(
                event.step_index, y,
                marker="o", s=55,
                c=color, edgecolors=color,
                linewidths=0, alpha=0.85, zorder=5,
            )

    # ── Similarity comparison bars (primacy dominance) ─────────────────────────
    if primacy.get("detected"):
        ev          = primacy.get("evidence", {})
        first_sim   = ev.get("first_retrieval_similarity_to_answer", 0)
        later_sim   = ev.get("avg_later_retrieval_similarity", 0)

        bar_x     = trace.total_steps + 1.2
        bar_width = 0.45
        bar_height_max = n_lanes * 0.35

        # First retrieval bar (red)
        ax_main.barh(
            n_lanes * 0.65,
            first_sim * bar_height_max,
            height=0.22,
            left=bar_x,
            color="#E24B4A", alpha=0.85,
        )
        ax_main.text(
            bar_x + first_sim * bar_height_max + 0.05,
            n_lanes * 0.65,
            f"{first_sim:.0%}",
            va="center", fontsize=8, color="#A32D2D", fontweight="bold",
        )
        ax_main.text(
            bar_x,
            n_lanes * 0.65 + 0.15,
            "step 0",
            va="bottom", fontsize=7, color="#A32D2D",
        )

        # Later retrievals bar (green)
        ax_main.barh(
            n_lanes * 0.35,
            later_sim * bar_height_max,
            height=0.22,
            left=bar_x,
            color="#1D9E75", alpha=0.85,
        )
        ax_main.text(
            bar_x + later_sim * bar_height_max + 0.05,
            n_lanes * 0.35,
            f"{later_sim:.0%}",
            va="center", fontsize=8, color="#0F6E56", fontweight="bold",
        )
        ax_main.text(
            bar_x,
            n_lanes * 0.35 + 0.15,
            "later",
            va="bottom", fontsize=7, color="#0F6E56",
        )

        ax_main.text(
            bar_x,
            n_lanes * 0.90,
            "answer\nsimilarity",
            va="top", fontsize=7, color="#5F5E5A",
        )

    # ── Axes formatting ────────────────────────────────────────────────────────
    ax_main.set_yticks(list(EVENT_Y.values()))
    ax_main.set_yticklabels(
        [LANE_DISPLAY[lane] for lane in reversed(active_lanes)],
        fontsize=10, color="#5F5E5A",
    )
    ax_main.set_xlabel("step", fontsize=9, color="#5F5E5A")

    # Integer x-axis ticks only
    max_step = trace.total_steps
    step_interval = max(1, max_step // 10)
    ax_main.set_xticks(range(0, max_step + 1, step_interval))
    ax_main.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: str(int(x))))

    ax_main.set_xlim(-0.8, trace.total_steps + (3.5 if primacy.get("detected") else 0.8))
    ax_main.set_ylim(-0.55, n_lanes - 0.45 + 0.55)

    ax_main.tick_params(colors="#888780", labelsize=9)
    ax_main.grid(axis="x", linestyle=":", alpha=0.25, color="#888780")
    ax_main.spines["top"].set_visible(False)
    ax_main.spines["right"].set_visible(False)
    ax_main.spines["left"].set_color("#D3D1C7")
    ax_main.spines["bottom"].set_color("#D3D1C7")

    # ── Rationality score (top right) ──────────────────────────────────────────
    score = report.get("overall_rationality_score", 1.0)
    score_color = (
        "#16a34a" if score > 0.7
        else "#d97706" if score > 0.4
        else "#dc2626"
    )
    ax_main.text(
        0.99, 0.97,
        f"rationality  {score:.2f}",
        transform=ax_main.transAxes,
        ha="right", va="top",
        fontsize=11, color=score_color, fontweight="bold",
    )

    # ── Legend ─────────────────────────────────────────────────────────────────
    legend_elements = []

    # Anchor source — only if primacy dominance detected
    if anchor_source_step is not None:
        legend_elements.append(
            mlines.Line2D([0], [0], marker="D", color="w",
                          markerfacecolor="#F59E0B", markeredgecolor="#B45309",
                          markersize=8, label="anchor source")
        )

    # Event type dots — only for types present in this trace
    if "retrieval" in active_types:
        legend_elements.append(
            mlines.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor="#1D9E75", markersize=7, label="Retrieval")
        )
    if "tool_call" in active_types:
        legend_elements.append(
            mlines.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor="#378ADD", markersize=7, label="Tool call")
        )
    if "llm_call" in active_types:
        legend_elements.append(
            mlines.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor="#7F77DD", markersize=7, label="LLM call")
        )

    # Failure — only if any failure events exist
    if any(not e.success for e in trace.events):
        legend_elements.append(
            mlines.Line2D([0], [0], marker="X", color="w",
                          markerfacecolor="#E24B4A", markersize=7, label="Error")
        )

    # Pattern zone — only if any pattern zone shading exists
    if any(d.get("detected") for d in detectors.values()):
        legend_elements.append(
            mpatches.Patch(facecolor="#E24B4A", alpha=0.15, label="pattern zone")
        )

    ax_main.legend(
        handles=legend_elements,
        loc="lower right",
        fontsize=8,
        framealpha=0.92,
        edgecolor="#D3D1C7",
        ncol=min(3, len(legend_elements)),
    )

    # ── Insight strip (below chart) ────────────────────────────────────────────
    metrics  = report.get("metrics", {})
    efficiency = metrics.get("efficiency", {}).get("value", 0)

    insight_lines = []

    if primacy.get("detected"):
        ev         = primacy.get("evidence", {})
        first_sim  = ev.get("first_retrieval_similarity_to_answer", 0)
        later_sim  = ev.get("avg_later_retrieval_similarity", 0)
        insight_lines.append(
            f"primacy_dominance: Answer was {first_sim:.0%} similar to first retrieval. "
            f"Later retrievals averaged {later_sim:.0%} similarity. Primacy dominance pattern detected."
        )

    if entropy.get("detected"):
        ev = entropy.get("evidence", {})
        insight_lines.append(
            f"query_entropy_collapse: {entropy.get('interpretation', 'Single tool dominates completely. No diversity. Query entropy collapse detected.')}"
        )

    if strategy.get("detected"):
        ev    = strategy.get("evidence", {})
        steps = ev.get("steps_after_failure", 0)
        insight_lines.append(
            f"strategy_persistence: {steps} steps continued after first failure before pivot."
        )

    if cyclic.get("detected"):
        score_val = cyclic.get("score", 0.0)
        insight_lines.append(
            f"cyclic_redundancy: Redundancy score {score_val:.3f}. Repetitive loop pattern detected."
        )

    if context.get("detected"):
        ev = context.get("evidence", {})
        cp = ev.get("changepoint_step", "unknown")
        insight_lines.append(
            f"context_decay: Efficiency drop detected at step {cp}."
        )

    if not insight_lines:
        insight_lines.append(
            "No patterns detected — agent reasoning within normal parameters."
        )

    # Render each insight line
    y_pos = 0.96
    line_spacing = min(0.32, 0.92 / max(len(insight_lines), 1))

    for line in insight_lines:
        ax_info.text(
            0.01, y_pos,
            line,
            transform=ax_info.transAxes,
            fontsize=11,
            color="#2C2C2A",
            va="top",
            fontweight="normal",
            wrap=False,
        )
        y_pos -= line_spacing

    # Footer metadata
    ax_info.text(
        0.01, 0.02,
        f"efficiency {efficiency:.0%}  ·  "
        f"{trace.total_steps} steps  ·  "
        f"{report.get('agent_name', 'agent')}  ·  "
        f"xaudit v0.1.0",
        transform=ax_info.transAxes,
        fontsize=9,
        color="#B4B2A9",
        va="top",
    )

    # ── Save ───────────────────────────────────────────────────────────────────
    plt.tight_layout(h_pad=0.8)

    abs_output = ensure_output_dir(output_dir)
    out_path   = safe_output_path(abs_output, "behavioral_timeline.png")
    plt.savefig(out_path, bbox_inches="tight", facecolor="white", dpi=150)
    plt.close()
    return out_path
