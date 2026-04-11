"""CLI entry point for cognidrift.

analyze command: zero API keys required.
generate command: requires GEMINI_API_KEY environment variable.
demo command: instant one-command trial experience.
"""

import json
import logging
import os
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from cognidrift.recorder.format_adapters import get_adapter
from cognidrift.utils.format_detect import detect_format
from cognidrift.utils.validators import TraceValidationError
from cognidrift.utils.paths import PathTraversalError
from cognidrift.utils.logger import setup_logger
from cognidrift.report.builder import build_report
from cognidrift.report.visualize import generate_timeline

console = Console()

# P0: file size limit — check before json.load() to prevent OOM
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# P0: batch size limit — prevent resource exhaustion
MAX_BATCH_SIZE = 50


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
@click.option("--debug", is_flag=True, default=False,
              help="Enable debug logging to console and cognidrift.log.")
def analyze(trace_path, fmt, output_dir, no_viz, json_only, debug):
    """Analyze an agent trace for behavioral bias patterns.

    Does not require any API key.

    Examples:
        cognidrift analyze --trace ./runs/sample_runs/run_anchoring.json
        cognidrift analyze --trace ./runs/sample_runs/
        cognidrift analyze --trace ./trace.json --format langsmith --output ./results/
    """
    logger = setup_logger(output_dir=output_dir if debug else None, debug=debug)
    logger.info(f"Starting analysis: {trace_path}")

    path = Path(trace_path)

    # Collect trace files
    if path.is_dir():
        trace_files = sorted(path.glob("*.json"))
        if not trace_files:
            console.print(f"[red]No JSON files found in {trace_path}[/red]")
            sys.exit(1)
    elif path.is_file():
        trace_files = [path]
    else:
        console.print(f"[red]Path not found: {trace_path}[/red]")
        sys.exit(1)

    # P0: Batch size limit
    if len(trace_files) > MAX_BATCH_SIZE:
        console.print(
            f"[red]Found {len(trace_files)} trace files. "
            f"Batch limit is {MAX_BATCH_SIZE}. "
            f"Split into smaller directories.[/red]"
        )
        sys.exit(1)

    for trace_file in trace_files:
        # P0: File size check before json.load()
        file_size = trace_file.stat().st_size
        if file_size > MAX_FILE_SIZE:
            console.print(
                f"[red]{trace_file.name} exceeds 10MB limit "
                f"({file_size / 1024 / 1024:.1f}MB) — skipping[/red]"
            )
            logger.warning(f"Skipped oversized file: {trace_file.name} ({file_size} bytes)")
            continue

        if not json_only:
            console.print(f"\n[bold]Analyzing:[/bold] {trace_file.name}")

        try:
            with open(trace_file) as f:
                raw = json.load(f)
        except json.JSONDecodeError as e:
            console.print(f"[red]Invalid JSON in {trace_file.name}: {e}[/red]")
            logger.error(f"JSON parse failed: {trace_file.name}: {e}")
            continue

        # Format detection
        try:
            resolved_fmt = fmt if fmt != "auto" else detect_format(raw)
            logger.info(f"Detected format: {resolved_fmt}")
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            continue

        # Adapt to AgentTrace
        try:
            adapter = get_adapter(resolved_fmt)
            trace = adapter.adapt(raw)
        except (TraceValidationError, ValueError) as e:
            console.print(f"[red]Trace validation failed: {e}[/red]")
            logger.error(f"Adapter failed: {type(e).__name__}: {e}")
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
            except ImportError:
                console.print(
                    "[yellow]Timeline generation requires matplotlib.[/yellow]\n"
                    "  Install: pip install matplotlib"
                )
            except Exception as e:
                logger.error(f"Visualization failed: {type(e).__name__}: {e}")
                console.print(
                    "[yellow]Timeline generation failed (non-fatal).[/yellow]\n"
                    "  Report JSON saved successfully."
                )
                if debug:
                    console.print(f"  Check cognidrift.log for details.")

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
        for bias_name in report["biases_detected"]:
            interpretation = (
                report["detectors"]
                .get(bias_name, {})
                .get("interpretation", "")
            )
            if interpretation:
                console.print(f"  [dim]→ {interpretation}[/dim]")
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
def demo():
    """Run cognidrift on a built-in sample trace — no API key, no setup.

    Analyzes a pre-packaged anchoring bias trace and outputs the full
    behavioral report with visualization. Results saved to ./demo_output/.

    Example:
        cognidrift demo
    """
    from importlib import resources

    console.print("[bold]cognidrift demo — behavioral auditing for autonomous agents[/bold]\n")

    sample_ref = resources.files("cognidrift") / "sample_traces" / "run_anchoring.json"

    try:
        with resources.as_file(sample_ref) as sample_path:
            if not sample_path.exists():
                console.print(
                    "[red]Sample traces not found. "
                    "Reinstall cognidrift: pip install cognidrift[/red]"
                )
                sys.exit(1)

            with open(sample_path) as f:
                raw = json.load(f)
    except Exception:
        console.print(
            "[red]Could not load sample trace. "
            "Reinstall cognidrift: pip install cognidrift[/red]"
        )
        sys.exit(1)

    resolved_fmt = detect_format(raw)
    adapter = get_adapter(resolved_fmt)
    trace = adapter.adapt(raw)

    output_dir = "./demo_output/"
    with console.status("Running behavioral detectors..."):
        report = build_report(trace, output_dir)

    _print_results(report)

    try:
        png_path = generate_timeline(trace, report, output_dir)
        console.print(f"\n[green]Timeline saved:[/green] {png_path}")
    except Exception:
        pass  # Non-fatal — report is the primary output

    console.print(
        f"[green]Report saved:[/green] {output_dir}behavior_report.json\n\n"
        "[bold]Try your own traces:[/bold]\n"
        "  cognidrift analyze --trace <path-to-trace.json>\n\n"
        "[dim]No API key required for analysis. "
        "Generate synthetic traces with: cognidrift generate (requires GEMINI_API_KEY)[/dim]"
    )


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
