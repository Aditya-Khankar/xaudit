"""CLI entry point for XAudit.

analyze command: zero API keys required.
generate command: requires GEMINI_API_KEY environment variable.
demo command: instant one-command trial experience.
"""

import json
import logging
import os
import sys
import contextlib
from pathlib import Path

import click
from rich.console import Console

from xaudit.recorder.format_adapters import get_adapter
from xaudit.utils.format_detect import detect_format
from xaudit.utils.validators import TraceValidationError
from xaudit.utils.paths import PathTraversalError
from xaudit.utils.logger import setup_logger
from xaudit.report.builder import build_report
from xaudit.report.visualize import generate_timeline

# P0: file size limit — check before json.load() to prevent OOM
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# P0: batch size limit — prevent resource exhaustion
MAX_BATCH_SIZE = 50


@click.group()
@click.version_option(version="0.1.0", prog_name="xaudit")
def cli():
    """XAudit — behavioral auditing for autonomous agents."""
    pass



def _analyze_single_file(trace_file: Path, fmt: str, output_dir: str, no_viz: bool, json_only: bool, theme_name: str | None) -> dict | None:
    """Core analysis logic for a single file. Picklable for multiprocessing."""
    from xaudit.themes import get_theme
    from xaudit.utils.logger import setup_logger
    from xaudit.recorder.format_adapters import get_adapter
    from xaudit.utils.format_detect import detect_format
    from xaudit.report.builder import build_report
    from xaudit.report.visualize import generate_timeline
    from xaudit.utils.validators import TraceValidationError
    from xaudit.utils.paths import PathTraversalError

    t = get_theme(theme_name)
    logger = setup_logger(output_dir=output_dir, debug=False)

    try:
        with open(trace_file) as f:
            raw = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read {trace_file.name}: {e}")
        return None

    try:
        resolved_fmt = fmt if fmt != "auto" else detect_format(raw)
        adapter = get_adapter(resolved_fmt)
        trace = adapter.adapt(raw)
        
        # Report building
        report = build_report(trace, output_dir)
        
        # Visualization
        png_path = "skipped"
        if not no_viz:
            try:
                png_path = generate_timeline(trace, report, output_dir)
            except Exception:
                pass
        
        from pathlib import Path
        import os

        out_p = Path(output_dir)
        rel_report = (out_p / "behavior_report.json").as_posix()
        if not rel_report.startswith("."):
            rel_report = f"./{rel_report}"

        rel_png = png_path
        if png_path != "skipped":
            try:
                rel_png = "./" + Path(png_path).relative_to(Path.cwd()).as_posix()
            except ValueError:
                rel_png = "./" + Path(os.path.relpath(png_path)).as_posix()

        return {
            "file": trace_file.name,
            "report_path": rel_report,
            "png_path": rel_png,
            "report": report
        }
    except Exception as e:
        logger.error(f"Analysis failed for {trace_file.name}: {e}")
        return None


@cli.command()
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
              help="Enable debug logging to console and xaudit.log.")
@click.option("--theme", "theme_name", default=None,
              type=click.Choice(["amber", "nebula"]),
              help="Visual theme. Default: amber. Saved with: xaudit config set theme <name>")
def analyze(trace_path, fmt, output_dir, no_viz, json_only, debug, theme_name):
    """Analyze agent traces for behavioral reasoning patterns.

    Automatically switches to parallel processing for >5 files.
    """
    from xaudit.themes import get_theme
    t = get_theme(theme_name)

    logger = setup_logger(output_dir=output_dir if debug else None, debug=debug)
    logger.info(f"Starting analysis: {trace_path}")

    path = Path(trace_path)

    # Collect trace files
    if path.is_dir():
        trace_files = sorted(path.glob("*.json"))
        if not trace_files:
            t.print_error(f"No JSON files found in {trace_path}")
            sys.exit(1)
    elif path.is_file():
        trace_files = [path]
    else:
        t.print_error(f"Path not found: {trace_path}")
        sys.exit(1)

    # P0: Batch size limit
    if len(trace_files) > MAX_BATCH_SIZE:
        t.print_error(
            f"Found {len(trace_files)} trace files. Batch limit is {MAX_BATCH_SIZE}."
        )
        sys.exit(1)

    # DUAL-MODE LOGIC: Sequential (<=5) vs Parallel (>5)
    if len(trace_files) <= 5:
        # Sequential Mode: High-fidelity UI with status indicators
        for trace_file in trace_files:
            if not json_only:
                t.console.print(f"\n[bold]Analyzing:[/bold] {trace_file.name}")
            
            with t.status("Processing trace...") if not json_only else contextlib.nullcontext():
                res = _analyze_single_file(trace_file, fmt, output_dir, no_viz, json_only, theme_name)
            
            if res and not json_only:
                _print_results(res["report"], t)
                t.print_output_paths(res["png_path"], res["report_path"])
            elif not res:
                t.print_error(f"Analysis failed for {trace_file.name}. Check xaudit.log for details.")
    else:
        # Parallel Mode: Raw speed for batch processing
        from concurrent.futures import ProcessPoolExecutor
        import functools

        t.console.print(f"\n[bold]Batch analysis starting:[/bold] {len(trace_files)} files (Parallel Mode)")
        
        worker = functools.partial(_analyze_single_file, fmt=fmt, output_dir=output_dir, 
                                   no_viz=no_viz, json_only=True, theme_name=theme_name)
        
        with t.status(f"Analyzing {len(trace_files)} traces across CPU cores..."):
            with ProcessPoolExecutor() as executor:
                results = list(executor.map(worker, trace_files))
        
        success_count = len([r for r in results if r is not None])
        t.console.print(f"\n[bold green][OK] Batch Complete:[/bold green] {success_count}/{len(trace_files)} traces analyzed.")
        t.console.print(f"Results saved to: [dim]{output_dir}[/dim]")


def _print_results(report: dict, theme) -> None:
    """Print results using the active theme."""
    score = report.get("overall_rationality_score", 1.0)
    patterns = report.get("patterns_detected", [])
    detectors = report.get("detectors", {})

    theme.print_score(score, patterns)

    # Print interpretation for detected patterns
    for pattern_name in patterns:
        interp = detectors.get(pattern_name, {}).get("interpretation", "")
        if interp:
            theme.print_pattern_detail(pattern_name, interp)

    theme.console.print()
    theme.print_detector_table(detectors)


@cli.command()
@click.option("--theme", default=None,
              type=click.Choice(["amber", "nebula"]),
              help="Visual theme.")
@click.option("--debug", is_flag=True, default=False,
              help="Enable debug logging to console and xaudit.log.")
def demo(theme, debug):
    """Run XAudit on a pre-packaged sample trace."""
    from importlib import resources
    from xaudit.themes import get_theme
    from xaudit.utils.logger import setup_logger
    t = get_theme(theme)
    t.print_banner()
    
    logger = setup_logger(output_dir="./demo_output/", debug=debug)

    sample_ref = resources.files("xaudit") / "sample_traces" / "run_demo.json"

    try:
        with resources.as_file(sample_ref) as sample_path:
            if not sample_path.exists():
                t.print_error("Sample traces not found. Reinstall XAudit: pip install -e .")
                sys.exit(1)

            with open(sample_path) as f:
                raw = json.load(f)
    except Exception:
        t.print_error("Could not load sample trace. Please reinstall XAudit.")
        sys.exit(1)

    resolved_fmt = detect_format(raw)
    adapter = get_adapter(resolved_fmt)
    trace = adapter.adapt(raw)

    output_dir = "./demo_output/"
    with t.status("Running behavioral detectors..."):
        report = build_report(trace, output_dir)

    _print_results(report, t)

    png_path = "skipped"
    try:
        # P0: Fix Windows absolute paths in demo output — make relative with forward slashes
        raw_png_path = generate_timeline(trace, report, output_dir)
        png_path = "./" + str(Path(raw_png_path).relative_to(Path.cwd())).replace("\\", "/")
    except Exception:
        pass

    report_display_path = "./" + str((Path(output_dir) / "behavior_report.json")).replace("\\", "/")
    t.print_output_paths(png_path, report_display_path)
    t.print_footer()


@cli.command(hidden=True)
@click.option("--scenarios", default="all",
              help="Comma-separated bias scenarios to generate.")
@click.option("--output", "output_dir", default="./runs/",
              help="Output directory for generated traces.")
@click.option("--theme", default=None,
              type=click.Choice(["amber", "nebula"]))
def generate(scenarios, output_dir, theme):
    """Generate synthetic agent traces using LangGraph + Gemini."""
    from xaudit.themes import get_theme
    t = get_theme(theme)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        t.print_error("GEMINI_API_KEY environment variable not set.")
        sys.exit(1)

    try:
        # P0: Add project root to path so 'runs' can be found
        project_root = str(Path(__file__).parent.parent)
        if project_root not in sys.path:
            sys.path.append(project_root)
            
        from runs.generate_runs import run_generation
        run_generation(scenarios=scenarios, output_dir=output_dir, api_key=api_key)
        t.print_success(f"Traces generated successfully in {output_dir}")
    except Exception:
        t.print_error("Generation failed. Check API key and quota.")
        sys.exit(1)


@cli.command()
@click.argument("action", type=click.Choice(["set", "get", "reset"]))
@click.argument("key", required=False)
@click.argument("value", required=False)
def config(action, key, value):
    """Manage XAudit configuration.

    Examples:
        xaudit config set theme amber
        xaudit config set primacy_dominance.threshold 0.5
        xaudit config get theme
        xaudit config reset
    """
    from xaudit.themes import (
        save_config, load_config,
        CONFIG_PATH, THEME_COLORS
    )
    from rich.console import Console
    console = Console()
    config = load_config()

    if action == "set":
        if not key or not value:
            console.print("[red]Error: Action 'set' requires both key and value.[/red]")
            return

        if key == "theme":
            if value not in THEME_COLORS:
                console.print(
                    f"[red]Unknown theme '{value}'. "
                    f"Choose: {', '.join(THEME_COLORS.keys())}[/red]"
                )
                return
            config["theme"] = value
            save_config(config)
            console.print(f"[green]Theme set to '{value}'.[/green]")
        
        elif ".threshold" in key:
            try:
                detector_name = key.split(".")[0]
                threshold_val = float(value)
                if not (0.0 <= threshold_val <= 1.0):
                    raise ValueError("Threshold must be between 0.0 and 1.0")
                
                if "thresholds" not in config:
                    config["thresholds"] = {}
                config["thresholds"][detector_name] = threshold_val
                save_config(config)
                console.print(f"[green]Threshold for '{detector_name}' set to {threshold_val}.[/green]")
            except ValueError as e:
                console.print(f"[red]Invalid threshold value: {e}[/red]")
        else:
            console.print(f"[red]Unknown key '{key}'. Available: theme, <detector>.threshold[/red]")

    elif action == "get":
        if not key:
            console.print(json.dumps(config, indent=2))
            return
        
        if key == "theme":
            console.print(f"theme: {config.get('theme', 'amber')}")
        elif ".threshold" in key:
            detector_name = key.split(".")[0]
            val = config.get("thresholds", {}).get(detector_name)
            if val is not None:
                console.print(f"{key}: {val}")
            else:
                console.print(f"[yellow]No custom threshold for '{detector_name}'. Using default.[/yellow]")
        else:
            console.print(f"[red]Unknown key '{key}'.[/red]")

    elif action == "reset":
        if CONFIG_PATH.exists():
            CONFIG_PATH.unlink()
        console.print("[green]Config reset to defaults.[/green]")


@cli.command()
def themes():
    """Preview all available themes."""
    from xaudit.themes import THEME_COLORS, get_theme
    from rich.console import Console
    console = Console()

    console.print("\nAvailable themes:\n")
    for name in THEME_COLORS.keys():
        t = get_theme(name)
        status = "(default)" if name == "amber" else ""
        console.print(f"  [bold]{name}[/bold] {status}")
        t.print_score(0.81, ["primacy_dominance"])
        console.print()

    console.print("Set a theme:")
    console.print("  xaudit config set theme amber\n")
    console.print("Use once:")
    console.print("  xaudit demo --theme nebula\n")


if __name__ == "__main__":
    cli()
