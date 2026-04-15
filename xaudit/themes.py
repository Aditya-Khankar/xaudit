"""
XAudit theme system.

Five themes, all cross-platform compatible.
Themes control: banner style, colors, separators, score display, table format.

Usage:
    from xaudit.themes import get_theme, Theme
    theme = get_theme("amber")
    theme.print_banner()
    theme.print_score(0.81, ["primacy_dominance"])
"""

import os
import sys
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich import box as rich_box

ThemeName = Literal["amber", "nebula"]

CONFIG_PATH = Path.home() / ".xaudit" / "config.json"

# ─── ASCII art banner (pyfiglet) ──────────────────────────────────────────────

BANNER_FONTS = {
    "amber":  "slant",
    "nebula": "slant",
}

BANNER_SUBTITLES = {
    "amber":  "behavioral auditing for autonomous agents",
    "nebula": "behavioral auditing for autonomous agents",
}


def _render_banner(theme_name: ThemeName) -> str:
    """Render ASCII art banner for the given theme. Falls back gracefully."""
    font = BANNER_FONTS.get(theme_name)
    if font is None:
        return BANNER_SUBTITLES[theme_name]
    try:
        import pyfiglet
        return pyfiglet.figlet_format("XAudit", font=font)
    except Exception:
        # pyfiglet unavailable or font missing — plain text fallback
        return f"XAudit\n{BANNER_SUBTITLES[theme_name]}"


# ─── Score bar ────────────────────────────────────────────────────────────────

def _score_bar(score: float, width: int = 10, use_unicode: bool = True) -> str:
    """Render a progress bar for the rationality score."""
    filled = round(score * width)
    empty = width - filled
    if use_unicode:
        # Solid block characters for premium look
        return "█" * filled + "░" * empty
    return "#" * filled + "-" * empty


# ─── Theme definitions ────────────────────────────────────────────────────────

@dataclass
class ThemeColors:
    banner:    str = "white"
    subtitle:  str = "dim"
    separator: str = "dim"
    score_good: str = "green"
    score_mid:  str = "yellow"
    score_bad:  str = "red"
    detected:   str = "red"
    clean:      str = "green"
    label:      str = "white"
    dim:        str = "dim"
    info:       str = "blue"
    accent:     str = "white"


THEME_COLORS: dict[ThemeName, ThemeColors] = {
    "amber": ThemeColors(
        banner="bold #e8a87c",
        subtitle="dim #5a4a36",
        separator="#5a4a36",
        score_good="#8fb87a",
        score_mid="#c4a45a",
        score_bad="#c4705a",
        detected="#c4705a",
        clean="#8fb87a",
        label="#e8d5b8",
        dim="#7a6a55",
        info="#7aaac4",
        accent="#e8a87c",
    ),
    "nebula": ThemeColors(
        banner="bold #bc8cff",
        subtitle="dim",
        separator="dim",
        score_good="green",
        score_mid="yellow",
        score_bad="red",
        detected="red",
        clean="green",
        label="white",
        dim="dim",
        info="blue",
        accent="#bc8cff",
    ),
}

SEPARATORS: dict[ThemeName, str] = {
    "amber":  "─" * 70,
    "nebula": "─" * 70,
}

TABLE_BOXES: dict[ThemeName, object] = {
    "amber":  rich_box.SIMPLE,
    "nebula": rich_box.SIMPLE,
}

DETECTED_LABELS: dict[ThemeName, tuple[str, str]] = {
    "amber":  ("DETECTED", "clean"),
    "nebula": ("YES", "no"),
}


# ─── Theme class ──────────────────────────────────────────────────────────────

class Theme:
    def __init__(self, name: ThemeName):
        self.name = name
        self.colors = THEME_COLORS[name]
        self.separator = SEPARATORS[name]
        self.table_box = TABLE_BOXES[name]
        self.detected_label, self.clean_label = DETECTED_LABELS[name]
        # Detect terminal capabilities
        self._supports_unicode = _supports_unicode()
        self._supports_color = _supports_color()

        # If terminal is basic, use simpler layout characters
        if not self._supports_unicode:
            self.separator = self.separator.replace("─", "-").replace("═", "=")
            self.table_box = rich_box.ASCII

        self.console = Console(
            highlight=False,
            force_terminal=self._supports_color,
            color_system="auto" if self._supports_color else None,
        )

    def print_banner(self) -> None:
        """Print the full startup banner."""
        banner = _render_banner(self.name)
        self.console.print(
            Text(banner, style=self.colors.banner)
        )
        self.console.print(
            Text(BANNER_SUBTITLES[self.name], style=self.colors.subtitle)
        )
        self.console.print()

    def print_separator(self) -> None:
        self.console.print(
            Text(self.separator, style=self.colors.separator)
        )

    def print_task_info(self, task: str, agent: str, steps: int, trace_id: str) -> None:
        """Print the task/agent info block."""
        if self.name == "amber":
            # Boxed layout as seen in the premium Claude screenshot
            box_edge = "─" if self._supports_unicode else "-"
            self.console.print(f"┌{box_edge * 72}┐", style=self.colors.separator)
            self.console.print(
                Text("│ ", style=self.colors.separator) +
                Text(f"task    ", style=self.colors.label) +
                Text(task[:60].ljust(62), style=self.colors.dim) +
                Text(" │", style=self.colors.separator)
            )
            self.console.print(
                Text("│ ", style=self.colors.separator) +
                Text(f"agent   ", style=self.colors.label) +
                Text(f"{agent} · {steps} steps · {trace_id}".ljust(62), style=self.colors.dim) +
                Text(" │", style=self.colors.separator)
            )
            self.console.print(f"└{box_edge * 72}┘", style=self.colors.separator)
        else:
            # Simple line separators for Nebula
            self.print_separator()
            self.console.print(f"  agent: {agent} · task: {task[:50]}")
            self.print_separator()
        self.console.print()

    def print_score(self, score: float, biases: list[str]) -> None:
        """Print the overall rationality score with bar."""
        score_color = (
            self.colors.score_good if score > 0.7
            else self.colors.score_mid if score > 0.4
            else self.colors.score_bad
        )

        if self.name == "nebula":
            # Plain text layout for Nebula theme
            self.console.print(f"Overall rationality score: [bold {score_color}]{score:.2f}[/bold {score_color}]")
            if biases:
                self.console.print(f"Biases detected: [bold {self.colors.detected}]{', '.join(biases)}[/bold {self.colors.detected}]")
            else:
                self.console.print("No biases detected.")
        else:
            # Premium styled layout for Amber theme
            bar = _score_bar(score, use_unicode=self._supports_unicode)
            self.console.print(
                Text("  rationality   ", style=self.colors.label) +
                Text(f"{score:.2f}", style=score_color) +
                Text(f"   {bar}   ", style=self.colors.dim) +
                Text(
                    f"{len(biases)} bias{'es' if len(biases) > 1 else ''} detected",
                    style=self.colors.dim
                )
            )
        self.console.print()

    def print_bias_detail(self, bias_name: str, interpretation: str) -> None:
        """Print the interpretation line under a detected bias."""
        arrow = "→" if self._supports_unicode else "->"
        self.console.print(
            Text(f"  {arrow} {interpretation}", style=self.colors.dim)
        )

    def print_detector_table(self, detectors: dict) -> None:
        """Print the full detector results table."""
        if self.name == "nebula":
            # Minimal/Plain table for Nebula
            self.console.print("\n[bold]Detector[/bold]            [bold]Score[/bold]   [bold]Threshold[/bold]   [bold]Detected[/bold]")
            for name, result in detectors.items():
                detected = result.get("detected", False)
                score_val = result.get("score", 0.0)
                threshold = result.get("threshold", 0.0)
                
                status = self.detected_label if detected else self.clean_label
                color = self.colors.detected if detected else self.colors.clean
                
                # Manual padding for that "plain table" look
                self.console.print(
                    f"{name:<20} {score_val:<7.3f} {threshold:<11.2f} {status}"
                )
            return

        table = Table(
            box=self.table_box,
            show_header=True,
            header_style=self.colors.dim,
            padding=(0, 1),
        )
        table.add_column("detector", style=self.colors.label, min_width=20)
        table.add_column("score", justify="right", min_width=7)
        table.add_column("threshold", justify="right", min_width=10)
        table.add_column("detected", justify="left", min_width=10)

        for name, result in detectors.items():
            detected = result.get("detected", False)
            score_val = result.get("score", 0.0)
            threshold = result.get("threshold", 0.0)

            if detected:
                score_text = Text(f"{score_val:.3f}", style=self.colors.detected)
                status_text = Text(self.detected_label, style=self.colors.detected)
                name_text = Text(name, style=self.colors.detected)
            else:
                score_text = Text(f"{score_val:.3f}", style=self.colors.clean)
                status_text = Text(self.clean_label, style=self.colors.clean)
                name_text = Text(name, style=self.colors.label)

            table.add_row(
                name_text,
                score_text,
                Text(f"{threshold:.2f}", style=self.colors.dim),
                status_text,
            )

        self.console.print(table)

    def print_output_paths(self, timeline_path: str, report_path: str) -> None:
        """Print the output file paths."""
        self.print_separator()
        self.console.print(
            Text("  timeline  ", style=self.colors.info) +
            Text(timeline_path, style=self.colors.dim)
        )
        self.console.print(
            Text("  report    ", style=self.colors.info) +
            Text(report_path, style=self.colors.dim)
        )
        self.print_separator()

    def print_footer(self) -> None:
        """Print the footer line."""
        # Footer removed for cleaner professional look
        pass

    def print_error(self, message: str) -> None:
        self.console.print(Text(f"error: {message}", style=self.colors.detected))

    def print_warning(self, message: str) -> None:
        self.console.print(Text(f"warning: {message}", style=self.colors.score_mid))

    def print_success(self, message: str) -> None:
        self.console.print(Text(message, style=self.colors.clean))

    def status(self, message: str):
        """Return a Rich status context manager. Fall back to plain text for legacy terminals."""
        if not self._supports_unicode:
            self.console.print(f"  [dim].. {message}[/dim]")
            # Return a dummy context manager
            from contextlib import contextmanager
            @contextmanager
            def dummy():
                yield
            return dummy()
        return self.console.status(message)


# ─── Terminal detection ───────────────────────────────────────────────────────

def _supports_unicode() -> bool:
    """Detect if the terminal supports Unicode symbols (like the Braille spinner)."""
    # Check if we can encode a sample complex character
    try:
        "→".encode(sys.stdout.encoding or "ascii")
        return True
    except (UnicodeEncodeError, TypeError):
        return False


def _supports_color() -> bool:
    """Detect if the terminal supports ANSI color codes."""
    # Manual override
    if os.environ.get("XAUDIT_FORCE_COLOR"):
        return True

    # Check NO_COLOR env var (https://no-color.org)
    if os.environ.get("NO_COLOR"):
        return False

    # Check TERM
    term = os.environ.get("TERM", "")
    if term in ("dumb", ""):
        if sys.platform != "win32":
            return False

    # Windows CMD/PowerShell 
    if sys.platform == "win32":
        # Check if ANSI is supported (Windows 10+)
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            # Enable VIRTUAL_TERMINAL_PROCESSING
            mode = kernel32.GetStdHandle(-11)
            if kernel32.SetConsoleMode(mode, 7):
                return True
        except Exception:
            pass
        
        # Fallback: Many environments handle color even if SetConsoleMode fails
        # but rich usually handles this fine if we set force_terminal=True
        return True

    return True


# ─── Config persistence ───────────────────────────────────────────────────────

def load_config() -> dict:
    """Load global XAudit configuration."""
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"theme": "amber", "thresholds": {}}


def save_config(config: dict) -> None:
    """Save global XAudit configuration."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        json.dumps(config, indent=2),
        encoding="utf-8"
    )


def load_theme_config() -> ThemeName:
    """Load saved theme from ~/.xaudit/config.json."""
    config = load_config()
    saved = config.get("theme", "amber")
    return saved if saved in THEME_COLORS else "amber"


def save_theme_config(theme_name: ThemeName) -> None:
    """Save theme choice to ~/.xaudit/config.json."""
    config = load_config()
    config["theme"] = theme_name
    save_config(config)


def get_theme(name: str | None = None) -> Theme:
    """Get a Theme instance by name. Falls back to saved config, then amber."""
    if name is None:
        name = load_theme_config()
    if not name or name not in THEME_COLORS:
        name = "amber"
    return Theme(name)
