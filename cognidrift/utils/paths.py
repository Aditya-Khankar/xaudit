"""Safe path handling — prevents directory traversal attacks on --output flag."""

import os


class PathTraversalError(ValueError):
    pass


def safe_output_path(output_dir: str, filename: str) -> str:
    """Resolve output path, raising if it escapes the output directory."""
    output_dir = os.path.abspath(output_dir)
    resolved = os.path.realpath(os.path.join(output_dir, filename))
    if not resolved.startswith(output_dir):
        raise PathTraversalError(
            f"Output path '{filename}' would escape output directory. "
            "Use a simple filename without path separators."
        )
    return resolved


def ensure_output_dir(output_dir: str) -> str:
    """Create output directory if it doesn't exist. Returns absolute path."""
    abs_path = os.path.abspath(output_dir)
    os.makedirs(abs_path, exist_ok=True)
    return abs_path
