"""
Research Agent Toolkit — Utilities.

File I/O helpers, shell execution, and generic research data loaders.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from . import config


# -- File I/O -----------------------------------------------------------------

def read_json(path: str | Path) -> dict | list:
    """Read and parse a JSON file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(path: str | Path, data: dict | list, indent: int = 2) -> Path:
    """Write data to a JSON file, creating parent dirs as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=indent, default=str), encoding="utf-8")
    return p


def read_text(path: str | Path) -> str:
    """Read a text file."""
    return Path(path).read_text(encoding="utf-8")


def write_text(path: str | Path, content: str) -> Path:
    """Write content to a text file, creating parent dirs as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def list_files(directory: str | Path, pattern: str = "*") -> list[Path]:
    """List files matching a glob pattern in a directory."""
    d = Path(directory)
    if not d.is_dir():
        return []
    return sorted(d.glob(pattern))


# -- Shell Execution ----------------------------------------------------------

def run_shell(
    command: str,
    cwd: str | Path | None = None,
    timeout: float = 300.0,
) -> dict:
    """Run a shell command synchronously.

    Returns dict with 'stdout', 'stderr', 'returncode'.
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=str(cwd) if cwd else None,
            timeout=timeout,
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": "Command timed out", "returncode": -1}


def run_python(
    script_path: str | Path,
    args: list[str] | None = None,
    cwd: str | Path | None = None,
    timeout: float = 600.0,
) -> dict:
    """Run a Python script as a subprocess."""
    cmd_parts = [sys.executable, str(script_path)]
    if args:
        cmd_parts.extend(args)

    try:
        result = subprocess.run(
            cmd_parts,
            capture_output=True,
            text=True,
            cwd=str(cwd) if cwd else None,
            timeout=timeout,
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": "Script timed out", "returncode": -1}


# -- Research Data Loaders ----------------------------------------------------

def load_results(
    results_dir: str | Path,
    pattern: str = "*.json",
) -> dict[str, dict | list]:
    """Load all JSON result files from a directory.

    Scans *results_dir* for files matching *pattern* and returns a dict
    mapping each filename (without extension) to its parsed contents.

    Parameters
    ----------
    results_dir : str | Path
        Directory to scan for result files.
    pattern : str
        Glob pattern for matching files (default ``"*.json"``).

    Returns
    -------
    dict[str, dict | list]
        Mapping of ``{stem: parsed_json}`` for every matched file.
    """
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        return {}

    out: dict[str, dict | list] = {}
    for fp in sorted(results_dir.glob(pattern)):
        if fp.is_file():
            try:
                out[fp.stem] = json.loads(fp.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                # Skip files that cannot be parsed
                continue
    return out


def load_experiment_data(experiment_dir: str | Path) -> dict:
    """Load experiment metadata and results from a standard directory layout.

    Expects the directory to optionally contain:

    * ``metadata.json`` — experiment configuration / hyperparameters.
    * ``results/``       — subdirectory of JSON result files.
    * ``config.json``    — runtime configuration snapshot.

    Any file that is missing is silently omitted from the returned dict.

    Parameters
    ----------
    experiment_dir : str | Path
        Root directory of a single experiment.

    Returns
    -------
    dict
        A dict with the following optional keys:

        * ``"metadata"``  — parsed ``metadata.json``
        * ``"config"``    — parsed ``config.json``
        * ``"results"``   — output of :func:`load_results` on the
          ``results/`` subdirectory
        * ``"path"``      — resolved :class:`~pathlib.Path` of the
          experiment directory (always present)
    """
    experiment_dir = Path(experiment_dir).resolve()
    data: dict = {"path": experiment_dir}

    if not experiment_dir.is_dir():
        return data

    # Load metadata
    metadata_path = experiment_dir / "metadata.json"
    if metadata_path.is_file():
        try:
            data["metadata"] = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, OSError):
            pass

    # Load config snapshot
    config_path = experiment_dir / "config.json"
    if config_path.is_file():
        try:
            data["config"] = json.loads(
                config_path.read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, OSError):
            pass

    # Load results
    results_subdir = experiment_dir / "results"
    if results_subdir.is_dir():
        results = load_results(results_subdir)
        if results:
            data["results"] = results

    return data
