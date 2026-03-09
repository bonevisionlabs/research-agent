"""
Research Agent -- Metrics & Statistical Analysis.

Generic utilities for loading experiment results, computing descriptive
statistics, and running paired statistical tests (Wilcoxon signed-rank,
paired t-test).  Designed for cross-validation and multi-group experimental
workflows common in scientific research.

No LLM calls -- pure computation on real data.

Usage:
    from research_agent.metrics import (
        load_experiment_summary,
        run_group_comparison,
        compute_descriptive_stats,
    )
    summary = load_experiment_summary(Path("results/experiment_01"))
    comparison = run_group_comparison(errors_a, errors_b, labels=("Model A", "Model B"))
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# -- Experiment Results --------------------------------------------------------

def load_experiment_summary(results_dir: Path) -> dict:
    """Load all JSON result files from a directory and compute aggregate stats.

    Scans *results_dir* for ``*.json`` files, loads each into a list, and
    computes aggregate statistics across any numeric fields that appear in
    every result file.

    Parameters
    ----------
    results_dir : Path
        Directory containing one or more JSON result files.

    Returns
    -------
    dict
        ``{
            "results_dir": str,
            "n_files": int,
            "files": [str, ...],
            "results": [dict, ...],
            "aggregate": {
                "<numeric_key>": {"mean": float, "std": float, "median": float,
                                  "min": float, "max": float, "values": [float, ...]},
                ...
            }
        }``
        If *results_dir* does not exist or contains no JSON files, returns a
        dict with ``"error"`` key.
    """
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        return {"error": f"Directory not found: {results_dir}"}

    json_files = sorted(results_dir.glob("*.json"))
    if not json_files:
        return {"error": f"No JSON files found in {results_dir}"}

    results: list[dict] = []
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                data["_source_file"] = jf.name
                results.append(data)
            elif isinstance(data, list):
                # If the file contains a list of dicts, include each entry
                for i, entry in enumerate(data):
                    if isinstance(entry, dict):
                        entry["_source_file"] = f"{jf.name}[{i}]"
                        results.append(entry)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping %s: %s", jf, exc)

    if not results:
        return {"error": f"No valid JSON dicts loaded from {results_dir}"}

    # Identify numeric keys common to all result dicts (ignoring _source_file)
    common_keys: set[str] | None = None
    for r in results:
        numeric_keys = {
            k for k, v in r.items()
            if k != "_source_file" and isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        common_keys = numeric_keys if common_keys is None else common_keys & numeric_keys

    aggregate: dict[str, dict[str, Any]] = {}
    for key in sorted(common_keys or set()):
        values = [float(r[key]) for r in results]
        arr = np.array(values)
        aggregate[key] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "values": values,
        }

    return {
        "results_dir": str(results_dir),
        "n_files": len(json_files),
        "files": [jf.name for jf in json_files],
        "results": results,
        "aggregate": aggregate,
    }


def get_results_table(results: list[dict], columns: list[str]) -> list[dict]:
    """Format a list of result dicts into a table with the specified columns.

    Each entry in *results* is projected onto *columns*.  Missing keys are
    filled with ``None``.  Numeric values are left as-is (not stringified) so
    callers can decide formatting.

    Parameters
    ----------
    results : list[dict]
        Raw result dicts (e.g., from ``load_experiment_summary()["results"]``).
    columns : list[str]
        Column names to include in the output rows.

    Returns
    -------
    list[dict]
        One dict per result, containing only the requested columns.
    """
    table: list[dict] = []
    for r in results:
        row = {col: r.get(col) for col in columns}
        table.append(row)
    return table


def compute_fold_statistics(fold_results: list[dict], metric_key: str) -> dict:
    """Extract a single metric across folds and compute descriptive statistics.

    Useful for cross-validation workflows where each fold produces a result
    dict containing (among other things) the metric of interest.

    Parameters
    ----------
    fold_results : list[dict]
        One dict per fold.  Each must contain *metric_key* with a numeric value.
    metric_key : str
        Key to extract from each fold dict.

    Returns
    -------
    dict
        Descriptive statistics for the extracted metric values, plus the raw
        ``"values"`` list and ``"n_folds"`` count.  Returns an ``"error"`` key
        if the metric is missing or no folds are provided.
    """
    if not fold_results:
        return {"error": "No fold results provided"}

    values: list[float] = []
    missing_folds: list[int] = []
    for i, fold in enumerate(fold_results):
        v = fold.get(metric_key)
        if v is not None and isinstance(v, (int, float)) and not isinstance(v, bool):
            values.append(float(v))
        else:
            missing_folds.append(i)

    if not values:
        return {"error": f"Metric '{metric_key}' not found in any fold result"}

    arr = np.array(values)
    stats = {
        "metric": metric_key,
        "n_folds": len(values),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "q1": float(np.percentile(arr, 25)),
        "q3": float(np.percentile(arr, 75)),
        "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        "values": values,
    }

    if missing_folds:
        stats["missing_folds"] = missing_folds
        logger.warning(
            "Metric '%s' missing in %d fold(s): %s",
            metric_key, len(missing_folds), missing_folds,
        )

    return stats


# -- Statistical Tests ---------------------------------------------------------

def run_wilcoxon_test(errors_a: list[float], errors_b: list[float]) -> dict:
    """Run Wilcoxon signed-rank test between two paired error arrays.

    Parameters
    ----------
    errors_a, errors_b : list[float]
        Paired measurements.  Must be equal length and at least 5 samples.

    Returns
    -------
    dict
        Test statistic, p-value, significance flags, and difference stats.
    """
    from scipy import stats as scipy_stats

    a = np.array(errors_a)
    b = np.array(errors_b)

    if len(a) != len(b):
        return {"error": f"Array lengths differ: {len(a)} vs {len(b)}"}

    if len(a) < 5:
        return {"error": f"Too few samples for Wilcoxon test: {len(a)}"}

    stat, p_val = scipy_stats.wilcoxon(a, b)

    return {
        "test": "Wilcoxon signed-rank",
        "statistic": float(stat),
        "p_value": float(p_val),
        "significant_005": p_val < 0.05,
        "significant_001": p_val < 0.01,
        "n": len(a),
        "mean_diff": float(np.mean(a - b)),
        "std_diff": float(np.std(a - b)),
    }


def run_paired_ttest(errors_a: list[float], errors_b: list[float]) -> dict:
    """Run paired t-test between two error arrays.

    Parameters
    ----------
    errors_a, errors_b : list[float]
        Paired measurements.  Must be equal length.

    Returns
    -------
    dict
        Test statistic, p-value, significance flags, and Cohen's d effect size.
    """
    from scipy import stats as scipy_stats

    a = np.array(errors_a)
    b = np.array(errors_b)

    if len(a) != len(b):
        return {"error": f"Array lengths differ: {len(a)} vs {len(b)}"}

    stat, p_val = scipy_stats.ttest_rel(a, b)

    return {
        "test": "Paired t-test",
        "statistic": float(stat),
        "p_value": float(p_val),
        "significant_005": p_val < 0.05,
        "significant_001": p_val < 0.01,
        "n": len(a),
        "mean_diff": float(np.mean(a - b)),
        "effect_size_d": float(np.mean(a - b) / np.std(a - b)) if np.std(a - b) > 0 else 0.0,
    }


def run_group_comparison(
    group_a: list[float],
    group_b: list[float],
    labels: tuple[str, str] = ("A", "B"),
) -> dict:
    """Run both Wilcoxon and paired t-test between two groups.

    Convenience wrapper that runs both non-parametric and parametric paired
    tests and returns combined results alongside descriptive summaries.

    Parameters
    ----------
    group_a, group_b : list[float]
        Paired measurements for each group.
    labels : tuple[str, str]
        Human-readable names for the two groups (used in summary keys).

    Returns
    -------
    dict
        ``{
            "wilcoxon": { ... },
            "paired_ttest": { ... },
            "<label_a>_summary": {"mean", "std", "n"},
            "<label_b>_summary": {"mean", "std", "n"},
        }``
    """
    label_a, label_b = labels

    result: dict[str, Any] = {
        "wilcoxon": run_wilcoxon_test(group_a, group_b),
        "paired_ttest": run_paired_ttest(group_a, group_b),
    }

    a = np.array(group_a)
    b = np.array(group_b)

    result[f"{label_a}_summary"] = {
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "n": len(a),
    }
    result[f"{label_b}_summary"] = {
        "mean": float(np.mean(b)),
        "std": float(np.std(b)),
        "n": len(b),
    }

    return result


def compute_descriptive_stats(errors: list[float], label: str = "") -> dict:
    """Compute descriptive statistics for a numeric array.

    Parameters
    ----------
    errors : list[float]
        Numeric measurements (errors, scores, durations, etc.).
    label : str, optional
        Human-readable label for the dataset.

    Returns
    -------
    dict
        Mean, std, median, min, max, Q1, Q3, and IQR.
    """
    a = np.array(errors)
    return {
        "label": label,
        "n": len(a),
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "median": float(np.median(a)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "q1": float(np.percentile(a, 25)),
        "q3": float(np.percentile(a, 75)),
        "iqr": float(np.percentile(a, 75) - np.percentile(a, 25)),
    }


# -- Save Results --------------------------------------------------------------

def save_results(name: str, data: dict, output_dir: Path | str = Path("results")) -> Path:
    """Save evaluation results as JSON to the specified output directory.

    Parameters
    ----------
    name : str
        Base filename (without extension).
    data : dict
        JSON-serializable results to persist.
    output_dir : Path or str
        Directory to write to.  Created if it does not exist.

    Returns
    -------
    Path
        Absolute path to the written JSON file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    logger.info("Saved results to %s", path)
    return path
