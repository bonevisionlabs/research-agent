"""
Research Agent — Publication-Quality Figure Generation.

Matplotlib-based figure generators for academic papers. All functions accept
data as arguments (no hardcoded domain values) and return paths to saved PNGs.

Features:
    - Journal-specific style presets (IEEE, Elsevier, default)
    - Colorblind-friendly Wong palette (Wong, 2011)
    - Pipeline diagrams, metric charts, regional analysis, ablation studies
    - Generic placeholder and comparison figure generators

Usage:
    from research_agent.figures import create_pipeline_diagram, create_metric_charts

    create_pipeline_diagram(
        stages=[
            {"label": "Data Collection", "color": "#0072B2"},
            {"label": "Preprocessing", "color": "#E69F00"},
            {"label": "Model Training", "color": "#009E73"},
        ],
        title="Method Pipeline Overview",
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Publication Style Defaults
# ---------------------------------------------------------------------------

JOURNAL_STYLES: dict[str, dict[str, Any]] = {
    "default": {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    },
    "ieee": {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "figure.figsize": (3.5, 2.5),
        "figure.dpi": 300,
    },
    "elsevier": {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "figure.figsize": (7.0, 4.0),
        "figure.dpi": 300,
    },
}

# Colorblind-friendly palette (Wong, 2011)
COLORS: dict[str, str] = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "cyan": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#000000",
}
COLOR_LIST: list[str] = list(COLORS.values())


def apply_style(journal: str = "default") -> None:
    """Apply journal-specific matplotlib rcParams globally.

    Args:
        journal: One of the keys in ``JOURNAL_STYLES`` (e.g. ``"ieee"``,
            ``"elsevier"``). Falls back to ``"default"`` for unknown keys.
    """
    style = JOURNAL_STYLES.get(journal, JOURNAL_STYLES["default"])
    plt.rcParams.update(JOURNAL_STYLES["default"])
    plt.rcParams.update(style)


# ---------------------------------------------------------------------------
# Pipeline Diagram
# ---------------------------------------------------------------------------

def create_pipeline_diagram(
    stages: list[dict[str, str]],
    title: str,
    *,
    output_dir: Path | None = None,
    filename: str = "fig_pipeline.png",
    figsize: tuple[float, float] | None = None,
    journal: str = "default",
) -> Path:
    """Create a horizontal pipeline/flowchart diagram.

    Each stage is rendered as a rounded box with an arrow connecting it to
    the next stage. Suitable for illustrating method overviews in papers.

    Args:
        stages: Ordered list of pipeline stages. Each dict must have:
            - ``"label"`` (str): Display text (may contain ``\\n`` for wrapping).
            - ``"color"`` (str): Hex color for the box border and fill tint.
        title: Figure title string.
        output_dir: Directory to save the figure. Created if it does not exist.
            Defaults to the current working directory.
        filename: Output filename (default ``"fig_pipeline.png"``).
        figsize: Optional ``(width, height)`` tuple. Auto-calculated from
            the number of stages when *None*.
        journal: Journal style preset to apply before rendering.

    Returns:
        Path to the saved PNG file.

    Raises:
        ValueError: If *stages* is empty.

    Example::

        create_pipeline_diagram(
            stages=[
                {"label": "Raw Data", "color": "#0072B2"},
                {"label": "Feature\\nExtraction", "color": "#E69F00"},
                {"label": "Classification", "color": "#009E73"},
            ],
            title="Classification Pipeline",
        )
    """
    if not stages:
        raise ValueError("stages must be a non-empty list of dicts")

    apply_style(journal)
    out = Path(output_dir or Path.cwd())
    out.mkdir(parents=True, exist_ok=True)

    n = len(stages)
    box_w = 1.6
    box_h = 2.0
    spacing = 2.0  # center-to-center horizontal distance
    margin = 1.0

    total_width = (n - 1) * spacing + box_w + 2 * margin
    if figsize is None:
        figsize = (max(total_width, 6), 3.5)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_xlim(0, total_width)
    ax.set_ylim(0, 3)
    ax.axis("off")

    # Assign x positions centered in the available width
    x_start = margin + box_w / 2
    positioned = []
    for i, stage in enumerate(stages):
        positioned.append({
            "label": stage["label"],
            "color": stage.get("color", COLOR_LIST[i % len(COLOR_LIST)]),
            "x": x_start + i * spacing,
        })

    for stage in positioned:
        x = stage["x"]
        color = stage["color"]

        # Translucent fill
        rect = mpatches.FancyBboxPatch(
            (x - box_w / 2, 0.5), box_w, box_h,
            boxstyle="round,pad=0.1", facecolor=color,
            edgecolor="black", alpha=0.15, linewidth=1.5,
        )
        ax.add_patch(rect)

        # Colored border
        border = mpatches.FancyBboxPatch(
            (x - box_w / 2, 0.5), box_w, box_h,
            boxstyle="round,pad=0.1", facecolor="none",
            edgecolor=color, linewidth=2.0,
        )
        ax.add_patch(border)

        ax.text(
            x, 1.5, stage["label"],
            ha="center", va="center",
            fontsize=9, fontweight="bold", color="#1a1a1a",
        )

    # Arrows between boxes
    for i in range(len(positioned) - 1):
        x_end_arrow = positioned[i + 1]["x"] - box_w / 2 - 0.05
        x_start_arrow = positioned[i]["x"] + box_w / 2 + 0.05
        ax.annotate(
            "", xy=(x_end_arrow, 1.5), xytext=(x_start_arrow, 1.5),
            arrowprops=dict(arrowstyle="->", color="#555", lw=2),
        )

    ax.set_title(title, fontsize=12, fontweight="bold", pad=15)

    path = out / filename
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    logger.info("Created pipeline diagram: %s", path)
    return path


# ---------------------------------------------------------------------------
# Metric Charts (Bar + Box)
# ---------------------------------------------------------------------------

def create_metric_charts(
    groups: list[dict[str, Any]],
    *,
    title: str = "Evaluation Metrics",
    metric_name: str = "Error",
    metric_unit: str = "mm",
    secondary_metric_name: str | None = None,
    secondary_metric_unit: str | None = None,
    output_dir: Path | None = None,
    filename_bar: str = "fig_metrics_bar.png",
    filename_box: str = "fig_metrics_boxplot.png",
    journal: str = "default",
) -> dict[str, Path]:
    """Create bar charts and optional box plots from evaluation results.

    Produces a bar chart comparing primary (and optionally secondary) metrics
    across experimental groups. If any group provides per-fold/per-sample
    data, an additional box plot is generated.

    Args:
        groups: List of result groups. Each dict must have:
            - ``"name"`` (str): Display label (e.g. ``"Model A"``).
            - ``"mean"`` (float): Primary metric mean.
            - ``"std"`` (float): Primary metric standard deviation.
          Optional keys:
            - ``"secondary_mean"`` (float): Secondary metric mean
              (e.g. a distance metric).
            - ``"per_fold_errors"`` (list[float]): Per-fold/per-sample values
              for box plot generation.
            - ``"color"`` (str): Override bar color for this group.
        title: Super-title for the figure.
        metric_name: Label for the primary metric (default ``"Error"``).
        metric_unit: Unit string for the primary metric (default ``"mm"``).
        secondary_metric_name: Label for the secondary metric panel. When
            *None*, the secondary panel is omitted.
        secondary_metric_unit: Unit for the secondary metric.
        output_dir: Directory to save figures. Created if needed.
        filename_bar: Filename for the bar chart PNG.
        filename_box: Filename for the box plot PNG.
        journal: Journal style preset.

    Returns:
        Dict with ``"bar_chart"`` and (if applicable) ``"box_plot"`` paths.

    Raises:
        ValueError: If *groups* is empty.

    Example::

        create_metric_charts(
            groups=[
                {"name": "Baseline", "mean": 4.2, "std": 1.1,
                 "secondary_mean": 12.5, "per_fold_errors": [3.8, 4.1, 4.7]},
                {"name": "Proposed", "mean": 3.1, "std": 0.9,
                 "secondary_mean": 8.2, "per_fold_errors": [2.9, 3.0, 3.4]},
            ],
            title="Model Comparison (5-fold CV)",
            metric_name="Mean Surface Error",
            metric_unit="mm",
            secondary_metric_name="HD95",
            secondary_metric_unit="mm",
        )
    """
    if not groups:
        raise ValueError("groups must be a non-empty list")

    apply_style(journal)
    out = Path(output_dir or Path.cwd())
    out.mkdir(parents=True, exist_ok=True)

    names = [g["name"] for g in groups]
    means = [g["mean"] for g in groups]
    stds = [g["std"] for g in groups]
    colors = [
        g.get("color", COLOR_LIST[i % len(COLOR_LIST)])
        for i, g in enumerate(groups)
    ]

    has_secondary = (
        secondary_metric_name is not None
        and any("secondary_mean" in g for g in groups)
    )
    n_panels = 2 if has_secondary else 1

    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    # Primary metric bar chart
    bars = axes[0].bar(
        names, means, yerr=stds, capsize=5,
        color=colors, edgecolor="black", linewidth=0.5, alpha=0.8,
    )
    axes[0].set_ylabel(f"{metric_name} ({metric_unit})")
    axes[0].set_title(metric_name, fontsize=11, fontweight="bold")
    for bar, val, err in zip(bars, means, stds):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + err + 0.1,
            f"{val:.2f}", ha="center", va="bottom", fontsize=9,
        )

    # Secondary metric bar chart (optional)
    if has_secondary:
        sec_vals = [g.get("secondary_mean", 0) for g in groups]
        bars2 = axes[1].bar(
            names, sec_vals, capsize=5,
            color=colors, edgecolor="black", linewidth=0.5, alpha=0.8,
        )
        sec_unit = secondary_metric_unit or metric_unit
        axes[1].set_ylabel(f"{secondary_metric_name} ({sec_unit})")
        axes[1].set_title(secondary_metric_name, fontsize=11, fontweight="bold")
        for bar, val in zip(bars2, sec_vals):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9,
            )

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()

    bar_path = out / filename_bar
    fig.savefig(bar_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    result: dict[str, Path] = {"bar_chart": bar_path}

    # Box plot (only if per-fold data provided by at least one group)
    fold_data = [g.get("per_fold_errors", []) for g in groups]
    if any(fold_data):
        # Only include groups that have fold data
        box_groups = [
            (g["name"], g["per_fold_errors"], colors[i])
            for i, g in enumerate(groups)
            if g.get("per_fold_errors")
        ]

        fig2, ax2 = plt.subplots(1, 1, figsize=(max(3, 2 * len(box_groups)), 5))
        bp = ax2.boxplot(
            [bg[1] for bg in box_groups],
            positions=list(range(len(box_groups))),
            widths=0.5,
            patch_artist=True,
            showmeans=True,
            meanprops=dict(
                marker="D", markeredgecolor="black",
                markerfacecolor="white", markersize=6,
            ),
        )
        for patch, (_, _, color) in zip(bp["boxes"], box_groups):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax2.set_xticklabels([bg[0] for bg in box_groups])
        ax2.set_ylabel(f"{metric_name} ({metric_unit})")
        ax2.set_title(f"Per-Sample Distribution", fontsize=12, fontweight="bold")

        box_path = out / filename_box
        fig2.savefig(box_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig2)
        result["box_plot"] = box_path

    logger.info("Created metric charts: %s", bar_path)
    return result


# ---------------------------------------------------------------------------
# Regional / Category Analysis
# ---------------------------------------------------------------------------

def create_regional_analysis(
    regions: dict[str, dict[str, tuple[float, float]]],
    *,
    title: str = "Regional Error Analysis",
    metric_name: str = "Mean Error",
    metric_unit: str = "mm",
    output_dir: Path | None = None,
    filename: str = "fig_regional.png",
    journal: str = "default",
) -> Path:
    """Create horizontal bar charts comparing metrics across sub-regions.

    Renders one panel per group, each showing a horizontal bar for every
    region/category within that group. Useful for spatial error analysis,
    per-class breakdowns, or any grouped categorical comparison.

    Args:
        regions: Nested mapping of ``group_name -> {region_name: (mean, std)}``.
            Each top-level key becomes a subplot panel. Each inner key is a
            category/region with its ``(mean_value, std_value)`` tuple.
        title: Super-title for the entire figure.
        metric_name: Axis label for the measured quantity.
        metric_unit: Unit string appended to the axis label.
        output_dir: Directory to save the figure. Created if needed.
        filename: Output filename.
        journal: Journal style preset.

    Returns:
        Path to the saved PNG.

    Raises:
        ValueError: If *regions* is empty.

    Example::

        create_regional_analysis(
            regions={
                "Model A": {
                    "Category 1": (2.5, 0.3),
                    "Category 2": (3.1, 0.5),
                    "Category 3": (1.8, 0.2),
                },
                "Model B": {
                    "Category 1": (2.1, 0.4),
                    "Category 2": (2.8, 0.3),
                    "Category 3": (2.0, 0.3),
                },
            },
            title="Per-Region Accuracy Breakdown",
        )
    """
    if not regions:
        raise ValueError("regions must be a non-empty dict")

    apply_style(journal)
    out = Path(output_dir or Path.cwd())
    out.mkdir(parents=True, exist_ok=True)

    group_names = list(regions.keys())
    n_groups = len(group_names)

    fig, axes = plt.subplots(
        1, n_groups,
        figsize=(7 * n_groups, max(4, 0.6 * max(len(v) for v in regions.values()) + 2)),
        squeeze=False,
    )
    axes = axes[0]  # flatten from (1, n) to (n,)

    for idx, (group_name, region_data) in enumerate(regions.items()):
        ax = axes[idx]
        color = COLOR_LIST[idx % len(COLOR_LIST)]

        r_names = list(region_data.keys())
        r_means = [v[0] for v in region_data.values()]
        r_stds = [v[1] for v in region_data.values()]

        bars = ax.barh(
            r_names, r_means, xerr=r_stds,
            color=color, alpha=0.7, edgecolor="black",
            linewidth=0.5, capsize=3,
        )
        ax.set_xlabel(f"{metric_name} ({metric_unit})")
        ax.set_title(
            f"{group_name}: Regional Analysis",
            fontsize=11, fontweight="bold",
        )

        overall_mean = float(np.mean(r_means))
        ax.axvline(
            x=overall_mean, color=COLORS["red"], linestyle="--",
            label=f"Mean: {overall_mean:.2f}{metric_unit}",
        )
        ax.legend(fontsize=9)

        for bar, val in zip(bars, r_means):
            ax.text(
                val + 0.15, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=8,
            )

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    path = out / filename
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    logger.info("Created regional analysis: %s", path)
    return path


# ---------------------------------------------------------------------------
# Ablation Chart
# ---------------------------------------------------------------------------

def create_ablation_chart(
    panels: list[dict[str, Any]],
    *,
    title: str = "Ablation Studies",
    output_dir: Path | None = None,
    filename: str = "fig_ablation.png",
    journal: str = "default",
) -> Path:
    """Create multi-panel ablation study charts.

    Each panel represents one ablation experiment. Panels can be bar charts
    (comparing discrete conditions) or line charts (sweeping a continuous
    hyperparameter).

    Args:
        panels: List of panel specifications. Each dict must have:
            - ``"title"`` (str): Panel subtitle.
            - ``"type"`` (str): ``"bar"`` or ``"line"``.

          For ``"bar"`` panels:
            - ``"labels"`` (list[str]): Category labels.
            - ``"values"`` (list[float]): Bar heights.
            - ``"errors"`` (list[float], optional): Error bar sizes.
            - ``"colors"`` (list[str], optional): Per-bar colors.

          For ``"line"`` panels:
            - ``"x"`` (list[float]): X-axis values.
            - ``"y"`` (list[float]): Y-axis values.
            - ``"xlabel"`` (str, optional): X-axis label.
            - ``"ylabel"`` (str, optional): Y-axis label.
        title: Super-title for the figure.
        output_dir: Directory to save the figure. Created if needed.
        filename: Output filename.
        journal: Journal style preset.

    Returns:
        Path to the saved PNG.

    Raises:
        ValueError: If *panels* is empty or a panel has an unknown type.

    Example::

        create_ablation_chart(
            panels=[
                {
                    "title": "Loss Function",
                    "type": "bar",
                    "labels": ["MSE", "L1", "Huber"],
                    "values": [4.2, 3.8, 3.5],
                    "errors": [0.3, 0.4, 0.2],
                },
                {
                    "title": "Learning Rate",
                    "type": "line",
                    "x": [1e-4, 5e-4, 1e-3, 5e-3],
                    "y": [4.5, 3.5, 3.2, 4.1],
                    "xlabel": "Learning Rate",
                    "ylabel": "Error (mm)",
                },
            ],
            title="Ablation Studies",
        )
    """
    if not panels:
        raise ValueError("panels must be a non-empty list")

    apply_style(journal)
    out = Path(output_dir or Path.cwd())
    out.mkdir(parents=True, exist_ok=True)

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for idx, panel in enumerate(panels):
        ax = axes[idx]
        ptype = panel.get("type", "bar")

        if ptype == "bar":
            labels = panel["labels"]
            vals = panel["values"]
            errs = panel.get("errors", [0] * len(vals))
            bar_colors = panel.get(
                "colors",
                [COLOR_LIST[j % len(COLOR_LIST)] for j in range(len(vals))],
            )
            ax.bar(
                labels, vals, yerr=errs, capsize=5,
                color=bar_colors, edgecolor="black",
                linewidth=0.5, alpha=0.8,
            )
            ax.set_ylabel(panel.get("ylabel", "Metric Value"))

        elif ptype == "line":
            x_vals = panel["x"]
            y_vals = panel["y"]
            ax.plot(
                x_vals, y_vals, "o-",
                color=panel.get("color", COLORS["blue"]),
                linewidth=2, markersize=6, markeredgecolor="black",
            )
            ax.set_xlabel(panel.get("xlabel", "Parameter"))
            ax.set_ylabel(panel.get("ylabel", "Metric Value"))

        else:
            raise ValueError(
                f"Unknown panel type '{ptype}' in panel {idx}. "
                f"Expected 'bar' or 'line'."
            )

        ax.set_title(panel.get("title", f"Ablation {idx + 1}"),
                      fontsize=10, fontweight="bold")

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()

    path = out / filename
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    logger.info("Created ablation chart: %s", path)
    return path


# ---------------------------------------------------------------------------
# Placeholder Figure
# ---------------------------------------------------------------------------

def create_placeholder_figure(
    title: str,
    labels: list[str],
    *,
    output_dir: Path | None = None,
    filename: str = "fig_placeholder.png",
    journal: str = "default",
) -> Path:
    """Create a labeled multi-panel placeholder figure.

    Generates a row of empty panels with centered labels, useful as a
    stand-in for figures that will later be replaced with real data
    (e.g., qualitative results, medical images, rendered visualizations).

    Args:
        title: Figure super-title.
        labels: List of panel labels. One subplot is created per label.
        output_dir: Directory to save the figure. Created if needed.
        filename: Output filename.
        journal: Journal style preset.

    Returns:
        Path to the saved PNG.

    Raises:
        ValueError: If *labels* is empty.

    Example::

        create_placeholder_figure(
            title="Qualitative Results",
            labels=["Input", "Predicted", "Ground Truth"],
        )
    """
    if not labels:
        raise ValueError("labels must be a non-empty list")

    apply_style(journal)
    out = Path(output_dir or Path.cwd())
    out.mkdir(parents=True, exist_ok=True)

    n = len(labels)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, label in zip(axes, labels):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.text(
            0.5, 0.5, f"[{label}]",
            ha="center", va="center",
            fontsize=14, color="#888888", fontstyle="italic",
        )
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linestyle("--")
            spine.set_color("#cccccc")

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()

    path = out / filename
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    logger.info("Created placeholder figure: %s", path)
    return path


# ---------------------------------------------------------------------------
# Comparison Figure
# ---------------------------------------------------------------------------

def create_comparison_figure(
    title: str,
    panel_titles: list[str],
    *,
    output_dir: Path | None = None,
    filename: str = "fig_comparison.png",
    journal: str = "default",
) -> Path:
    """Create a multi-panel comparison placeholder figure.

    Generates a row of panels with schematic shapes to represent a
    predicted-vs-ground-truth comparison layout. The last panel renders
    as an error heatmap placeholder. Replace with actual data renders
    (e.g., from domain-specific visualization tools) for the final paper.

    Args:
        title: Figure super-title.
        panel_titles: List of panel labels (e.g.
            ``["Predicted", "Ground Truth", "Error Map"]``).
            If a panel title contains ``"error"`` (case-insensitive), it
            is rendered as a scatter-based heatmap; otherwise as a filled
            contour shape.
        output_dir: Directory to save the figure. Created if needed.
        filename: Output filename.
        journal: Journal style preset.

    Returns:
        Path to the saved PNG.

    Raises:
        ValueError: If *panel_titles* is empty.

    Example::

        create_comparison_figure(
            title="Predicted vs Ground Truth",
            panel_titles=["Predicted", "Reference", "Error Heatmap"],
        )
    """
    if not panel_titles:
        raise ValueError("panel_titles must be a non-empty list")

    apply_style(journal)
    out = Path(output_dir or Path.cwd())
    out.mkdir(parents=True, exist_ok=True)

    n = len(panel_titles)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for idx, (ax, ptitle) in enumerate(zip(axes, panel_titles)):
        theta = np.linspace(0, 2 * np.pi, 100)
        r = 1 + 0.3 * np.cos(3 * theta)

        # Add slight noise to non-reference panels
        if idx == 0:
            rng = np.random.default_rng(42)
            r = r + rng.normal(0, 0.05, len(theta))

        x = r * np.cos(theta)
        y = r * np.sin(theta)

        if "error" in ptitle.lower():
            rng = np.random.default_rng(99)
            errors = np.abs(rng.normal(0, 0.05, len(theta)))
            scatter = ax.scatter(
                x, y, c=errors, cmap="RdYlGn_r", s=5, vmin=0, vmax=0.15,
            )
            plt.colorbar(scatter, ax=ax, label="Error", shrink=0.8)
        else:
            color = COLOR_LIST[idx % len(COLOR_LIST)]
            ax.fill(x, y, alpha=0.3, color=color)
            ax.plot(x, y, color=color, linewidth=1.5)

        ax.set_title(ptitle, fontsize=10, fontweight="bold")
        ax.set_aspect("equal")
        ax.axis("off")

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()

    path = out / filename
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    logger.info("Created comparison figure: %s", path)
    return path


# ---------------------------------------------------------------------------
# Generate All Figures
# ---------------------------------------------------------------------------

def create_all_figures(
    figure_specs: list[dict[str, Any]],
    *,
    journal: str = "default",
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """Generate multiple figures from a list of specifications.

    Each spec is a dict with a ``"type"`` key selecting the generator
    function and a ``"kwargs"`` dict passed as keyword arguments.

    Args:
        figure_specs: List of figure specifications. Each dict must have:
            - ``"name"`` (str): Key for the returned dict.
            - ``"type"`` (str): One of ``"pipeline"``, ``"metrics"``,
              ``"regional"``, ``"ablation"``, ``"placeholder"``,
              ``"comparison"``.
            - ``"kwargs"`` (dict): Keyword arguments forwarded to the
              corresponding ``create_*`` function. ``output_dir`` and
              ``journal`` are injected automatically if not present.
        journal: Journal style preset applied globally before generation.
        output_dir: Default output directory for all figures.

    Returns:
        Dict mapping each spec's ``"name"`` to the path (or dict of paths)
        returned by the generator.

    Example::

        create_all_figures(
            figure_specs=[
                {
                    "name": "pipeline",
                    "type": "pipeline",
                    "kwargs": {
                        "stages": [
                            {"label": "Input", "color": "#0072B2"},
                            {"label": "Process", "color": "#E69F00"},
                            {"label": "Output", "color": "#009E73"},
                        ],
                        "title": "Method Overview",
                    },
                },
                {
                    "name": "results",
                    "type": "metrics",
                    "kwargs": {
                        "groups": [
                            {"name": "Ours", "mean": 3.1, "std": 0.8},
                            {"name": "Baseline", "mean": 4.5, "std": 1.2},
                        ],
                    },
                },
            ],
            journal="ieee",
        )
    """
    apply_style(journal)
    out = Path(output_dir or Path.cwd())
    out.mkdir(parents=True, exist_ok=True)

    _GENERATORS = {
        "pipeline": create_pipeline_diagram,
        "metrics": create_metric_charts,
        "regional": create_regional_analysis,
        "ablation": create_ablation_chart,
        "placeholder": create_placeholder_figure,
        "comparison": create_comparison_figure,
    }

    figures: dict[str, Any] = {}

    for spec in figure_specs:
        name = spec["name"]
        fig_type = spec["type"]
        kwargs = dict(spec.get("kwargs", {}))

        generator = _GENERATORS.get(fig_type)
        if generator is None:
            logger.error(
                "Unknown figure type '%s' for spec '%s'. "
                "Available types: %s",
                fig_type, name, list(_GENERATORS.keys()),
            )
            continue

        # Inject defaults if not explicitly provided
        kwargs.setdefault("output_dir", out)
        kwargs.setdefault("journal", journal)

        try:
            result = generator(**kwargs)
            if isinstance(result, dict):
                figures.update(result)
            else:
                figures[name] = result
        except Exception as e:
            logger.error("Failed to generate '%s' (%s): %s", name, fig_type, e)

    return figures
