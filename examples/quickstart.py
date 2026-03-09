#!/usr/bin/env python3
"""
Research Agent — Quick Start Example.

Demonstrates the core workflow: configure project, compute stats,
generate figures, and compile a paper.
"""

from pathlib import Path

# ── 1. Project Configuration ─────────────────────────────────────────────────

from research_agent.config import ProjectConfig

cfg = ProjectConfig(
    name="demo-classification",
    project_dir=str(Path(__file__).parent),
    papers={1: "classification"},
    constants={"NUM_FOLDS": 5, "NUM_CLASSES": 10},
)
cfg.ensure_dirs()
print(f"Project initialized: {cfg.project_dir}")

# ── 2. Compute Statistics ────────────────────────────────────────────────────

from research_agent.metrics import (
    compute_descriptive_stats,
    run_group_comparison,
)

# Simulated per-fold accuracy results
model_a_acc = [92.1, 93.0, 91.8, 92.5, 92.1]
model_b_acc = [85.0, 86.2, 84.8, 85.5, 84.5]

stats_a = compute_descriptive_stats(model_a_acc, label="Our Model")
stats_b = compute_descriptive_stats(model_b_acc, label="Baseline")

print(f"\nOur Model:  {stats_a['mean']:.1f} +/- {stats_a['std']:.1f}%")
print(f"Baseline:   {stats_b['mean']:.1f} +/- {stats_b['std']:.1f}%")

comparison = run_group_comparison(model_a_acc, model_b_acc, labels=("Ours", "Baseline"))
print(f"\nWilcoxon p-value: {comparison['wilcoxon']['p_value']:.4f}")
print(f"Cohen's d:        {comparison['paired_ttest']['effect_size_d']:.2f}")

# ── 3. Generate Figures ──────────────────────────────────────────────────────

from research_agent.figures import create_pipeline_diagram, create_metric_charts

pipeline_fig = create_pipeline_diagram(
    stages=[
        {"label": "Raw\nImages", "color": "#0072B2"},
        {"label": "Augment\n& Normalize", "color": "#56B4E9"},
        {"label": "CNN\nEncoder", "color": "#009E73"},
        {"label": "Classification\nHead", "color": "#E69F00"},
        {"label": "Evaluation\n(5-fold CV)", "color": "#D55E00"},
    ],
    title="Classification Pipeline",
    output_dir=cfg.paper_figures(1),
)
print(f"\nPipeline figure: {pipeline_fig}")

charts = create_metric_charts(
    groups=[
        {"name": "Our Model", "mean": 92.3, "std": 1.5},
        {"name": "Baseline", "mean": 85.2, "std": 3.0},
    ],
    metric_name="Accuracy (%)",
    title="5-Fold Cross-Validation Results",
    output_dir=cfg.paper_figures(1),
)
print(f"Metric charts: {charts}")

# ── 4. Compile Paper ─────────────────────────────────────────────────────────

from research_agent.docx_builder import register_paper, compile_paper, PaperConfig

register_paper(1, PaperConfig(
    title="Automated Image Classification with Domain-Specific Fine-Tuning",
    authors="Research Agent Demo",
    sections_dir=cfg.paper_sections(1),
    figures_dir=cfg.paper_figures(1),
    drafts_dir=cfg.paper_drafts(1),
    figure_captions={
        "fig1_pipeline": "Figure 1: End-to-end classification pipeline.",
    },
    tables={
        "table1": {
            "caption": "Table 1: Model comparison (5-fold CV).",
            "columns": ["Model", "Accuracy (%)", "F1", "AUC"],
            "rows": [
                ["Ours", "92.3 +/- 1.5", "0.91", "0.95"],
                ["Baseline", "85.2 +/- 3.0", "0.83", "0.89"],
            ],
        },
    },
    output_name="demo_paper.docx",
))

paper_path = compile_paper(paper_id=1)
print(f"\nPaper compiled: {paper_path}")
print("\nDone! Open the DOCX to see the result.")
