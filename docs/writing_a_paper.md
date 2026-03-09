# Writing a Paper with Research Agent

This guide walks through the end-to-end process of using Research Agent with Claude Code to produce a publication-ready manuscript.

## 1. Project Setup

```python
from research_agent.config import ProjectConfig

cfg = ProjectConfig(
    name="my-study",
    project_dir="/path/to/my/project",
    papers={
        1: "image-classification",
        2: "ablation-study",
    },
    constants={
        "NUM_FOLDS": 5,
        "NUM_SUBJECTS": 100,
        "MODEL_NAME": "ResNet-50",
    },
)
cfg.ensure_dirs()
```

This creates:
```
state/
├── figures/generated/
├── evaluation/results/
├── paper1_image-classification/
│   ├── sections/
│   ├── figures/
│   └── drafts/
└── paper2_ablation-study/
    ├── sections/
    ├── figures/
    └── drafts/
```

## 2. Initialize the Workflow

```python
from research_agent.workflow import Workflow

wf = Workflow.from_template("templates/paper_workflow.json")
wf.save("state/workflow.json")
```

Or tell Claude Code: *"Initialize the paper workflow and show me what tasks need to be done."*

## 3. Literature Review Phase

Claude Code handles this with its built-in capabilities:
- Searches papers using web search or research-lookup skill
- Reads and synthesizes related work
- Writes the literature review section

```python
from research_agent.docx_builder import save_section

# Claude Code writes the section text, then saves it:
save_section("introduction", intro_text, paper_id=1)
save_section("methods", methods_text, paper_id=1)
```

## 4. Evaluation Phase

```python
from research_agent.metrics import (
    compute_descriptive_stats,
    run_group_comparison,
    compute_fold_statistics,
)

# Compute stats on your experimental results
model_a_stats = compute_descriptive_stats(model_a_errors, label="Model A")
model_b_stats = compute_descriptive_stats(model_b_errors, label="Model B")

# Run statistical comparisons
comparison = run_group_comparison(
    model_a_errors, model_b_errors,
    labels=("Model A", "Model B"),
)
# → {"wilcoxon": {...}, "ttest": {...}, "group_a": {...}, "group_b": {...}}

# Summarize cross-validation results
fold_stats = compute_fold_statistics(fold_results, metric_key="accuracy")
```

## 5. Figure Generation

```python
from research_agent.figures import (
    create_pipeline_diagram,
    create_metric_charts,
    create_ablation_chart,
    apply_style,
)

# Apply journal style globally
apply_style("ieee")

# Create a pipeline diagram
pipeline_path = create_pipeline_diagram(
    stages=[
        {"label": "Data\nCollection", "color": "#0072B2"},
        {"label": "Preprocessing", "color": "#56B4E9"},
        {"label": "Feature\nExtraction", "color": "#009E73"},
        {"label": "Classification", "color": "#E69F00"},
        {"label": "Evaluation", "color": "#D55E00"},
    ],
    title="Our Method: End-to-End Pipeline",
    output_dir=cfg.paper_figures(1),
)

# Create metric comparison charts
charts = create_metric_charts(
    groups=[
        {"name": "Model A", "mean": 92.3, "std": 1.5},
        {"name": "Model B", "mean": 89.7, "std": 2.1},
        {"name": "Baseline", "mean": 85.2, "std": 3.0},
    ],
    metric_name="Accuracy (%)",
    title="Model Comparison",
    output_dir=cfg.paper_figures(1),
)
```

## 6. Paper Compilation

```python
from research_agent.docx_builder import register_paper, compile_paper, PaperConfig

register_paper(1, PaperConfig(
    title="Deep Learning for Medical Image Classification",
    authors="Jane Smith, John Doe\nUniversity of Example",
    sections_dir=cfg.paper_sections(1),
    figures_dir=cfg.paper_figures(1),
    drafts_dir=cfg.paper_drafts(1),
    figure_captions={
        "fig1_pipeline": "Figure 1: Our end-to-end classification pipeline...",
        "fig2_results": "Figure 2: Comparison of model accuracy across...",
    },
    tables={
        "table1": {
            "caption": "Table 1: Classification accuracy by model.",
            "columns": ["Model", "Accuracy", "F1-Score", "AUC"],
            "rows": [
                ["Ours", "92.3 +/- 1.5", "0.91", "0.95"],
                ["Baseline", "85.2 +/- 3.0", "0.83", "0.89"],
            ],
        },
    },
    output_name="classification_paper.docx",
))

output_path = compile_paper(paper_id=1)
print(f"Paper compiled: {output_path}")
```

## 7. Review Phase

Use Claude Code's built-in reasoning with the review criteria:

```python
from research_agent.review_criteria import (
    REVIEW_CRITERIA,
    COMPLETENESS_CHECKLIST,
    SECTION_GUIDELINES,
)

# Claude Code reads these and reviews the draft
# Then updates sections based on review feedback
```

## Tips

- **Let Claude Code drive**: Don't try to call every function manually. Tell Claude Code what you want and it will pick the right tools.
- **Iterate on sections**: Use `save_section()` to overwrite sections as you refine them. `compile_paper()` always reads the latest from disk.
- **Use the workflow tracker**: It helps Claude Code remember where you are across sessions.
- **Check your stats**: The metrics module provides proper statistical tests with effect sizes — use them in your results section.
