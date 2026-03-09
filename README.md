# Research Agent

**PhD-level research paper toolkit — Claude Code as brain, pure Python functions as tools.**

> The anti-framework for AI-assisted scientific writing. No multi-agent orchestration, no API costs, no token overhead. Just callable Python functions that an LLM orchestrator (Claude Code, Cursor, etc.) uses to produce publication-ready manuscripts.

## Why This Exists

Every "AI research agent" framework creates a complex multi-agent system where LLMs call other LLMs, pass messages, and burn tokens coordinating. This is backwards.

**Research Agent takes a different approach:**

```
┌──────────────────────────────────┐
│     Claude Code (The Brain)      │
│                                  │
│  Makes decisions, writes prose,  │
│  calls functions as needed       │
└──────────┬───────────────────────┘
           │ imports & calls
           ▼
┌──────────────────────────────────┐
│     research_agent (The Tools)   │
│                                  │
│  figures.py   metrics.py         │
│  docx_builder.py  workflow.py    │
│  review_criteria.py  utils.py    │
│                                  │
│  Pure Python. Zero API calls.    │
└──────────────────────────────────┘
           │ reads & writes
           ▼
┌──────────────────────────────────┐
│        state/ (On Disk)          │
│                                  │
│  workflow.json  sections/*.txt   │
│  results/*.json figures/*.png    │
│  drafts/*.docx                   │
└──────────────────────────────────┘
```

The LLM is the orchestrator. The toolkit is pure computation. State lives on disk.

## Quick Start

### Install

```bash
pip install research-agent
```

Or from source:

```bash
git clone https://github.com/jaron-mo/research-agent.git
cd research-agent
pip install -e .
```

### Use with Claude Code

Once installed, Claude Code can import and use the toolkit directly:

```
You: "Set up a new paper project for my image classification study with 5-fold CV"

Claude Code:
  → from research_agent.config import ProjectConfig
  → cfg = ProjectConfig(name="image-clf", papers={1: "classification"}, ...)
  → cfg.ensure_dirs()
  → Created project structure at ./state/
```

```
You: "Compute stats comparing model A vs baseline and generate the figures"

Claude Code:
  → from research_agent.metrics import run_group_comparison
  → comparison = run_group_comparison(model_a, baseline, labels=("Ours", "Baseline"))
  → Wilcoxon p=0.0012, Cohen's d=2.31
  → from research_agent.figures import create_metric_charts
  → Saved: state/paper1_classification/figures/fig_metrics.png
```

```
You: "Compile the paper into a Word doc"

Claude Code:
  → from research_agent.docx_builder import compile_paper
  → Compiled: state/paper1_classification/drafts/paper.docx
```

## Modules

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| **config** | Project layout, per-paper directories | `ProjectConfig`, `ensure_dirs()` |
| **utils** | File I/O, shell execution | `read_json()`, `write_json()`, `run_shell()` |
| **figures** | Publication-quality matplotlib figures | `create_pipeline_diagram()`, `create_metric_charts()` |
| **metrics** | Statistical tests & evaluation | `run_wilcoxon_test()`, `run_paired_ttest()`, `compute_descriptive_stats()` |
| **docx_builder** | DOCX paper compilation | `compile_paper()`, `insert_figure()`, `insert_table()` |
| **review_criteria** | Peer review rubrics & checklists | `REVIEW_CRITERIA`, `COMPLETENESS_CHECKLIST` |
| **workflow** | DAG-based task state tracking | `Workflow`, `Task`, `next_tasks()` |

## Features

### Publication Defaults
- **300 DPI** figures with journal styles (IEEE, Elsevier, default)
- **Colorblind-friendly** Wong (2011) palette built in
- **Times New Roman, 12pt, double-spaced** DOCX output
- **IMRAD structure** with standard section ordering

### Statistical Analysis
- Wilcoxon signed-rank test (non-parametric)
- Paired t-test with Cohen's d effect size
- Descriptive statistics (mean, std, median, IQR, quartiles)
- Group comparison with multiple test results
- Cross-validation fold statistics

### Workflow Tracking
- DAG-based task dependencies
- Resumable across sessions
- JSON state on disk (no databases)
- Template system for common workflows

### Paper Compilation
- Multi-paper support (N papers, not just 2)
- Inline figure and table insertion
- Section-based assembly from .txt files
- Dynamic paper registration

## Architecture

See [docs/architecture.md](docs/architecture.md) for the full design philosophy.

**TL;DR**: Multi-agent LLM frameworks are expensive and complex. Research Agent makes the LLM the single orchestrator and provides it with pure Python tools. The toolkit does zero LLM API calls — it's just computation, I/O, and formatting.

## Examples

- **[Quick Start](examples/quickstart.py)** — End-to-end demo: config → stats → figures → paper
- **[Demo Paper](examples/demo_paper/)** — Example paper sections showing the expected file structure
- **[Writing Guide](docs/writing_a_paper.md)** — Step-by-step guide for writing a paper

## Configuration

### Project Setup

```python
from research_agent.config import ProjectConfig

cfg = ProjectConfig(
    name="my-study",
    project_dir="/path/to/project",
    papers={
        1: "main-results",
        2: "supplementary",
    },
    constants={
        "NUM_FOLDS": 10,
        "SIGNIFICANCE_LEVEL": 0.05,
    },
)
```

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `RESEARCH_AGENT_PROJECT_DIR` | Override project root | Package parent directory |

### Extending Review Criteria

```python
from research_agent.review_criteria import merge_criteria, merge_checklist

# Add domain-specific review criteria
my_criteria = merge_criteria({
    "clinical_relevance": {
        "weight": 0.10,
        "description": "Is clinical significance clearly articulated?",
    },
})

# Extend the completeness checklist
my_checklist = merge_checklist([
    "Results include Dice similarity coefficient",
    "Methods describe segmentation architecture",
])
```

## Recommended Companion: Claude Scientific Skills

For a comprehensive set of 170+ scientific skills for Claude Code (literature search, database queries, lab automation, and more), see [claude-scientific-skills](https://github.com/K-Dense-AI/claude-scientific-skills) by K-Dense Inc.

## Requirements

- Python >= 3.10
- python-docx >= 1.1.0
- matplotlib >= 3.8.0
- numpy >= 1.24.0
- scipy >= 1.11.0

## License

MIT License. See [LICENSE](LICENSE).

## Contributing

Contributions welcome! Areas where help is needed:

- **LaTeX support** — Add LaTeX compilation alongside DOCX
- **More figure types** — Confusion matrices, ROC curves, Kaplan-Meier
- **Journal templates** — Pre-configured styles for Nature, Science, PLOS, etc.
- **Citation management** — BibTeX integration
- **Test suite** — Unit tests for all modules

## Origin

Born from a real PhD research pipeline that produced two peer-reviewed publications. The original system was tightly coupled to a specific biomedical imaging project. This is the generalized, domain-agnostic version — the architecture pattern that made a single researcher + Claude Code more productive than a traditional multi-person writing team.
