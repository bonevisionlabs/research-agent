# Architecture: Claude Code as Brain, Functions as Tools

## The Core Idea

Most AI agent frameworks (CrewAI, AutoGen, LangGraph) create multi-agent LLM orchestration systems where agents call APIs, pass messages to each other, and incur massive token costs.

**Research Agent takes the opposite approach:**

```
┌─────────────────────────────────────────────────┐
│              Claude Code (The Brain)             │
│                                                  │
│  Reads papers, makes decisions, writes content,  │
│  calls Python functions as needed                │
│                                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │
│  │ figures   │ │ metrics  │ │ docx_builder     │ │
│  │ .py       │ │ .py      │ │ .py              │ │
│  │           │ │          │ │                  │ │
│  │ Pure      │ │ Pure     │ │ Pure             │ │
│  │ Python    │ │ Python   │ │ Python           │ │
│  └──────────┘ └──────────┘ └──────────────────┘ │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │
│  │ workflow  │ │ utils    │ │ review_criteria  │ │
│  │ .py       │ │ .py      │ │ .py              │ │
│  └──────────┘ └──────────┘ └──────────────────┘ │
└─────────────────────────────────────────────────┘
                      │
                      ▼
              ┌──────────────┐
              │  state/      │
              │  (on disk)   │
              │              │
              │  workflow.json│
              │  sections/   │
              │  figures/    │
              │  results/    │
              └──────────────┘
```

## Why This Architecture?

| Multi-Agent LLM Systems | Research Agent |
|---|---|
| Agents call LLM APIs ($$) | Functions are pure Python (free) |
| Message passing overhead | Direct function calls |
| Agent orchestration complexity | Claude Code IS the orchestrator |
| Unpredictable agent behavior | Deterministic computations |
| Hard to debug | Standard Python debugging |
| Token-heavy coordination | Zero token coordination overhead |

## The Workflow DAG

The `workflow.py` module tracks a Directed Acyclic Graph (DAG) of tasks:

```
lit_search ──► lit_analysis ──► lit_gaps ──┐
                                           │
eval_metrics ──► eval_stats ──► eval_tables┤
                                           │
                              fig_generate ┤
                                           ▼
                              write_sections ──► review_draft ──► compile_final
```

**The workflow is a state tracker, not an executor.** Claude Code reads the state, decides what to do next, calls the appropriate Python functions, and updates the state.

## Typical Session Flow

```python
# Claude Code reads the workflow state
from research_agent.workflow import Workflow
wf = Workflow.load("state/workflow.json")
next_tasks = wf.get_runnable_tasks()
# → [Task(id="lit_search", ...), Task(id="eval_metrics", ...)]

# Claude Code decides to run evaluation first
wf.start_task("eval_metrics")

# Claude Code calls the metrics module
from research_agent.metrics import compute_descriptive_stats
stats = compute_descriptive_stats(my_errors, label="Model A")

# Claude Code saves results and marks task complete
from research_agent.utils import write_json
write_json("state/evaluation/results/model_a.json", stats)
wf.complete_task("eval_metrics", score=0.95)
wf.save("state/workflow.json")
```

## State on Disk

All state lives in a structured directory hierarchy:

```
state/
├── workflow.json              # Task DAG state
├── evaluation/
│   └── results/               # JSON evaluation results
├── figures/
│   └── generated/             # PNG figure outputs
├── paper1_your-label/
│   ├── sections/              # .txt files per section
│   ├── figures/               # Paper-specific figures
│   └── drafts/                # DOCX outputs
└── paper2_another-label/
    ├── sections/
    ├── figures/
    └── drafts/
```

This means:
- **Reproducibility**: Every step is recorded
- **Resumability**: Crash mid-session? Pick up where you left off
- **Inspectability**: Open any JSON file to see exactly what happened
- **No databases**: Just files on disk

## Design Principles

1. **No LLM API calls in toolkit code** — The toolkit is pure computation. The LLM (Claude Code) is the orchestrator that decides what to compute and when.

2. **State is just files** — No databases, no message queues, no Redis. JSON files on disk that any tool can read.

3. **Functions over classes** — Most toolkit functions are standalone. Pass data in, get results out. No complex object hierarchies.

4. **Fail loudly** — Functions raise exceptions on bad input rather than silently returning empty results. Claude Code handles errors.

5. **Publication-quality defaults** — Figure styles, DOCX formatting, and statistical tests all default to journal-ready settings (300 DPI, Times New Roman, colorblind-friendly palettes).
