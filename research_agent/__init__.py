"""
Research Agent — PhD-Level Research Paper Toolkit.

A pure-Python toolkit for producing publication-ready scientific papers.
Claude Code (or any LLM) acts as the brain; these modules are the callable
tools — no API keys required, no multi-agent overhead.

Modules:
    config          — Project layout and per-paper directory management
    utils           — File I/O, shell execution, experiment data loaders
    figures         — Publication-quality matplotlib figure generation
    docx_builder    — DOCX document assembly (python-docx)
    metrics         — Statistical tests and evaluation helpers
    review_criteria — Peer review rubrics and completeness checklists
    workflow        — DAG-based task state tracking

Usage (from Claude Code or any orchestrator):
    from research_agent.config import ProjectConfig
    from research_agent.figures import create_pipeline_diagram
    from research_agent.metrics import run_wilcoxon_test
    from research_agent.docx_builder import compile_paper
"""

__version__ = "0.1.0"
