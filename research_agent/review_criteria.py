"""
Research Agent — Review Criteria & Checklists.

Domain-agnostic rubrics and checklists for peer-review evaluation of
scientific papers following the IMRAD structure. No LLM calls — just
reference constants that an LLM agent (or human reviewer) consults
when scoring, reviewing, or drafting a manuscript.

The defaults cover standard expectations for any empirical research
paper. Use ``merge_criteria()`` and ``merge_checklist()`` to layer on
domain-specific requirements (medical imaging metrics, NLP benchmarks,
materials-science characterisation, etc.) without touching the base
definitions.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# Review criteria with weights (for scoring guidance)
# ---------------------------------------------------------------------------

REVIEW_CRITERIA: Dict[str, Dict] = {
    "novelty": {
        "weight": 0.15,
        "description": (
            "Are novelty claims supported? Is the contribution clearly "
            "differentiated from prior work?"
        ),
    },
    "methodology": {
        "weight": 0.20,
        "description": (
            "Is the approach clearly described and reproducible? "
            "Are design choices justified?"
        ),
    },
    "results": {
        "weight": 0.20,
        "description": (
            "Do metrics support conclusions? Are statistical tests "
            "appropriate? Are results comprehensive?"
        ),
    },
    "completeness": {
        "weight": 0.15,
        "description": (
            "Are all required elements present? Missing experiments, "
            "references, or details?"
        ),
    },
    "writing_quality": {
        "weight": 0.15,
        "description": (
            "Clarity, grammar, structure, scientific tone, "
            "consistent terminology?"
        ),
    },
    "figures": {
        "weight": 0.10,
        "description": (
            "Are figures informative, publication-quality, properly "
            "labeled and captioned?"
        ),
    },
    "impact": {
        "weight": 0.05,
        "description": (
            "Is real-world significance clearly articulated? Does the "
            "work address a meaningful problem with practical implications?"
        ),
    },
}

# ---------------------------------------------------------------------------
# Paper completeness checklist (domain-agnostic IMRAD)
# ---------------------------------------------------------------------------

COMPLETENESS_CHECKLIST: List[str] = [
    # Abstract
    "Abstract includes Purpose, Methods, Results, Conclusion",
    # Introduction
    "Introduction states the problem and its significance",
    "Introduction clearly articulates the contribution",
    # Related Work / Background
    "Related Work covers the key research areas relevant to the paper",
    "Related Work identifies gaps the current work addresses",
    # Methods
    "Methods describes the proposed approach in reproducible detail",
    "Methods describes model or system architecture and design choices",
    "Methods describes training or fitting procedure (if applicable)",
    "Methods describes evaluation protocol and validation strategy",
    # Experiments
    "Experiments lists dataset details (size, source, preprocessing)",
    "Experiments specifies implementation details (hardware, software, hyperparameters)",
    # Results
    "Results includes primary quantitative metrics with uncertainty estimates",
    "Results includes comparison with relevant baselines or prior work",
    "Results includes ablation studies or sensitivity analysis",
    "Results includes statistical significance tests where appropriate",
    # Discussion
    "Discussion interprets key findings in context of prior work",
    "Discussion addresses limitations honestly",
    "Discussion outlines future work",
    # General
    "Figures are referenced in text",
    "All claims cite supporting evidence or results",
    "References are complete and consistently formatted",
]

# ---------------------------------------------------------------------------
# Section writing guidelines (IMRAD structure, domain-agnostic)
# ---------------------------------------------------------------------------

SECTION_GUIDELINES: Dict[str, Dict] = {
    "abstract": {
        "target_words": (250, 300),
        "structure": ["Purpose", "Methods", "Results", "Conclusion"],
        "notes": (
            "Highlight the core contribution, quantitative results, "
            "and practical significance."
        ),
    },
    "introduction": {
        "target_words": (800, 1000),
        "structure": [
            "Problem statement and motivation",
            "Current limitations of existing approaches",
            "Summary of contributions",
            "Paper organization",
        ],
    },
    "related_work": {
        "target_words": (1500, 2000),
        "subsections": [
            "Research Area A (primary field)",
            "Research Area B (secondary / intersecting field)",
            "Research Area C (methodological foundations)",
        ],
        "notes": (
            "Subsection topics depend on the paper's domain. Replace "
            "the placeholders above with the actual research areas. "
            "Each subsection should end by identifying gaps the current "
            "work addresses."
        ),
    },
    "methods": {
        "target_words": (2000, 2500),
        "subsections": [
            "Data Acquisition and Preprocessing",
            "Proposed Approach / Model Architecture",
            "Training or Fitting Procedure",
            "Evaluation Protocol and Metrics",
        ],
        "notes": (
            "Add or rename subsections to match the paper's methodology. "
            "Every design choice should be justified."
        ),
    },
    "experiments": {
        "target_words": (800, 1000),
        "subsections": [
            "Dataset description",
            "Implementation details",
            "Evaluation metrics",
            "Baseline comparisons",
            "Ablation study setup",
        ],
    },
    "results": {
        "target_words": (1000, 1500),
        "subsections": [
            "Main Quantitative Results",
            "Comparison with Baselines",
            "Ablation Studies",
            "Qualitative Analysis (if applicable)",
        ],
    },
    "discussion": {
        "target_words": (1000, 1500),
        "subsections": [
            "Key findings and significance",
            "Comparison with existing methods",
            "Limitations",
            "Future work",
        ],
    },
    "conclusion": {
        "target_words": (200, 300),
        "notes": (
            "Restate the core contribution, headline results, practical "
            "impact, and future directions."
        ),
    },
}


# ---------------------------------------------------------------------------
# Extension helpers
# ---------------------------------------------------------------------------

def merge_criteria(
    extra: Dict[str, Dict],
    *,
    base: Optional[Dict[str, Dict]] = None,
    normalize_weights: bool = True,
) -> Dict[str, Dict]:
    """Return a new criteria dict with *extra* entries merged into *base*.

    Parameters
    ----------
    extra : dict
        Mapping of criterion name to ``{"weight": float, "description": str}``.
        If a key already exists in *base* the entry is **replaced**.
    base : dict, optional
        Starting criteria. Defaults to a copy of ``REVIEW_CRITERIA``.
    normalize_weights : bool
        If *True* (default), rescale all weights so they sum to 1.0 after
        the merge. Set to *False* to keep weights exactly as provided.

    Returns
    -------
    dict
        A fresh dict — neither *base* nor ``REVIEW_CRITERIA`` is mutated.

    Example
    -------
    >>> medical = merge_criteria({
    ...     "clinical_relevance": {
    ...         "weight": 0.05,
    ...         "description": "Is clinical significance clearly articulated?",
    ...     },
    ... })
    """
    merged = copy.deepcopy(base if base is not None else REVIEW_CRITERIA)
    merged.update(copy.deepcopy(extra))

    if normalize_weights:
        total = sum(c["weight"] for c in merged.values())
        if total > 0:
            for criterion in merged.values():
                criterion["weight"] = round(criterion["weight"] / total, 4)

    return merged


def merge_checklist(
    extra: Sequence[str],
    *,
    base: Optional[List[str]] = None,
    position: str = "append",
) -> List[str]:
    """Return a new checklist with *extra* items merged into *base*.

    Parameters
    ----------
    extra : sequence of str
        Additional checklist items.
    base : list of str, optional
        Starting checklist. Defaults to a copy of ``COMPLETENESS_CHECKLIST``.
    position : {"append", "prepend"}
        Where to insert *extra* items relative to *base*.

    Returns
    -------
    list[str]
        A fresh list — *base* and ``COMPLETENESS_CHECKLIST`` are not mutated.

    Example
    -------
    >>> medical_checks = merge_checklist([
    ...     "Results includes segmentation metrics (Dice, HD95, ASSD)",
    ...     "Results includes comparison with TotalSegmentator",
    ... ])
    """
    merged = list(base if base is not None else COMPLETENESS_CHECKLIST)
    extra_list = list(extra)

    if position == "prepend":
        merged = extra_list + merged
    else:
        merged.extend(extra_list)

    # Deduplicate while preserving order
    seen: set[str] = set()
    deduped: List[str] = []
    for item in merged:
        if item not in seen:
            seen.add(item)
            deduped.append(item)

    return deduped
