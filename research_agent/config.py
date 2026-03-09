"""
Research Agent Toolkit -- Configuration.

Path constants, project layout, and per-paper directory management.
No API keys, no credentials.

Usage
-----
    from research_agent.config import ProjectConfig

    # Minimal -- just a project name
    cfg = ProjectConfig(name="my-study")

    # With multiple papers and custom constants
    cfg = ProjectConfig(
        name="my-study",
        papers={
            1: "language-modeling",
            2: "transfer-learning",
            3: "scaling-laws",
        },
        constants={"NUM_FOLDS": 5, "IMAGE_SIZE": 224},
    )
    cfg.ensure_dirs()

    # Access paths
    cfg.paper_dir(1)          # -> .../state/paper1_language-modeling
    cfg.paper_sections(2)     # -> .../state/paper2_transfer-learning/sections
    cfg.figures_dir            # -> .../state/figures/generated
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


def _resolve_project_dir(explicit: Optional[Path] = None) -> Path:
    """Determine the project root directory.

    Resolution order:
    1. *explicit* argument passed by caller.
    2. ``RESEARCH_AGENT_PROJECT_DIR`` environment variable.
    3. Parent of the ``research_agent/`` package directory (i.e. the
       repository root when installed in development mode).
    """
    if explicit is not None:
        return Path(explicit).resolve()

    env = os.environ.get("RESEARCH_AGENT_PROJECT_DIR")
    if env:
        return Path(env).resolve()

    # Default: assume <project_root>/research_agent/config.py
    return Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Paper directory helpers
# ---------------------------------------------------------------------------

def _paper_slug(paper_id: int, label: Optional[str] = None) -> str:
    """Build a filesystem-safe directory name for a paper.

    Examples
    --------
    >>> _paper_slug(1, "classification")
    'paper1_classification'
    >>> _paper_slug(3)
    'paper3'
    """
    base = f"paper{paper_id}"
    if label:
        safe = label.strip().replace(" ", "-").lower()
        return f"{base}_{safe}"
    return base


# ---------------------------------------------------------------------------
# PaperPaths -- paths for a single paper
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PaperPaths:
    """Resolved directory paths for one paper."""

    root: Path
    sections: Path
    figures: Path
    drafts: Path

    def all_dirs(self) -> list[Path]:
        """Return every directory that should exist on disk."""
        return [self.root, self.sections, self.figures, self.drafts]


# ---------------------------------------------------------------------------
# ProjectConfig
# ---------------------------------------------------------------------------

@dataclass
class ProjectConfig:
    """Central configuration for a research-agent project.

    Parameters
    ----------
    name:
        Human-readable project name (used in logs and summaries, not paths).
    project_dir:
        Explicit project root.  Falls back to ``RESEARCH_AGENT_PROJECT_DIR``
        env var, then to the parent of the installed package directory.
    papers:
        Mapping of ``{paper_id: label}`` for each paper in the project.
        Labels are optional (pass ``None`` or ``""`` for unlabelled papers).
        An empty dict is valid -- the toolkit still works without papers.
    constants:
        Arbitrary key/value pairs for experiment-specific settings
        (fold counts, image sizes, hyperparameters, etc.).  Accessible
        via ``cfg.constants["KEY"]`` or ``cfg.get("KEY", default)``.
    """

    name: str = "research-project"
    project_dir: Optional[Path] = None
    papers: Dict[int, Optional[str]] = field(default_factory=dict)
    constants: Dict[str, Any] = field(default_factory=dict)

    # -- resolved paths (set in __post_init__) ------------------------------
    root: Path = field(init=False)
    state_dir: Path = field(init=False)
    figures_dir: Path = field(init=False)
    results_dir: Path = field(init=False)
    drafts_dir: Path = field(init=False)
    sections_dir: Path = field(init=False)

    _paper_cache: Dict[int, PaperPaths] = field(
        init=False, default_factory=dict, repr=False
    )

    def __post_init__(self) -> None:
        self.root = _resolve_project_dir(self.project_dir)
        self.state_dir = self.root / "state"

        # Shared output directories (legacy / single-paper workflows)
        self.figures_dir = self.state_dir / "figures" / "generated"
        self.results_dir = self.state_dir / "evaluation" / "results"
        self.drafts_dir = self.state_dir / "drafts"
        self.sections_dir = self.state_dir / "sections"

        # Pre-build paper path objects
        for pid, label in self.papers.items():
            self._paper_cache[pid] = self._build_paper_paths(pid, label)

    # -- paper accessors ----------------------------------------------------

    def _build_paper_paths(
        self, paper_id: int, label: Optional[str] = None
    ) -> PaperPaths:
        slug = _paper_slug(paper_id, label)
        base = self.state_dir / slug
        return PaperPaths(
            root=base,
            sections=base / "sections",
            figures=base / "figures",
            drafts=base / "drafts",
        )

    def paper(self, paper_id: int) -> PaperPaths:
        """Return *PaperPaths* for the given paper id.

        Raises ``KeyError`` if the paper was not registered in *papers*.
        """
        try:
            return self._paper_cache[paper_id]
        except KeyError:
            registered = sorted(self._paper_cache) or "none"
            raise KeyError(
                f"Paper {paper_id} not registered.  "
                f"Registered papers: {registered}"
            ) from None

    # Convenience shortcuts
    def paper_dir(self, paper_id: int) -> Path:
        return self.paper(paper_id).root

    def paper_sections(self, paper_id: int) -> Path:
        return self.paper(paper_id).sections

    def paper_figures(self, paper_id: int) -> Path:
        return self.paper(paper_id).figures

    def paper_drafts(self, paper_id: int) -> Path:
        return self.paper(paper_id).drafts

    # -- constants accessor -------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve an experiment constant by name, with optional default."""
        return self.constants.get(key, default)

    # -- directory creation -------------------------------------------------

    def _all_dirs(self) -> list[Path]:
        """Collect every directory the project needs on disk."""
        dirs: list[Path] = [
            self.figures_dir,
            self.results_dir,
            self.drafts_dir,
            self.sections_dir,
        ]
        for pp in self._paper_cache.values():
            dirs.extend(pp.all_dirs())
        return dirs

    def ensure_dirs(self) -> None:
        """Create all output directories (idempotent)."""
        for d in self._all_dirs():
            d.mkdir(parents=True, exist_ok=True)

    # -- repr / summary -----------------------------------------------------

    def summary(self) -> str:
        """Human-readable one-liner for logs."""
        n = len(self._paper_cache)
        papers = f"{n} paper{'s' if n != 1 else ''}"
        return f"ProjectConfig('{self.name}', root={self.root}, {papers})"


# ---------------------------------------------------------------------------
# Module-level convenience: a default singleton for simple scripts
# ---------------------------------------------------------------------------

_default: Optional[ProjectConfig] = None


def init(
    name: str = "research-project",
    project_dir: Optional[Path] = None,
    papers: Optional[Dict[int, Optional[str]]] = None,
    constants: Optional[Dict[str, Any]] = None,
) -> ProjectConfig:
    """Initialise (or re-initialise) the module-level default config.

    Returns the ``ProjectConfig`` instance so callers can use it directly::

        cfg = config.init(name="my-study", papers={1: "pretraining"})
        cfg.ensure_dirs()
    """
    global _default
    _default = ProjectConfig(
        name=name,
        project_dir=project_dir,
        papers=papers or {},
        constants=constants or {},
    )
    return _default


def get_config() -> ProjectConfig:
    """Return the module-level default config, creating one if needed."""
    global _default
    if _default is None:
        _default = ProjectConfig()
    return _default
