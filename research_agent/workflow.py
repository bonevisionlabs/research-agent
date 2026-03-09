"""
Research Agent Toolkit -- Workflow Engine.

DAG-based task state tracker for multi-agent research paper workflows.

The workflow engine is a *pure state tracker* -- it never executes tasks
itself.  An external orchestrator (Claude Code, a script, a CI pipeline)
reads the state, decides which task to run next, performs the work, and
then marks the task as completed or failed.

Key concepts
------------
* **Task** -- a unit of work with an id, agent assignment, and a list of
  dependency ids that must complete before the task can start.
* **Workflow** -- an ordered collection of Tasks forming a DAG.  Persists
  to/from a single JSON file so state survives across sessions.
* **Template** -- a reusable workflow definition (JSON or Python dict)
  that can be instantiated into a fresh Workflow.

Usage
-----
    from research_agent.workflow import Workflow

    # Create from the built-in template
    wf = Workflow.from_template()
    wf.save("state/workflow.json")

    # Load existing state
    wf = Workflow.load("state/workflow.json")

    # Query next runnable tasks
    for task in wf.next_tasks():
        print(task.id, task.name, task.agent)

    # Mark lifecycle
    wf.start_task("lit_search")
    # ... orchestrator does the work ...
    wf.complete_task("lit_search", score=0.92)
    wf.save("state/workflow.json")

    # Progress snapshot
    print(wf.summary())
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Task dataclass
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """A single unit of work inside a workflow DAG.

    Attributes
    ----------
    id : str
        Unique identifier (e.g. ``"lit_search"``).
    name : str
        Human-readable display name.
    agent : str
        Agent role responsible for execution (e.g. ``"literature"``,
        ``"evaluation"``, ``"figures"``, ``"writer"``).
    status : str
        One of ``"pending"``, ``"in_progress"``, ``"completed"``,
        ``"failed"``.
    depends_on : list[str]
        Task ids that must reach ``"completed"`` before this task is
        eligible to run.
    started_at : str | None
        ISO-8601 timestamp of when the task was started.
    completed_at : str | None
        ISO-8601 timestamp of when the task finished (success or failure).
    duration_s : float | None
        Wall-clock seconds between start and completion.
    score : float | None
        Optional quality score assigned on completion (0.0 -- 1.0).
    error : str | None
        Error description if the task failed.
    """

    id: str
    name: str
    agent: str
    status: str = "pending"
    depends_on: List[str] = field(default_factory=list)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_s: Optional[float] = None
    score: Optional[float] = None
    error: Optional[str] = None
    iteration: int = 0
    feedback: Optional[Dict[str, Any]] = None
    max_iterations: int = 3

    # -- validation ---------------------------------------------------------

    _VALID_STATUSES = frozenset({"pending", "in_progress", "completed", "failed"})

    def __post_init__(self) -> None:
        if self.status not in self._VALID_STATUSES:
            raise ValueError(
                f"Invalid status '{self.status}' for task '{self.id}'.  "
                f"Must be one of {sorted(self._VALID_STATUSES)}."
            )

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe dictionary representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Task:
        """Construct a Task from a dictionary (e.g. parsed JSON)."""
        return cls(
            id=data["id"],
            name=data["name"],
            agent=data["agent"],
            status=data.get("status", "pending"),
            depends_on=list(data.get("depends_on") or []),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            duration_s=data.get("duration_s"),
            score=data.get("score"),
            error=data.get("error"),
            iteration=data.get("iteration", 0),
            feedback=data.get("feedback"),
            max_iterations=data.get("max_iterations", 3),
        )

    # -- display ------------------------------------------------------------

    def __repr__(self) -> str:
        deps = f", deps={self.depends_on}" if self.depends_on else ""
        return f"Task({self.id!r}, agent={self.agent!r}, status={self.status!r}{deps})"


# ---------------------------------------------------------------------------
# DAG validation errors
# ---------------------------------------------------------------------------

class WorkflowValidationError(Exception):
    """Raised when a workflow DAG is structurally invalid."""


# ---------------------------------------------------------------------------
# Workflow class
# ---------------------------------------------------------------------------

class Workflow:
    """DAG-based workflow state tracker.

    A Workflow holds an ordered list of :class:`Task` objects and exposes
    methods for querying runnable tasks, marking lifecycle transitions,
    and persisting state to JSON.

    Parameters
    ----------
    tasks : list[Task]
        The tasks that comprise the workflow.
    metadata : dict, optional
        Arbitrary metadata stored alongside the task list (e.g.
        ``{"paper": "my-study", "created": "2026-01-15"}``).

    Raises
    ------
    WorkflowValidationError
        If the task graph contains cycles or dangling dependency
        references.
    """

    def __init__(
        self,
        tasks: Sequence[Task],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._tasks: Dict[str, Task] = {}
        self.metadata: Dict[str, Any] = dict(metadata or {})

        for t in tasks:
            if t.id in self._tasks:
                raise WorkflowValidationError(
                    f"Duplicate task id: '{t.id}'"
                )
            self._tasks[t.id] = t

        self._validate()

    # -- persistence --------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Persist the workflow state to a JSON file.

        Creates parent directories as needed.  Returns the resolved path.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": self.metadata,
            "tasks": [t.to_dict() for t in self._tasks.values()],
        }
        p.write_text(
            json.dumps(payload, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        return p.resolve()

    @classmethod
    def load(cls, path: str | Path) -> Workflow:
        """Load a workflow from a previously saved JSON file.

        Parameters
        ----------
        path : str | Path
            Path to the JSON file.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Workflow file not found: {p}")

        data = json.loads(p.read_text(encoding="utf-8"))
        tasks = [Task.from_dict(td) for td in data.get("tasks", [])]
        metadata = data.get("metadata", {})
        return cls(tasks=tasks, metadata=metadata)

    @classmethod
    def from_template(
        cls,
        template_path: Optional[str | Path] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Workflow:
        """Create a fresh workflow from a template JSON file.

        If *template_path* is not given, the built-in default template
        shipped with the package (``templates/paper_workflow.json``) is
        used.

        All task statuses are reset to ``"pending"`` and timestamps are
        cleared.

        Parameters
        ----------
        template_path : str | Path | None
            Path to a template JSON file.  Defaults to the built-in
            ``paper_workflow.json``.
        metadata : dict | None
            Optional metadata to attach.  Merged on top of any metadata
            already present in the template.
        """
        if template_path is None:
            # Built-in template lives at <package_root>/templates/paper_workflow.json
            template_path = (
                Path(__file__).resolve().parent.parent
                / "templates"
                / "paper_workflow.json"
            )
        else:
            template_path = Path(template_path)

        if not template_path.exists():
            raise FileNotFoundError(
                f"Workflow template not found: {template_path}"
            )

        data = json.loads(template_path.read_text(encoding="utf-8"))

        # Build tasks, resetting runtime state
        tasks: list[Task] = []
        for td in data.get("tasks", []):
            tasks.append(Task(
                id=td["id"],
                name=td["name"],
                agent=td["agent"],
                status="pending",
                depends_on=list(td.get("depends_on") or []),
            ))

        # Merge metadata
        merged_meta = dict(data.get("metadata", {}))
        if metadata:
            merged_meta.update(metadata)

        return cls(tasks=tasks, metadata=merged_meta)

    # -- DAG validation -----------------------------------------------------

    def _validate(self) -> None:
        """Validate the task graph.

        Checks:
        1. Every dependency reference points to an existing task id.
        2. The graph is acyclic (topological sort via Kahn's algorithm).

        Raises
        ------
        WorkflowValidationError
            On dangling references or cycles.
        """
        task_ids = set(self._tasks.keys())

        # Check for dangling dependencies
        for t in self._tasks.values():
            for dep_id in t.depends_on:
                if dep_id not in task_ids:
                    raise WorkflowValidationError(
                        f"Task '{t.id}' depends on '{dep_id}', "
                        f"which does not exist in the workflow."
                    )

        # Cycle detection via Kahn's algorithm (topological sort)
        in_degree: Dict[str, int] = {tid: 0 for tid in task_ids}
        adjacency: Dict[str, list[str]] = {tid: [] for tid in task_ids}

        for t in self._tasks.values():
            for dep_id in t.depends_on:
                adjacency[dep_id].append(t.id)
                in_degree[t.id] += 1

        queue: deque[str] = deque(
            tid for tid, deg in in_degree.items() if deg == 0
        )
        visited = 0

        while queue:
            node = queue.popleft()
            visited += 1
            for neighbor in adjacency[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if visited != len(task_ids):
            # Find the tasks involved in cycle(s)
            cycle_tasks = [
                tid for tid, deg in in_degree.items() if deg > 0
            ]
            raise WorkflowValidationError(
                f"Workflow contains a cycle involving tasks: {cycle_tasks}"
            )

    # -- task accessors -----------------------------------------------------

    @property
    def tasks(self) -> List[Task]:
        """Return all tasks in insertion order."""
        return list(self._tasks.values())

    def get_task(self, task_id: str) -> Task:
        """Return a task by id.

        Raises
        ------
        KeyError
            If *task_id* does not exist.
        """
        try:
            return self._tasks[task_id]
        except KeyError:
            raise KeyError(
                f"Task '{task_id}' not found.  "
                f"Available: {sorted(self._tasks.keys())}"
            ) from None

    def tasks_by_agent(self, agent: str) -> List[Task]:
        """Return all tasks assigned to a given agent role."""
        return [t for t in self._tasks.values() if t.agent == agent]

    def tasks_by_status(self, status: str) -> List[Task]:
        """Return all tasks with a given status."""
        return [t for t in self._tasks.values() if t.status == status]

    # -- DAG queries --------------------------------------------------------

    def next_tasks(self) -> List[Task]:
        """Return tasks that are ready to run.

        A task is runnable when:
        * Its status is ``"pending"``.
        * All of its dependencies have status ``"completed"``.

        Returns
        -------
        list[Task]
            Runnable tasks in insertion order.
        """
        runnable: list[Task] = []
        for t in self._tasks.values():
            if t.status != "pending":
                continue
            deps_met = all(
                self._tasks[dep_id].status == "completed"
                for dep_id in t.depends_on
            )
            if deps_met:
                runnable.append(t)
        return runnable

    def topological_order(self) -> List[Task]:
        """Return tasks in a valid topological execution order.

        This is the order in which tasks *could* be executed sequentially
        while respecting all dependency constraints.
        """
        in_degree: Dict[str, int] = {tid: 0 for tid in self._tasks}
        adjacency: Dict[str, list[str]] = {tid: [] for tid in self._tasks}

        for t in self._tasks.values():
            for dep_id in t.depends_on:
                adjacency[dep_id].append(t.id)
                in_degree[t.id] += 1

        queue: deque[str] = deque(
            tid for tid, deg in in_degree.items() if deg == 0
        )
        order: list[Task] = []

        while queue:
            node = queue.popleft()
            order.append(self._tasks[node])
            for neighbor in adjacency[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return order

    def is_blocked(self, task_id: str) -> bool:
        """Check whether a task is blocked by failed dependencies.

        A task is blocked if any of its transitive dependencies have
        status ``"failed"``.
        """
        task = self.get_task(task_id)
        if task.status in ("completed", "failed"):
            return False

        visited: set[str] = set()
        stack = list(task.depends_on)

        while stack:
            dep_id = stack.pop()
            if dep_id in visited:
                continue
            visited.add(dep_id)

            dep = self._tasks[dep_id]
            if dep.status == "failed":
                return True
            stack.extend(dep.depends_on)

        return False

    # -- lifecycle transitions ----------------------------------------------

    def start_task(self, task_id: str) -> Task:
        """Mark a task as in-progress with a start timestamp.

        Parameters
        ----------
        task_id : str
            The task to start.

        Returns
        -------
        Task
            The updated task.

        Raises
        ------
        ValueError
            If the task is not in ``"pending"`` status or its
            dependencies are not all completed.
        """
        task = self.get_task(task_id)

        if task.status != "pending":
            raise ValueError(
                f"Cannot start task '{task_id}': "
                f"status is '{task.status}', expected 'pending'."
            )

        # Verify dependencies
        for dep_id in task.depends_on:
            dep = self._tasks[dep_id]
            if dep.status != "completed":
                raise ValueError(
                    f"Cannot start task '{task_id}': "
                    f"dependency '{dep_id}' has status '{dep.status}' "
                    f"(must be 'completed')."
                )

        task.status = "in_progress"
        task.started_at = _now_iso()
        return task

    def complete_task(
        self,
        task_id: str,
        score: Optional[float] = None,
    ) -> Task:
        """Mark a task as successfully completed.

        Automatically computes *duration_s* from the start timestamp.

        Parameters
        ----------
        task_id : str
            The task to complete.
        score : float | None
            Optional quality score (0.0 -- 1.0).

        Returns
        -------
        Task
            The updated task.

        Raises
        ------
        ValueError
            If the task is not ``"in_progress"``.
        """
        task = self.get_task(task_id)

        if task.status != "in_progress":
            raise ValueError(
                f"Cannot complete task '{task_id}': "
                f"status is '{task.status}', expected 'in_progress'."
            )

        task.status = "completed"
        task.completed_at = _now_iso()
        task.score = score

        if task.started_at:
            start = datetime.fromisoformat(task.started_at)
            end = datetime.fromisoformat(task.completed_at)
            task.duration_s = round((end - start).total_seconds(), 2)

        return task

    def fail_task(
        self,
        task_id: str,
        error: Optional[str] = None,
    ) -> Task:
        """Mark a task as failed.

        Parameters
        ----------
        task_id : str
            The task to fail.
        error : str | None
            Description of the failure.

        Returns
        -------
        Task
            The updated task.

        Raises
        ------
        ValueError
            If the task is not ``"in_progress"``.
        """
        task = self.get_task(task_id)

        if task.status != "in_progress":
            raise ValueError(
                f"Cannot fail task '{task_id}': "
                f"status is '{task.status}', expected 'in_progress'."
            )

        task.status = "failed"
        task.completed_at = _now_iso()
        task.error = error

        if task.started_at:
            start = datetime.fromisoformat(task.started_at)
            end = datetime.fromisoformat(task.completed_at)
            task.duration_s = round((end - start).total_seconds(), 2)

        return task

    def reset_task(self, task_id: str) -> Task:
        """Reset a task back to ``"pending"`` status.

        Clears all runtime fields (timestamps, score, error).  Useful
        for retrying a failed task.

        Parameters
        ----------
        task_id : str
            The task to reset.

        Returns
        -------
        Task
            The reset task.
        """
        task = self.get_task(task_id)
        task.status = "pending"
        task.started_at = None
        task.completed_at = None
        task.duration_s = None
        task.score = None
        task.error = None
        return task

    def retry_task(self, task_id: str) -> Task:
        """Reset a task for retry, incrementing the iteration counter.

        Unlike :meth:`reset_task`, this preserves ``score`` and ``feedback``
        from the previous attempt for comparison, and increments
        ``iteration``.

        Parameters
        ----------
        task_id : str
            The task to retry.

        Returns
        -------
        Task
            The task reset to ``"pending"`` with ``iteration`` incremented.

        Raises
        ------
        ValueError
            If the task has reached ``max_iterations``.
        """
        task = self.get_task(task_id)
        if task.iteration >= task.max_iterations:
            raise ValueError(
                f"Task '{task_id}' has reached max iterations "
                f"({task.max_iterations}).  Accept or reset manually."
            )
        task.iteration += 1
        task.status = "pending"
        task.started_at = None
        task.completed_at = None
        task.duration_s = None
        task.error = None
        return task

    def tasks_needing_review(self) -> List[Task]:
        """Return completed tasks that have no feedback attached yet."""
        return [
            t for t in self._tasks.values()
            if t.status == "completed" and t.feedback is None
        ]

    def set_feedback(self, task_id: str, feedback: Dict[str, Any]) -> Task:
        """Attach review feedback to a completed task.

        Parameters
        ----------
        task_id : str
            The task to annotate.
        feedback : dict
            Review summary (typically from ``ReviewResult.to_dict()``).

        Returns
        -------
        Task
            The updated task.
        """
        task = self.get_task(task_id)
        task.feedback = feedback
        return task

    # -- progress & summary -------------------------------------------------

    def progress(self) -> Dict[str, Any]:
        """Return a progress snapshot of the workflow.

        Returns
        -------
        dict
            Keys: ``total``, ``pending``, ``in_progress``, ``completed``,
            ``failed``, ``percent_complete``, ``is_finished``,
            ``agents`` (per-agent breakdown).
        """
        counts: Dict[str, int] = {
            "pending": 0,
            "in_progress": 0,
            "completed": 0,
            "failed": 0,
        }
        agent_counts: Dict[str, Dict[str, int]] = {}

        for t in self._tasks.values():
            counts[t.status] += 1

            if t.agent not in agent_counts:
                agent_counts[t.agent] = {
                    "total": 0, "completed": 0, "failed": 0,
                }
            agent_counts[t.agent]["total"] += 1
            if t.status in ("completed", "failed"):
                agent_counts[t.agent][t.status] += 1

        total = len(self._tasks)
        completed = counts["completed"]
        pct = round(100.0 * completed / total, 1) if total > 0 else 0.0
        is_finished = counts["pending"] == 0 and counts["in_progress"] == 0

        return {
            "total": total,
            **counts,
            "percent_complete": pct,
            "is_finished": is_finished,
            "agents": agent_counts,
        }

    def summary(self) -> str:
        """Return a human-readable multi-line progress summary."""
        p = self.progress()
        lines: list[str] = []
        lines.append(
            f"Workflow: {p['completed']}/{p['total']} tasks complete "
            f"({p['percent_complete']}%)"
        )

        if p["failed"] > 0:
            lines.append(f"  Failed: {p['failed']}")
        if p["in_progress"] > 0:
            lines.append(f"  In progress: {p['in_progress']}")

        # Per-agent breakdown
        lines.append("")
        lines.append("  Agent breakdown:")
        for agent_name, ac in sorted(p["agents"].items()):
            lines.append(
                f"    {agent_name}: "
                f"{ac['completed']}/{ac['total']} done"
                + (f", {ac['failed']} failed" if ac["failed"] else "")
            )

        # Next up
        runnable = self.next_tasks()
        if runnable:
            lines.append("")
            lines.append("  Next runnable:")
            for t in runnable:
                lines.append(f"    - {t.id} ({t.agent}): {t.name}")

        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._tasks)

    def __repr__(self) -> str:
        p = self.progress()
        return (
            f"Workflow({p['completed']}/{p['total']} complete, "
            f"{p['failed']} failed)"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Template helpers
# ---------------------------------------------------------------------------

DEFAULT_PAPER_WORKFLOW: List[Dict[str, Any]] = [
    # -- Literature agent --
    {
        "id": "lit_search",
        "name": "Literature Search",
        "agent": "literature",
        "depends_on": [],
    },
    {
        "id": "lit_analysis",
        "name": "Literature Analysis & Synthesis",
        "agent": "literature",
        "depends_on": ["lit_search"],
    },
    {
        "id": "lit_gaps",
        "name": "Research Gap Identification",
        "agent": "literature",
        "depends_on": ["lit_analysis"],
    },
    {
        "id": "lit_related_work",
        "name": "Related Work Section Draft",
        "agent": "literature",
        "depends_on": ["lit_gaps"],
    },
    # -- Evaluation agent --
    {
        "id": "eval_metrics",
        "name": "Compute Evaluation Metrics",
        "agent": "evaluation",
        "depends_on": [],
    },
    {
        "id": "eval_stats",
        "name": "Statistical Significance Tests",
        "agent": "evaluation",
        "depends_on": ["eval_metrics"],
    },
    {
        "id": "eval_comparison",
        "name": "Baseline Comparison Analysis",
        "agent": "evaluation",
        "depends_on": ["eval_stats"],
    },
    {
        "id": "eval_tables",
        "name": "Generate Result Tables",
        "agent": "evaluation",
        "depends_on": ["eval_comparison"],
    },
    # -- Figures agent --
    {
        "id": "fig_pipeline",
        "name": "Pipeline / Architecture Diagram",
        "agent": "figures",
        "depends_on": [],
    },
    {
        "id": "fig_results",
        "name": "Result Plots & Visualizations",
        "agent": "figures",
        "depends_on": ["eval_metrics"],
    },
    {
        "id": "fig_comparison",
        "name": "Comparison Figures",
        "agent": "figures",
        "depends_on": ["eval_comparison"],
    },
    # -- Writer agent --
    {
        "id": "write_intro",
        "name": "Write Introduction",
        "agent": "writer",
        "depends_on": ["lit_gaps"],
    },
    {
        "id": "write_methods",
        "name": "Write Methods Section",
        "agent": "writer",
        "depends_on": ["fig_pipeline"],
    },
    {
        "id": "write_results",
        "name": "Write Results Section",
        "agent": "writer",
        "depends_on": ["eval_tables", "fig_results", "fig_comparison"],
    },
    {
        "id": "write_discussion",
        "name": "Write Discussion",
        "agent": "writer",
        "depends_on": [
            "write_intro",
            "write_results",
            "lit_related_work",
        ],
    },
    # -- Reviewer agent --
    {
        "id": "review_intro",
        "name": "Review Introduction",
        "agent": "reviewer",
        "depends_on": ["write_intro"],
    },
    {
        "id": "review_methods",
        "name": "Review Methods",
        "agent": "reviewer",
        "depends_on": ["write_methods"],
    },
    {
        "id": "review_results",
        "name": "Review Results",
        "agent": "reviewer",
        "depends_on": ["write_results"],
    },
    {
        "id": "review_discussion",
        "name": "Review Discussion",
        "agent": "reviewer",
        "depends_on": ["write_discussion"],
    },
    # -- Compilation & final review --
    {
        "id": "compile_draft",
        "name": "Compile Full Draft",
        "agent": "writer",
        "depends_on": [
            "review_intro",
            "review_methods",
            "review_results",
            "review_discussion",
        ],
    },
    {
        "id": "review_draft",
        "name": "Review Full Draft",
        "agent": "reviewer",
        "depends_on": ["compile_draft"],
    },
    {
        "id": "apply_learnings",
        "name": "Apply Learnings & Archive",
        "agent": "reviewer",
        "depends_on": ["review_draft"],
    },
]


def create_workflow(
    tasks: Optional[List[Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Workflow:
    """Create a new Workflow from a task definition list.

    Parameters
    ----------
    tasks : list[dict] | None
        Task definitions.  Each dict must have ``id``, ``name``,
        ``agent``, and optionally ``depends_on``.  Defaults to
        :data:`DEFAULT_PAPER_WORKFLOW`.
    metadata : dict | None
        Arbitrary metadata to attach.

    Returns
    -------
    Workflow
        A fresh workflow with all tasks in ``"pending"`` status.
    """
    task_defs = tasks if tasks is not None else DEFAULT_PAPER_WORKFLOW
    task_objects = [
        Task(
            id=td["id"],
            name=td["name"],
            agent=td["agent"],
            depends_on=list(td.get("depends_on") or []),
        )
        for td in task_defs
    ]
    return Workflow(tasks=task_objects, metadata=metadata or {})
