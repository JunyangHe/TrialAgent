from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from trial_agent.models import CandidateTrial, Plan, QualityReport, QueryAttempt, SearchSpec, TrialRecord


@dataclass(slots=True)
class GraphRuntime:
    iteration: int = 0
    budgets_remaining: dict[str, int] = field(default_factory=dict)
    timestamps: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class GraphState:
    user_request: str
    search_spec: SearchSpec | None = None
    plan: Plan = field(default_factory=Plan)
    query_history: list[QueryAttempt] = field(default_factory=list)
    candidate_set: dict[str, CandidateTrial] = field(default_factory=dict)
    raw_records: dict[str, dict[str, Any]] = field(default_factory=dict)
    trial_records: dict[str, TrialRecord] = field(default_factory=dict)
    quality: QualityReport = field(default_factory=QualityReport)
    memory: dict[str, Any] = field(default_factory=dict)
    runtime: GraphRuntime = field(default_factory=GraphRuntime)
    errors: list[str] = field(default_factory=list)

    @property
    def target_k(self) -> int:
        return self.search_spec.target_k if self.search_spec else 25
