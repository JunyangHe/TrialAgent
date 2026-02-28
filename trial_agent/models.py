from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SearchSpec:
    raw_request: str
    conditions: list[str] = field(default_factory=list)
    interventions: list[str] = field(default_factory=list)
    sponsors: list[str] = field(default_factory=list)
    phases: list[str] = field(default_factory=list)
    statuses: list[str] = field(default_factory=list)
    geographies: list[str] = field(default_factory=list)
    must_have_fields: list[str] = field(default_factory=list)
    target_k: int = 25
    unresolved_ambiguities: list[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass(slots=True)
class QueryAction:
    tool: str
    query: str
    filters: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Plan:
    tool_order: list[str] = field(default_factory=lambda: ["ctgov_v2", "biomcp", "who_ictrp"])
    query_queue: list[QueryAction] = field(default_factory=list)
    ranking_policy: str = "relevance_then_recency"
    stop_conditions: dict[str, Any] = field(default_factory=dict)
    fallback_policy: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CandidateTrial:
    trial_key: str
    title: str
    source: str
    found_by_query: str
    relevance_score: float = 0.0
    identifiers: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class TrialRecord:
    trial_key: str
    identifiers: dict[str, str] = field(default_factory=dict)
    condition: list[str] = field(default_factory=list)
    interventions: list[str] = field(default_factory=list)
    molecular_targets: list[str] = field(default_factory=list)
    sponsor: str | None = None
    clinical_status: str | None = None
    phase: str | None = None
    study_type: str | None = None
    start_date: str | None = None
    primary_completion_date: str | None = None
    outcomes: dict[str, Any] = field(default_factory=dict)
    summary: str | None = None
    locations: list[str] = field(default_factory=list)
    evidence_snippets: list[dict[str, Any]] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    quality_flags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class QueryAttempt:
    tool: str
    query: str
    filters: dict[str, Any]
    yielded: int
    error: str | None = None


@dataclass(slots=True)
class QualityReport:
    schema_completeness: float = 0.0
    constraint_pass_rate: float = 0.0
    duplicate_rate: float = 0.0
    missingness_by_field: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


class ToolError(RuntimeError):
    pass
