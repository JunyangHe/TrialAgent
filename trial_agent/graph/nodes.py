from __future__ import annotations

from dataclasses import asdict

from trial_agent.config import RuntimeConfig
from trial_agent.io.jsonl_writer import write_trials_jsonl
from trial_agent.models import CandidateTrial, Plan, QueryAction, QueryAttempt, SearchSpec, TrialRecord, ToolError
from trial_agent.policies.fallbacks import (
    apply_ambiguity_fallback,
    apply_evidence_fallback,
    apply_low_yield_fallback,
    apply_missing_fields_fallback,
    apply_overbroad_fallback,
    seed_plan_defaults,
)
from trial_agent.state import GraphState
from trial_agent.tools.base import ToolAdapter


def parse_request(state: GraphState, config: RuntimeConfig) -> GraphState:
    """LLM role: parser. Replace placeholder with your model call."""
    if config.llm_factory is None:
        spec = SearchSpec(raw_request=state.user_request, target_k=config.default_target_k, confidence=0.25)
    else:
        # TODO: invoke configured LLM parser chain and map to SearchSpec.
        _llm = config.llm_factory()
        spec = SearchSpec(raw_request=state.user_request, target_k=config.default_target_k, confidence=0.5)
    state.search_spec = spec
    return state


def plan_queries(state: GraphState, config: RuntimeConfig) -> GraphState:
    """LLM role: planner. Replace query strategy with your prompt + parser."""
    _ = config
    assert state.search_spec, "search_spec must be set"

    if not state.plan.query_queue:
        seed_query = state.search_spec.raw_request
        state.plan = seed_plan_defaults(
            Plan(
                query_queue=[QueryAction(tool=tool_name, query=seed_query) for tool_name in state.plan.tool_order],
            )
        )
    return state


def act_discover(state: GraphState, tool_registry: dict[str, ToolAdapter]) -> GraphState:
    if not state.plan.query_queue:
        return state

    action = state.plan.query_queue.pop(0)
    tool = tool_registry[action.tool]
    try:
        result = tool.discover(action.query, action.filters)
        for row in result.records:
            trial_key = row.get("trial_key") or row.get("nct_id") or row.get("id")
            if not trial_key:
                continue
            state.candidate_set.setdefault(
                trial_key,
                CandidateTrial(
                    trial_key=trial_key,
                    title=row.get("title", ""),
                    source=action.tool,
                    found_by_query=action.query,
                    identifiers={"primary": trial_key},
                ),
            )
        state.query_history.append(QueryAttempt(tool=action.tool, query=action.query, filters=action.filters, yielded=len(result.records)))
    except Exception as exc:
        state.query_history.append(
            QueryAttempt(tool=action.tool, query=action.query, filters=action.filters, yielded=0, error=str(exc))
        )
        state.errors.append(f"discover:{action.tool}:{exc}")
    return state


def observe_discover(state: GraphState, config: RuntimeConfig) -> GraphState:
    """LLM critic role. This stub applies deterministic fallback triggers."""
    _ = config
    attempts = len(state.query_history)
    candidate_count = len(state.candidate_set)

    if candidate_count == 0 and attempts >= 2:
        apply_low_yield_fallback(state)
    elif candidate_count > state.target_k * 4:
        apply_overbroad_fallback(state)

    if state.search_spec and state.search_spec.unresolved_ambiguities:
        apply_ambiguity_fallback(state)

    return state


def act_fetch_records(state: GraphState, tool_registry: dict[str, ToolAdapter], config: RuntimeConfig) -> GraphState:
    selected = list(state.candidate_set.values())[: config.max_fetch_count]
    for candidate in selected:
        for source in [candidate.source, "ctgov_v2", "aact"]:
            if source not in tool_registry:
                continue
            try:
                result = tool_registry[source].fetch(candidate.trial_key)
                if result.records:
                    state.raw_records[candidate.trial_key] = result.records[0]
                    break
            except ToolError:
                continue
            except Exception as exc:
                state.errors.append(f"fetch:{source}:{candidate.trial_key}:{exc}")
    return state


def normalize_records(state: GraphState) -> GraphState:
    for trial_key, raw in state.raw_records.items():
        record = TrialRecord(
            trial_key=trial_key,
            identifiers=raw.get("identifiers", {"primary": trial_key}),
            condition=raw.get("conditions", []),
            interventions=raw.get("interventions", []),
            sponsor=raw.get("sponsor"),
            clinical_status=raw.get("status"),
            phase=raw.get("phase"),
            study_type=raw.get("study_type"),
            start_date=raw.get("start_date"),
            primary_completion_date=raw.get("primary_completion_date"),
            outcomes=raw.get("outcomes", {}),
            summary=raw.get("summary"),
            provenance={"source": raw.get("source", "unknown")},
        )
        state.trial_records[trial_key] = record
    return state


def decide_enrichment(state: GraphState) -> bool:
    return bool(state.search_spec and state.search_spec.include_evidence)


def act_enrich_evidence(state: GraphState, tool_registry: dict[str, ToolAdapter], config: RuntimeConfig) -> GraphState:
    tool = tool_registry.get("trialstreamer")
    if tool is None:
        return state

    enriched = 0
    for record in list(state.trial_records.values()):
        if enriched >= config.max_enrichment_count:
            break
        try:
            result = tool.enrich(asdict(record))
            if result.records:
                record.evidence_snippets.extend(result.records)
                state.evidence_records[record.trial_key] = result.records
            enriched += 1
        except Exception as exc:
            state.errors.append(f"enrich:trialstreamer:{record.trial_key}:{exc}")

    if enriched == 0 and state.trial_records:
        apply_evidence_fallback(state)

    return state


def validate_records(state: GraphState) -> GraphState:
    if not state.trial_records:
        state.quality.notes.append("no_records_normalized")
        return state

    required = ["identifiers", "condition", "interventions", "sponsor", "clinical_status"]
    missing = {k: 0 for k in required}
    for record in state.trial_records.values():
        for field_name in required:
            if not getattr(record, field_name):
                missing[field_name] += 1
                record.quality_flags.append(f"missing_{field_name}")
    total = len(state.trial_records)
    state.quality.missingness_by_field = {k: v / total for k, v in missing.items()}
    state.quality.schema_completeness = 1 - (sum(missing.values()) / (total * len(required)))

    if any(v > 0 for v in missing.values()):
        apply_missing_fields_fallback(state)

    return state


def write_jsonl(state: GraphState, config: RuntimeConfig) -> GraphState:
    write_trials_jsonl(config.output_jsonl, list(state.trial_records.values()))
    return state
