from __future__ import annotations

import json
import re

from trial_agent.config import RuntimeConfig, _dbg
from trial_agent.io.jsonl_writer import write_trials_jsonl
from trial_agent.models import CandidateTrial, Plan, QueryAction, QueryAttempt, SearchSpec, TrialRecord, ToolError
from trial_agent.policies.fallbacks import (
    apply_ambiguity_fallback,
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
        def _to_str_list(value: object) -> list[str]:
            if value is None:
                return []
            if isinstance(value, str):
                return [value] if value else []
            if isinstance(value, (list, tuple, set)):
                return [str(item).strip() for item in value if str(item).strip()]
            return [str(value).strip()] if str(value).strip() else []

        try:
            llm = config.llm_factory()
            parser_prompt = (
                "Extract a structured search spec from the user request. Return JSON only with keys: "
                "conditions, interventions, sponsors, phases, statuses, geographies, must_have_fields, "
                "target_k, unresolved_ambiguities, confidence.\n"
                f"User request: {state.user_request}"
            )
            response = llm.invoke(parser_prompt)
            content = getattr(response, "content", response)
            if isinstance(content, list):
                content = "\n".join(
                    chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
                    for chunk in content
                )
            text = str(content).strip()
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
            payload = json.loads(text)

            target_k = payload.get("target_k", config.default_target_k)
            try:
                parsed_target_k = int(target_k)
            except (TypeError, ValueError):
                parsed_target_k = config.default_target_k
            if parsed_target_k <= 0:
                parsed_target_k = config.default_target_k

            confidence = payload.get("confidence", 0.5)
            try:
                parsed_confidence = float(confidence)
            except (TypeError, ValueError):
                parsed_confidence = 0.5
            parsed_confidence = max(0.0, min(1.0, parsed_confidence))

            spec = SearchSpec(
                raw_request=state.user_request,
                conditions=_to_str_list(payload.get("conditions")),
                interventions=_to_str_list(payload.get("interventions")),
                sponsors=_to_str_list(payload.get("sponsors")),
                phases=_to_str_list(payload.get("phases")),
                statuses=_to_str_list(payload.get("statuses")),
                geographies=_to_str_list(payload.get("geographies")),
                must_have_fields=_to_str_list(payload.get("must_have_fields")),
                target_k=parsed_target_k,
                unresolved_ambiguities=_to_str_list(payload.get("unresolved_ambiguities")),
                confidence=parsed_confidence,
            )
        except Exception as exc:
            state.errors.append(f"parse_request:{exc}")
            spec = SearchSpec(raw_request=state.user_request, target_k=config.default_target_k, confidence=0.35)
    state.search_spec = spec
    return state


def plan_queries(state: GraphState, config: RuntimeConfig) -> GraphState:
    """LLM role: planner. Replace query strategy with your prompt + parser."""
    assert state.search_spec, "search_spec must be set"

    if config.tool_order:
        state.plan.tool_order = [t for t in config.tool_order if t in config.enabled_tools]
    if not state.plan.tool_order:
        state.plan.tool_order = list(config.enabled_tools)

    if not state.plan.query_queue:
        seed_query = state.search_spec.raw_request
        filters = {"limit": config.default_discovery_page_size}
        if state.search_spec.geographies:
            filters["geographies"] = state.search_spec.geographies
        state.plan = seed_plan_defaults(
            Plan(
                tool_order=state.plan.tool_order,
                query_queue=[
                    QueryAction(tool=tool_name, query=seed_query, filters=filters)
                    for tool_name in state.plan.tool_order
                ],
            )
        )
    return state


def act_discover(
    state: GraphState, tool_registry: dict[str, ToolAdapter], config: RuntimeConfig | None = None
) -> GraphState:
    if not state.plan.query_queue:
        return state

    action = state.plan.query_queue.pop(0)
    _dbg(f"act_discover: tool={action.tool}, query={action.query!r}, filters={action.filters}")
    tool = tool_registry[action.tool]
    try:
        result = tool.discover(action.query, action.filters)
        _dbg(f"act_discover: {action.tool} returned {len(result.records)} records")
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

        next_token = (result.metadata or {}).get("nextPageToken") or (result.metadata or {}).get("next_page")
        if next_token and result.records:
            pagination_count = state.memory.get("_pagination_count", {})
            key = f"{action.tool}:{action.query}"
            pagination_count[key] = pagination_count.get(key, 0) + 1
            state.memory["_pagination_count"] = pagination_count
            max_pages = getattr(config, "max_pagination_pages", 5) if config else 5
            if pagination_count[key] < max_pages:
                next_filters = dict(action.filters)
                next_filters["pageToken"] = next_token
                state.plan.query_queue.insert(0, QueryAction(tool=action.tool, query=action.query, filters=next_filters))
                _dbg(f"act_discover: queued next page for {action.tool} (page {pagination_count[key] + 1})")
    except Exception as exc:
        _dbg(f"act_discover EXCEPTION: {action.tool}: {exc}")
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
    fetch_order = [t for t in config.fetch_tool_order if t in tool_registry]
    if not fetch_order:
        fetch_order = list(tool_registry.keys())
    _dbg(f"act_fetch: fetching {len(selected)} candidates, fetch_order={fetch_order}")
    for candidate in selected:
        for source in fetch_order:
            if source not in tool_registry:
                _dbg(f"act_fetch: skip {candidate.trial_key} - source {source} not in registry")
                continue
            try:
                result = tool_registry[source].fetch(candidate.trial_key)
                if result.records:
                    state.raw_records[candidate.trial_key] = result.records[0]
                    _dbg(f"act_fetch: got {candidate.trial_key} via {source}")
                    break
                else:
                    _dbg(f"act_fetch: {source}.fetch({candidate.trial_key}) returned 0 records")
            except ToolError as e:
                _dbg(f"act_fetch: {source}.fetch({candidate.trial_key}) ToolError: {e}")
                continue
            except Exception as exc:
                _dbg(f"act_fetch: {source}.fetch({candidate.trial_key}) Exception: {exc}")
                state.errors.append(f"fetch:{source}:{candidate.trial_key}:{exc}")
    return state


def _to_str_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    return [str(value).strip()] if str(value).strip() else []


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
            locations=_to_str_list(raw.get("locations")),
            provenance={"source": raw.get("source", "unknown")},
        )
        state.trial_records[trial_key] = record
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
