from __future__ import annotations

import json
import re

from trial_agent.config import RuntimeConfig, _dbg

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # type: ignore[misc]
        return iterable
from trial_agent.io.jsonl_writer import write_trials_jsonl
from trial_agent.models import CandidateTrial, Plan, QueryAction, QueryAttempt, SearchSpec, TrialRecord, ToolError
from trial_agent.policies.fallbacks import (
    apply_low_yield_fallback,
    apply_overbroad_fallback,
    seed_plan_defaults,
)
from trial_agent.state import GraphState
from trial_agent.tools.base import ToolAdapter


def _to_str_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, (tuple, set)):
        return [str(x).strip() for x in value if str(x).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _to_int(value: object, default: int, *, minimum: int = 0) -> int:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return max(minimum, parsed)


def _extract_llm_text(response: object) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        content = "\n".join(chunk.get("text", "") if isinstance(chunk, dict) else str(chunk) for chunk in content)
    text = str(content).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    if not text.startswith("{"):
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            text = match.group(0)
    return text


def _invoke_llm_json(
    config: RuntimeConfig,
    prompt: str,
    default: dict[str, object],
    *,
    stage: str | None = None,
) -> dict[str, object]:
    if config.llm_factory is None:
        return default
    if stage:
        print(f"[TrialAgent] {stage}...")
    try:
        llm = config.llm_factory()
        response = llm.invoke(prompt)
        text = _extract_llm_text(response)
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else default
    except Exception as exc:
        _dbg(f"llm_json fallback: {exc}")
        return default


def _resolve_tool_order(config: RuntimeConfig) -> list[str]:
    if config.tool_order:
        order = [tool for tool in config.tool_order if tool in config.enabled_tools]
        if order:
            return order
    default_order = ["ctgov_v2", "biomcp", "who_ictrp"]
    order = [tool for tool in default_order if tool in config.enabled_tools]
    if order:
        return order
    return list(config.enabled_tools)


def _to_bool(value: object, default: bool = True) -> bool:
    """Convert payload value to bool; default True when missing or ambiguous."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "yes", "1")
    return bool(value)


def _set_overbroad_warning(state: GraphState, candidate_count: int) -> None:
    threshold = state.target_k * 4
    warning = {"threshold": threshold, "candidates": candidate_count}
    state.memory["overbroad_warning"] = warning
    # keep backwards-compatible key for existing consumers
    state.memory["manual_overbroad_warning"] = warning


def parse_request(state: GraphState, config: RuntimeConfig) -> GraphState:
    """Parse the user request into SearchSpec, optionally using LLM structured extraction."""
    confidence_default = 0.25
    payload: dict[str, object] = {}
    if config.llm_factory is not None:
        parser_prompt = (
            "Extract a structured search spec from the user request. Return JSON only with keys: "
            "conditions, interventions, sponsors, phases, statuses, geographies, must_have_fields, "
            "target_k, unresolved_ambiguities, confidence, is_trial_related.\n"
            "is_trial_related (boolean): true if the request is medically/health-related and could be answered by clinical trials—e.g. disease names (lung cancer, diabetes), drug names, conditions, interventions, sponsors. "
            "false only for clearly unrelated topics: weather, recipes, sports, general chat, non-medical news.\n"
            f"User request: {state.user_request}"
        )
        payload = _invoke_llm_json(config, parser_prompt, {}, stage="[Parser] extracting structured search spec")
        confidence_default = 0.5

        # Reject non-trial requests
        if not _to_bool(payload.get("is_trial_related"), default=True):
            state.memory["abort"] = True
            state.errors.append(
                "Input does not appear to be about clinical trials. "
                "Please ask about clinical trials or medical studies."
            )
            return state

    parsed_target_k = max(
        _to_int(payload.get("target_k"), default=config.default_target_k, minimum=1),
        config.default_target_k,
    )
    confidence_raw = payload.get("confidence", confidence_default)
    try:
        parsed_confidence = float(confidence_raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        parsed_confidence = confidence_default
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
    state.search_spec = spec
    return state


def plan_queries(state: GraphState, config: RuntimeConfig) -> GraphState:
    """Build the deterministic discovery queue from parsed request and config."""
    assert state.search_spec, "search_spec must be set"

    state.plan.tool_order = [t for t in state.plan.tool_order if t in config.enabled_tools]
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


def plan_queries_react(state: GraphState, config: RuntimeConfig) -> GraphState:
    """ReAct planner role: LLM creates tool order, query queue, budgets, and stop policy."""
    assert state.search_spec, "search_spec must be set"
    tool_order = _resolve_tool_order(config)
    spec = state.search_spec
    planner_default: dict[str, object] = {
        "tool_order": tool_order,
        "query_queue": [
            {"tool": tool_name, "query": spec.raw_request, "filters": {"limit": config.default_discovery_page_size}}
            for tool_name in tool_order
        ],
        "budgets": {
            "discovery_searches": config.max_discovery_attempts,
            "full_fetches": config.max_fetch_count,
            "repair_loops": config.react_max_repair_loops,
        },
        "ranking_policy": "relevance_then_recency",
        "stop_conditions": {"min_high_confidence_trials": spec.target_k, "or_budget_exhausted": True},
        "fallback_policy": {
            "low_yield": "fallback_1",
            "overbroad": "fallback_2",
        },
    }
    planner_prompt = (
        "You are a trial-search planner. Return JSON only with keys: tool_order, query_queue, budgets, "
        "ranking_policy, stop_conditions, fallback_policy.\n"
        "Constraints:\n"
        "- tool_order values must be from allowed tools.\n"
        "- query_queue items must have query and optional tool + filters.\n"
        "- keep budgets conservative.\n"
        f"Allowed tools: {tool_order}\n"
        f"SearchSpec: {spec}\n"
    )
    payload = _invoke_llm_json(config, planner_prompt, planner_default, stage="[ReAct Planner] creating query plan")

    raw_tool_order = payload.get("tool_order", tool_order)
    raw_tool_order = raw_tool_order if isinstance(raw_tool_order, list) else tool_order
    chosen_order = [str(t) for t in raw_tool_order if str(t) in tool_order]
    if not chosen_order:
        chosen_order = tool_order

    queue: list[QueryAction] = []
    queue_payload = payload.get("query_queue", [])
    if isinstance(queue_payload, list):
        for item in queue_payload:
            if isinstance(item, str):
                queue.append(QueryAction(tool=chosen_order[0], query=item, filters={"limit": config.default_discovery_page_size}))
                continue
            if not isinstance(item, dict):
                continue
            query = str(item.get("query", "")).strip()
            if not query:
                continue
            tool = str(item.get("tool") or chosen_order[0]).strip()
            if tool not in chosen_order:
                tool = chosen_order[0]
            filters = item.get("filters")
            filters_dict = filters if isinstance(filters, dict) else {}
            filters_dict.setdefault("limit", config.default_discovery_page_size)
            queue.append(QueryAction(tool=tool, query=query, filters=filters_dict))
    if not queue:
        queue = [
            QueryAction(tool=tool_name, query=spec.raw_request, filters={"limit": config.default_discovery_page_size})
            for tool_name in chosen_order
        ]

    state.plan = seed_plan_defaults(
        Plan(
            tool_order=chosen_order,
            query_queue=queue,
            ranking_policy=str(payload.get("ranking_policy") or "relevance_then_recency"),
            stop_conditions=payload.get("stop_conditions") if isinstance(payload.get("stop_conditions"), dict) else {},
            fallback_policy=payload.get("fallback_policy") if isinstance(payload.get("fallback_policy"), dict) else {},
        )
    )
    budgets = payload.get("budgets", {})
    budgets_dict = budgets if isinstance(budgets, dict) else {}
    state.runtime.budgets_remaining = {
        "discovery_searches": _to_int(
            budgets_dict.get("discovery_searches"),
            default=config.max_discovery_attempts,
            minimum=1,
        ),
        "full_fetches": _to_int(budgets_dict.get("full_fetches"), default=config.max_fetch_count, minimum=1),
        "repair_loops": _to_int(budgets_dict.get("repair_loops"), default=config.react_max_repair_loops, minimum=0),
    }
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
    """Apply deterministic discovery fallbacks based on yield and ambiguity."""
    _ = config
    attempts = len(state.query_history)
    candidate_count = len(state.candidate_set)

    if candidate_count == 0 and attempts >= 2:
        state.quality.notes.append("manual_low_yield_detected")
    elif candidate_count > state.target_k * 4:
        _set_overbroad_warning(state, candidate_count)
        apply_overbroad_fallback(state)

    return state


def observe_discover_react(state: GraphState, config: RuntimeConfig) -> GraphState:
    """ReAct critic role: LLM decides continue/replan/fetch and fallback trigger."""
    attempts = len(state.query_history)
    candidate_count = len(state.candidate_set)
    total_yield = sum(item.yielded for item in state.query_history)
    duplicate_rate = ((total_yield - candidate_count) / total_yield) if total_yield > 0 else 0.0
    stop_target = state.target_k

    default_action = "continue_discovery" if state.plan.query_queue else ("fetch" if candidate_count >= stop_target else "continue_discovery")
    default_payload: dict[str, object] = {"action": default_action, "fallback": "none", "add_queries": []}
    critic_prompt = (
        "You are a discovery critic in a ReAct loop. Return JSON only with keys: action, fallback, add_queries, reason.\n"
        "action in ['continue_discovery','fetch']; fallback in ['none','low_yield','overbroad','ambiguity'].\n"
        "add_queries is an optional list of extra text queries.\n"
        f"Metrics: attempts={attempts}, candidates={candidate_count}, target_k={stop_target}, duplicate_rate={duplicate_rate:.3f}, "
        f"queue_len={len(state.plan.query_queue)}.\n"
        f"SearchSpec={state.search_spec}\n"
    )
    payload = _invoke_llm_json(config, critic_prompt, default_payload, stage="[ReAct Critic] evaluating discovery progress")
    action = str(payload.get("action") or default_action).strip()
    fallback = str(payload.get("fallback") or "none").strip().lower()

    if fallback == "low_yield":
        apply_low_yield_fallback(state)
    elif fallback == "overbroad":
        _set_overbroad_warning(state, candidate_count)
        apply_overbroad_fallback(state)
    elif candidate_count > state.target_k * 4:
        _set_overbroad_warning(state, candidate_count)
        apply_overbroad_fallback(state)

    add_queries = payload.get("add_queries", [])
    if isinstance(add_queries, list):
        for q in add_queries:
            query = str(q).strip()
            if not query:
                continue
            for tool_name in state.plan.tool_order:
                state.plan.query_queue.append(
                    QueryAction(tool=tool_name, query=query, filters={"limit": config.default_discovery_page_size})
                )

    if action == "fetch":
        state.plan.query_queue = []
    elif candidate_count == 0 and attempts >= 2 and not state.plan.query_queue and state.plan.tool_order:
        seed = state.search_spec.raw_request if state.search_spec else state.user_request
        state.plan.query_queue.append(
            QueryAction(tool=state.plan.tool_order[0], query=f"{seed} randomized clinical trial", filters={"limit": config.default_discovery_page_size})
        )
    return state


def act_fetch_records(state: GraphState, tool_registry: dict[str, ToolAdapter], config: RuntimeConfig) -> GraphState:
    selected = list(state.candidate_set.values())
    fetch_order = [t for t in config.fetch_tool_order if t in tool_registry]
    if not fetch_order:
        fetch_order = list(tool_registry.keys())
    _dbg(f"act_fetch: fetching {len(selected)} candidates, fetch_order={fetch_order}")
    for candidate in tqdm(selected, desc="Fetching trials", unit="trial"):
        for source in fetch_order:
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


def act_fetch_records_react(state: GraphState, tool_registry: dict[str, ToolAdapter], config: RuntimeConfig) -> GraphState:
    """ReAct fetch node with ranking policy. Fetches all candidates (no limit)."""
    selected = list(state.candidate_set.values())

    def _priority(candidate: CandidateTrial) -> tuple[int, int]:
        score = 0
        title = candidate.title.lower()
        spec = state.search_spec
        if spec:
            for token in spec.conditions + spec.interventions:
                token_norm = token.lower().strip()
                if token_norm and token_norm in title:
                    score += 1
        return (-score, 0)

    selected.sort(key=_priority)
    fetch_order = [tool for tool in config.fetch_tool_order if tool in tool_registry]
    for tool in ("ctgov_v2", "biomcp", "who_ictrp"):
        if tool in tool_registry and tool not in fetch_order:
            fetch_order.append(tool)
    if not fetch_order:
        fetch_order = list(tool_registry.keys())

    _dbg(f"act_fetch: fetching {len(selected)} candidates, fetch_order={fetch_order}")
    for candidate in tqdm(selected, desc="Fetching trials", unit="trial"):
        for source in fetch_order:
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
    for trial_key, raw in tqdm(state.raw_records.items(), desc="Normalizing", unit="record"):
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


def normalize_records_react(state: GraphState, config: RuntimeConfig) -> GraphState:
    """Deterministic normalization with LLM post-normalization for sponsor/interventions."""
    state = normalize_records(state)
    # LLM normalization disabled for now
    # if config.llm_factory is None or not state.trial_records:
    #     return state
    #
    # try:
    #     from tqdm import tqdm
    # except ImportError:
    #     tqdm = lambda x, **kw: x
    #
    # records = list(state.trial_records.values())
    # batch_size = config.react_normalize_batch_size
    # batches = [records[i : i + batch_size] for i in range(0, len(records), batch_size)]
    #
    # for batch in tqdm(batches, desc="[ReAct Normalizer] enriching sponsor/interventions", unit="batch"):
    #     payload_records = [
    #         {
    #             "trial_key": record.trial_key,
    #             "sponsor": record.sponsor,
    #             "interventions": record.interventions,
    #             "condition": record.condition,
    #         }
    #         for record in batch
    #     ]
    #     default_payload: dict[str, object] = {"updates": []}
    #     prompt = (
    #         "Normalize trial sponsor/intervention strings. Return JSON only with key updates where each update has "
    #         "trial_key and optional sponsor/interventions. Do not invent missing values.\n"
    #         f"Records: {payload_records}\n"
    #     )
    #     result = _invoke_llm_json(config, prompt, default_payload, stage=None)
    #     updates = result.get("updates", [])
    #     if isinstance(updates, list):
    #         for item in updates:
    #             if not isinstance(item, dict):
    #                 continue
    #             trial_key = str(item.get("trial_key", "")).strip()
    #             if not trial_key or trial_key not in state.trial_records:
    #                 continue
    #             record = state.trial_records[trial_key]
    #             if "sponsor" in item and item.get("sponsor") is not None:
    #                 sponsor = str(item.get("sponsor")).strip()
    #                 record.sponsor = sponsor or record.sponsor
    #             if "interventions" in item:
    #                 record.interventions = _to_str_list(item.get("interventions")) or record.interventions
    #             record.provenance["normalization"] = "llm_react"
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

    return state


def validate_records_react(state: GraphState, config: RuntimeConfig) -> GraphState:
    """Deterministic quality checks plus LLM QC critic output."""
    state = validate_records(state)
    summary = {
        "trial_count": len(state.trial_records),
        "schema_completeness": state.quality.schema_completeness,
        "missingness_by_field": state.quality.missingness_by_field,
        "notes": state.quality.notes[-6:],
    }
    default_payload: dict[str, object] = {"needs_repair": False, "notes": [], "actions": []}
    prompt = (
        "You are a QC critic for trial records. Return JSON only with keys: needs_repair (bool), notes (list), actions (list).\n"
        f"Quality summary: {summary}\n"
    )
    payload = _invoke_llm_json(config, prompt, default_payload, stage="[ReAct QC Critic] validating trial records")
    state.memory["react_validate"] = payload
    notes = payload.get("notes", [])
    if isinstance(notes, list):
        for note in notes[:3]:
            note_text = str(note).strip()
            if note_text:
                state.quality.notes.append(f"react_qc:{note_text}")
    return state


def repair_or_replan_react(state: GraphState, tool_registry: dict[str, ToolAdapter], config: RuntimeConfig) -> GraphState:
    """ReAct repair/replan node to improve coverage and missing fields under budget."""
    decision = state.memory.get("react_validate", {})
    needs_repair = bool(isinstance(decision, dict) and decision.get("needs_repair"))

    discovery_budget = state.runtime.budgets_remaining.get("discovery_searches", config.max_discovery_attempts)
    if (
        len(state.trial_records) < state.target_k
        and discovery_budget > len(state.query_history)
        and not state.plan.query_queue
        and state.plan.tool_order
    ):
        seed = state.search_spec.raw_request if state.search_spec else state.user_request
        state.plan.query_queue.append(
            QueryAction(
                tool=state.plan.tool_order[0],
                query=f"{seed} randomized controlled trial",
                filters={"limit": config.default_discovery_page_size},
            )
        )
        state.quality.notes.append("react_replan_discovery")

    if not needs_repair:
        return state

    missing_trial_keys = [
        key
        for key, record in state.trial_records.items()
        if any(flag.startswith("missing_") for flag in record.quality_flags)
    ]
    if not missing_trial_keys:
        return state

    refetch_order = [tool for tool in ["ctgov_v2", "biomcp", "who_ictrp"] if tool in tool_registry]
    for trial_key in missing_trial_keys[:20]:
        for source in refetch_order:
            try:
                result = tool_registry[source].fetch(trial_key)
                if result.records:
                    state.raw_records[trial_key] = result.records[0]
                    break
            except ToolError:
                continue
            except Exception as exc:
                state.errors.append(f"repair_fetch:{source}:{trial_key}:{exc}")
    state.quality.notes.append("react_repair_refetch")
    return state


def finalize_run(state: GraphState) -> GraphState:
    sources_used = sorted(
        {
            record.provenance.get("source", "unknown")
            for record in state.trial_records.values()
            if isinstance(record.provenance, dict)
        }
    )
    state.memory["run_summary"] = {
        "candidates": len(state.candidate_set),
        "raw_records": len(state.raw_records),
        "trial_records": len(state.trial_records),
        "sources_used": sources_used,
        "errors": len(state.errors),
    }
    overbroad = state.memory.get("overbroad_warning") or state.memory.get("manual_overbroad_warning")
    if isinstance(overbroad, dict):
        state.memory["run_summary"]["overbroad_threshold"] = overbroad.get("threshold")
        state.memory["run_summary"]["overbroad_candidates"] = overbroad.get("candidates")
    return state


def write_jsonl(state: GraphState, config: RuntimeConfig) -> GraphState:
    write_trials_jsonl(config.output_jsonl, list(state.trial_records.values()))
    return state
