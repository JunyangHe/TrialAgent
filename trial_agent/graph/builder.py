from __future__ import annotations

from typing import Any

from trial_agent.config import RuntimeConfig, _dbg

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # type: ignore[misc]
        return iterable
from trial_agent.graph import nodes
from trial_agent.state import GraphState
from trial_agent.tools.adapters import build_tool_registry


def _load_tools(config: RuntimeConfig) -> dict[str, Any]:
    return {k: v for k, v in build_tool_registry().items() if k in config.enabled_tools}


def run_pipeline_rule(user_request: str, config: RuntimeConfig, langgraph: bool = False) -> GraphState | dict[str, Any]:
    """Run the deterministic pipeline (LLM only used in parse_request)."""
    if langgraph:
        app = build_langgraph_app(config)
        return app.invoke({"user_request": user_request})

    config.ensure_dirs()
    tools = _load_tools(config)
    state = GraphState(user_request=user_request)

    _dbg(f"enabled_tools={config.enabled_tools}, loaded tools={list(tools.keys())}")
    _dbg(f"user_request={user_request!r}")

    state = nodes.parse_request(state, config)
    if state.memory.get("abort"):
        state = nodes.finalize_run(state)
        return state

    state = nodes.plan_queries(state, config)

    _dbg(f"plan.query_queue len={len(state.plan.query_queue)}, tool_order={state.plan.tool_order}")

    for i in tqdm(range(config.max_discovery_attempts), desc="Discovery", unit="query"):
        if not state.plan.query_queue:
            _dbg(f"discovery loop exit at iter {i}: query_queue empty")
            break
        state = nodes.act_discover(state, tools, config)
        _dbg(f"after discover iter {i}: candidates={len(state.candidate_set)}, errors={len(state.errors)}")
        state = nodes.observe_discover(state, config)

    _dbg(f"before fetch: candidates={len(state.candidate_set)}")
    state = nodes.act_fetch_records(state, tools, config)
    _dbg(f"after fetch: raw_records={len(state.raw_records)}, errors={len(state.errors)}")
    state = nodes.normalize_records(state)
    _dbg(f"after normalize: trial_records={len(state.trial_records)}")
    state = nodes.validate_records(state)
    state = nodes.write_jsonl(state, config)
    state = nodes.finalize_run(state)
    _dbg(f"wrote {len(state.trial_records)} records, total errors={len(state.errors)}")
    return state


def run_pipeline_react(user_request: str, config: RuntimeConfig, langgraph: bool = False) -> GraphState | dict[str, Any]:
    """LLM ReAct runner: reason-plan, act, observe, validate, repair, finalize."""
    if langgraph:
        app = build_langgraph_app(config)
        return app.invoke({"user_request": user_request})

    config.ensure_dirs()
    tools = _load_tools(config)
    state = GraphState(user_request=user_request)

    _dbg(f"[react] enabled_tools={config.enabled_tools}, loaded tools={list(tools.keys())}")
    _dbg(f"[react] user_request={user_request!r}")

    state = nodes.parse_request(state, config)
    if state.memory.get("abort"):
        state = nodes.finalize_run(state)
        return state

    state = nodes.plan_queries_react(state, config)
    _dbg(f"[react] initial query_queue={len(state.plan.query_queue)}, tool_order={state.plan.tool_order}")

    discovery_budget = state.runtime.budgets_remaining.get("discovery_searches", config.max_discovery_attempts)
    for i in tqdm(range(min(config.max_discovery_attempts, discovery_budget)), desc="Discovery", unit="query"):
        if not state.plan.query_queue:
            state = nodes.observe_discover_react(state, config)
            if not state.plan.query_queue:
                _dbg(f"[react] discovery exit at iter {i}: queue empty after critic")
                break
        state = nodes.act_discover(state, tools, config)
        state = nodes.observe_discover_react(state, config)
        _dbg(f"[react] after discover iter {i}: candidates={len(state.candidate_set)}, queue={len(state.plan.query_queue)}")

    state = nodes.act_fetch_records_react(state, tools, config)
    state = nodes.normalize_records_react(state, config)
    state = nodes.validate_records_react(state, config)

    repair_loops = state.runtime.budgets_remaining.get("repair_loops", config.react_max_repair_loops)
    for i in range(repair_loops):
        before_raw = len(state.raw_records)
        state = nodes.repair_or_replan_react(state, tools, config)
        if len(state.raw_records) > before_raw:
            state = nodes.normalize_records_react(state, config)
            state = nodes.validate_records_react(state, config)
        _dbg(f"[react] repair iter {i}: raw={len(state.raw_records)}, trials={len(state.trial_records)}")

    state = nodes.write_jsonl(state, config)
    state = nodes.finalize_run(state)
    _dbg(f"[react] wrote {len(state.trial_records)} records, total errors={len(state.errors)}")
    return state


def build_langgraph_app(config: RuntimeConfig) -> Any:
    """Optional LangGraph construction entrypoint.

    IMPORTANT: This function requires `langgraph` installed.
    Hook your LLM in RuntimeConfig.llm_factory before production use.
    """
    try:
        from langgraph.graph import END, StateGraph
    except ImportError as exc:
        raise RuntimeError("langgraph is not installed. Use run_pipeline() or install langgraph.") from exc

    tools = _load_tools(config)
    graph = StateGraph(GraphState)

    if not config.use_llm_react:
        graph.add_node("parse_request", lambda s: nodes.parse_request(s, config))
        graph.add_node("plan", lambda s: nodes.plan_queries(s, config))
        graph.add_node("discover", lambda s: nodes.act_discover(s, tools, config))
        graph.add_node("observe", lambda s: nodes.observe_discover(s, config))
        graph.add_node("fetch", lambda s: nodes.act_fetch_records(s, tools, config))
        graph.add_node("normalize", nodes.normalize_records)
        graph.add_node("validate", nodes.validate_records)
        graph.add_node("write", lambda s: nodes.write_jsonl(s, config))
        graph.add_node("finalize", nodes.finalize_run)

        def route_after_parse(state: GraphState) -> str:
            return "finalize" if state.memory.get("abort") else "plan"

        graph.add_conditional_edges("parse_request", route_after_parse, {"finalize": "finalize", "plan": "plan"})
        graph.add_edge("plan", "discover")

        def should_continue_discovery(state: GraphState) -> str:
            if state.plan.query_queue and len(state.query_history) < config.max_discovery_attempts:
                return "discover"
            return "fetch"

        graph.add_edge("discover", "observe")
        graph.add_conditional_edges("observe", should_continue_discovery, {"discover": "discover", "fetch": "fetch"})
        graph.add_edge("fetch", "normalize")
        graph.add_edge("normalize", "validate")
        graph.add_edge("validate", "write")
        graph.add_edge("write", "finalize")
        graph.add_edge("finalize", END)
        graph.set_entry_point("parse_request")
        return graph.compile()

    graph.add_node("parse_request", lambda s: nodes.parse_request(s, config))
    graph.add_node("plan", lambda s: nodes.plan_queries_react(s, config))
    graph.add_node("discover", lambda s: nodes.act_discover(s, tools, config))
    graph.add_node("observe", lambda s: nodes.observe_discover_react(s, config))
    graph.add_node("fetch", lambda s: nodes.act_fetch_records_react(s, tools, config))
    graph.add_node("normalize", lambda s: nodes.normalize_records_react(s, config))
    graph.add_node("validate", lambda s: nodes.validate_records_react(s, config))

    def repair_step(state: GraphState) -> GraphState:
        count = int(state.memory.get("_react_repair_count", 0)) + 1
        state.memory["_react_repair_count"] = count
        before = len(state.raw_records)
        state = nodes.repair_or_replan_react(state, tools, config)
        state.memory["_react_repair_has_new_raw"] = len(state.raw_records) > before
        return state

    graph.add_node("repair", repair_step)
    graph.add_node("write", lambda s: nodes.write_jsonl(s, config))
    graph.add_node("finalize", nodes.finalize_run)

    def route_after_parse(state: GraphState) -> str:
        return "finalize" if state.memory.get("abort") else "plan"

    graph.add_conditional_edges("parse_request", route_after_parse, {"finalize": "finalize", "plan": "plan"})
    graph.add_edge("plan", "discover")

    def should_continue_discovery_react(state: GraphState) -> str:
        budget = state.runtime.budgets_remaining.get("discovery_searches", config.max_discovery_attempts)
        if state.plan.query_queue and len(state.query_history) < min(config.max_discovery_attempts, budget):
            return "discover"
        return "fetch"

    def should_continue_repair_react(state: GraphState) -> str:
        budget = state.runtime.budgets_remaining.get("repair_loops", config.react_max_repair_loops)
        count = int(state.memory.get("_react_repair_count", 0))
        if count >= budget:
            return "write"
        if bool(state.memory.get("_react_repair_has_new_raw")):
            return "normalize"
        return "repair"

    graph.add_edge("discover", "observe")
    graph.add_conditional_edges(
        "observe",
        should_continue_discovery_react,
        {"discover": "discover", "fetch": "fetch"},
    )
    graph.add_edge("fetch", "normalize")
    graph.add_edge("normalize", "validate")
    graph.add_edge("validate", "repair")
    graph.add_conditional_edges(
        "repair",
        should_continue_repair_react,
        {"normalize": "normalize", "repair": "repair", "write": "write"},
    )
    graph.add_edge("write", "finalize")
    graph.add_edge("finalize", END)
    graph.set_entry_point("parse_request")
    return graph.compile()


def run_pipeline(user_request: str, config: RuntimeConfig, langgraph: bool = False) -> GraphState | dict[str, Any]:
    
    # If user specified to use llm
    if config.use_llm_react:
        return run_pipeline_react(user_request, config, langgraph=langgraph)
    
    # If user did not specify to use llm, use rule mode
    return run_pipeline_rule(user_request, config, langgraph=langgraph)