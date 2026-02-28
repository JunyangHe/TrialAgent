from __future__ import annotations

from collections.abc import Callable
from typing import Any

from trial_agent.config import RuntimeConfig, _dbg
from trial_agent.graph import nodes
from trial_agent.state import GraphState
from trial_agent.tools.adapters import build_tool_registry


def run_pipeline(user_request: str, config: RuntimeConfig) -> GraphState:
    """Deterministic local runner for the prototype graph."""
    config.ensure_dirs()
    tools = {k: v for k, v in build_tool_registry().items() if k in config.enabled_tools}
    state = GraphState(user_request=user_request)

    _dbg(f"enabled_tools={config.enabled_tools}, loaded tools={list(tools.keys())}")
    _dbg(f"user_request={user_request!r}")

    state = nodes.parse_request(state, config)
    state = nodes.plan_queries(state, config)

    _dbg(f"plan.query_queue len={len(state.plan.query_queue)}, tool_order={state.plan.tool_order}")

    for i in range(config.max_discovery_attempts):
        if not state.plan.query_queue:
            _dbg(f"discovery loop exit at iter {i}: query_queue empty")
            break
        state = nodes.act_discover(state, tools, config)
        _dbg(f"after discover iter {i}: candidates={len(state.candidate_set)}, errors={state.errors}")
        state = nodes.observe_discover(state, config)

    _dbg(f"before fetch: candidates={len(state.candidate_set)}")
    state = nodes.act_fetch_records(state, tools, config)
    _dbg(f"after fetch: raw_records={len(state.raw_records)}, errors={state.errors}")
    state = nodes.normalize_records(state)
    _dbg(f"after normalize: trial_records={len(state.trial_records)}")
    state = nodes.validate_records(state)
    state = nodes.write_jsonl(state, config)
    _dbg(f"wrote {len(state.trial_records)} records, total errors={len(state.errors)}")
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

    tools = {k: v for k, v in build_tool_registry().items() if k in config.enabled_tools}
    graph = StateGraph(GraphState)

    graph.add_node("parse_request", lambda s: nodes.parse_request(s, config))
    graph.add_node("plan", lambda s: nodes.plan_queries(s, config))
    graph.add_node("discover", lambda s: nodes.act_discover(s, tools, config))
    graph.add_node("observe", lambda s: nodes.observe_discover(s, config))
    graph.add_node("fetch", lambda s: nodes.act_fetch_records(s, tools, config))
    graph.add_node("normalize", nodes.normalize_records)
    graph.add_node("validate", nodes.validate_records)
    graph.add_node("write", lambda s: nodes.write_jsonl(s, config))

    graph.add_edge("parse_request", "plan")
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
    graph.add_edge("write", END)

    graph.set_entry_point("parse_request")
    return graph.compile()
