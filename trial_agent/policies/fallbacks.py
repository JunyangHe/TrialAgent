from __future__ import annotations

from trial_agent.models import Plan
from trial_agent.state import GraphState


def apply_low_yield_fallback(state: GraphState) -> None:
    state.quality.notes.append("fallback_1_low_yield_applied")
    state.plan.tool_order = ["ctgov_v2", "who_ictrp", "biomcp"]


def apply_overbroad_fallback(state: GraphState) -> None:
    state.quality.notes.append("fallback_2_overbroad_applied")
    if state.search_spec and "study_type" not in state.search_spec.must_have_fields:
        state.search_spec.must_have_fields.append("study_type")


def apply_ambiguity_fallback(state: GraphState) -> None:
    state.quality.notes.append("fallback_3_ambiguity_applied")
    variants = state.memory.setdefault("query_variants", [])
    base = state.search_spec.raw_request if state.search_spec else state.user_request
    variants.extend([f"{base} clinical trial", f"{base} phase 2", f"{base} sponsor study"])


def apply_missing_fields_fallback(state: GraphState) -> None:
    state.quality.notes.append("fallback_4_missing_fields_refetch_applied")


def build_default_fallback_policy() -> dict[str, str]:
    return {
        "low_yield": "fallback_1",
        "overbroad": "fallback_2",
        "ambiguity": "fallback_3",
        "missing_fields": "fallback_4",
    }


def seed_plan_defaults(plan: Plan) -> Plan:
    if not plan.stop_conditions:
        plan.stop_conditions = {"min_high_confidence_trials": 25, "or_budget_exhausted": True}
    if not plan.fallback_policy:
        plan.fallback_policy = build_default_fallback_policy()
    return plan
