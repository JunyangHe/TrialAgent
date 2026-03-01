from __future__ import annotations

from trial_agent.models import Plan
from trial_agent.state import GraphState


def apply_low_yield_fallback(state: GraphState) -> None:
    state.quality.notes.append("fallback_1_low_yield_applied")


def apply_overbroad_fallback(state: GraphState) -> None:
    state.quality.notes.append("fallback_2_overbroad_applied")
    if state.search_spec and "study_type" not in state.search_spec.must_have_fields:
        state.search_spec.must_have_fields.append("study_type")


def build_default_fallback_policy() -> dict[str, str]:
    return {
        "low_yield": "fallback_1",
        "overbroad": "fallback_2",
    }


def seed_plan_defaults(plan: Plan) -> Plan:
    if not plan.stop_conditions:
        plan.stop_conditions = {"min_high_confidence_trials": 25, "or_budget_exhausted": True}
    if not plan.fallback_policy:
        plan.fallback_policy = build_default_fallback_policy()
    return plan
