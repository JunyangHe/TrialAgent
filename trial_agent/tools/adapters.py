from __future__ import annotations

from typing import Any

from trial_agent.models import ToolError
from trial_agent.tools.base import ToolAdapter, ToolResult


class BioMCPAdapter(ToolAdapter):
    name = "biomcp"

    def discover(self, query: str, filters: dict[str, Any]) -> ToolResult:
        # TODO: implement API call + auth handling.
        return ToolResult(records=[], metadata={"query": query, "filters": filters})

    def fetch(self, trial_id: str) -> ToolResult:
        # TODO: implement record retrieval.
        return ToolResult(records=[], metadata={"trial_id": trial_id})


class ClinicalTrialsGovV2Adapter(ToolAdapter):
    name = "ctgov_v2"

    def discover(self, query: str, filters: dict[str, Any]) -> ToolResult:
        return ToolResult(records=[], metadata={"query": query, "filters": filters})

    def fetch(self, trial_id: str) -> ToolResult:
        return ToolResult(records=[], metadata={"trial_id": trial_id})


class AACTAdapter(ToolAdapter):
    name = "aact"

    def discover(self, query: str, filters: dict[str, Any]) -> ToolResult:
        return ToolResult(records=[], metadata={"query": query, "filters": filters})

    def fetch(self, trial_id: str) -> ToolResult:
        return ToolResult(records=[], metadata={"trial_id": trial_id})


class WHOICTRPAdapter(ToolAdapter):
    name = "who_ictrp"

    def discover(self, query: str, filters: dict[str, Any]) -> ToolResult:
        return ToolResult(records=[], metadata={"query": query, "filters": filters})

    def fetch(self, trial_id: str) -> ToolResult:
        raise ToolError("WHO ICTRP fetch-by-id not configured in prototype")


class TrialstreamerAdapter(ToolAdapter):
    name = "trialstreamer"

    def discover(self, query: str, filters: dict[str, Any]) -> ToolResult:
        raise ToolError("Trialstreamer is enrichment-only in this architecture")

    def fetch(self, trial_id: str) -> ToolResult:
        raise ToolError("Trialstreamer does not provide registry-first trial fetch")

    def enrich(self, trial_record: dict[str, Any]) -> ToolResult:
        return ToolResult(records=[], metadata={"trial_key": trial_record.get("trial_key")})


def build_tool_registry() -> dict[str, ToolAdapter]:
    return {
        "biomcp": BioMCPAdapter(),
        "ctgov_v2": ClinicalTrialsGovV2Adapter(),
        "aact": AACTAdapter(),
        "who_ictrp": WHOICTRPAdapter(),
        "trialstreamer": TrialstreamerAdapter(),
    }
