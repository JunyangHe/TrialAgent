from __future__ import annotations

import os
from typing import Any

from trial_agent.models import ToolError
from trial_agent.tools.base import HTTPConfig, HTTPToolAdapter, ToolAdapter, ToolResult


class BioMCPAdapter(HTTPToolAdapter):
    """BioMCP adapter.

    Docs were unavailable in this environment, so this implementation uses configurable
    endpoint paths and robust response normalization.

    Environment variables:
      - BIOMCP_BASE_URL (default: https://biomcp.org/api)
      - BIOMCP_API_KEY (optional)
      - BIOMCP_DISCOVER_PATH (default: /trials/search)
      - BIOMCP_FETCH_PATH_TEMPLATE (default: /trials/{trial_id})
    """

    name = "biomcp"

    def __init__(self) -> None:
        super().__init__(
            HTTPConfig(
                base_url=os.getenv("BIOMCP_BASE_URL", "https://biomcp.org/api"),
                api_key_env="BIOMCP_API_KEY",
            )
        )
        self.discover_path = os.getenv("BIOMCP_DISCOVER_PATH", "/trials/search")
        self.fetch_path_template = os.getenv("BIOMCP_FETCH_PATH_TEMPLATE", "/trials/{trial_id}")

    def discover(self, query: str, filters: dict[str, Any]) -> ToolResult:
        payload = {"query": query, **filters}
        response = self._request_json("POST", self.discover_path, body=payload)
        records = self._extract_records(response)
        return ToolResult(
            records=[self._normalize_record(row) for row in records],
            metadata={"query": query, "filters": filters, "raw_keys": sorted(response.keys())},
        )

    def fetch(self, trial_id: str) -> ToolResult:
        path = self.fetch_path_template.format(trial_id=trial_id)
        response = self._request_json("GET", path)
        row = self._extract_single_record(response)
        if row is None:
            return ToolResult(records=[], metadata={"trial_id": trial_id, "raw_keys": sorted(response.keys())})
        return ToolResult(
            records=[self._normalize_record(row)],
            metadata={"trial_id": trial_id, "raw_keys": sorted(response.keys())},
        )

    @staticmethod
    def _extract_records(response: dict[str, Any]) -> list[dict[str, Any]]:
        for key in ("records", "trials", "results", "data", "items"):
            value = response.get(key)
            if isinstance(value, list):
                return [v for v in value if isinstance(v, dict)]
            if isinstance(value, dict):
                nested = value.get("records") or value.get("results")
                if isinstance(nested, list):
                    return [v for v in nested if isinstance(v, dict)]
        return []

    @classmethod
    def _extract_single_record(cls, response: dict[str, Any]) -> dict[str, Any] | None:
        direct = response.get("record") or response.get("trial")
        if isinstance(direct, dict):
            return direct
        records = cls._extract_records(response)
        return records[0] if records else None

    @staticmethod
    def _normalize_record(raw: dict[str, Any]) -> dict[str, Any]:
        nct_id = raw.get("nct_id") or raw.get("nctId") or raw.get("nctNumber")
        trial_key = raw.get("trial_key") or raw.get("id") or raw.get("registry_id") or nct_id
        conditions = raw.get("conditions") or raw.get("condition") or []
        interventions = raw.get("interventions") or raw.get("intervention") or []

        if isinstance(conditions, str):
            conditions = [conditions]
        if isinstance(interventions, str):
            interventions = [interventions]

        return {
            "trial_key": trial_key,
            "id": raw.get("id", trial_key),
            "nct_id": nct_id,
            "title": raw.get("title") or raw.get("brief_title") or raw.get("briefTitle") or "",
            "conditions": conditions,
            "interventions": interventions,
            "sponsor": raw.get("sponsor") or raw.get("lead_sponsor") or raw.get("leadSponsor"),
            "status": raw.get("status") or raw.get("overall_status") or raw.get("overallStatus"),
            "phase": raw.get("phase"),
            "study_type": raw.get("study_type") or raw.get("studyType"),
            "start_date": raw.get("start_date") or raw.get("startDate"),
            "primary_completion_date": raw.get("primary_completion_date") or raw.get("primaryCompletionDate"),
            "summary": raw.get("summary") or raw.get("brief_summary") or raw.get("briefSummary"),
            "outcomes": raw.get("outcomes") or {},
            "source": "biomcp",
            "identifiers": {
                "primary": trial_key,
                **({"nct": nct_id} if nct_id else {}),
                **({"registry": raw["registry_id"]} if raw.get("registry_id") else {}),
            },
            "_raw": raw,
        }


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
