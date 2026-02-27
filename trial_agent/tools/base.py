from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ToolResult:
    records: list[dict[str, Any]]
    metadata: dict[str, Any]


class ToolAdapter:
    name: str = "tool"

    def discover(self, query: str, filters: dict[str, Any]) -> ToolResult:
        raise NotImplementedError

    def fetch(self, trial_id: str) -> ToolResult:
        raise NotImplementedError

    def enrich(self, trial_record: dict[str, Any]) -> ToolResult:
        return ToolResult(records=[], metadata={"note": "enrichment not supported"})
