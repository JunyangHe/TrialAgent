import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEBUG = os.environ.get("TRIAL_AGENT_DEBUG", "1").lower() in ("1", "true", "yes")


def _dbg(msg: str) -> None:
    if DEBUG:
        print(f"[TrialAgent] {msg}")


@dataclass(slots=True)
class RuntimeConfig:
    """Runtime configuration for the agent.

    IMPORTANT: wire your LLM client in `llm_factory` before running graph execution.
    """

    output_jsonl: Path = Path("artifacts/trials.jsonl")
    checkpoint_db: Path = Path("artifacts/langgraph_checkpoints.sqlite")
    cache_db: Path = Path("artifacts/trialagent_cache.sqlite")
    default_target_k: int = 25
    default_discovery_page_size: int = 250  # max results per tool per discover call (was 25)
    max_discovery_attempts: int = 20  # raised to allow pagination (was 8)
    max_pagination_pages: int = 5  # max additional pages per tool when nextPageToken returned
    max_fetch_count: int = 250  # max candidates to fetch full records for (was 50; limits final trial count)
    max_enrichment_count: int = 30
    llm_factory: Callable[[], Any] | None = None
    tool_order: list[str] | None = None  # override Plan default; None = use ["ctgov_v2", "biomcp", "who_ictrp"]
    fetch_tool_order: list[str] = field(
        default_factory=lambda: ["ctgov_v2", "biomcp"]  # prefer ClinicalTrials.gov for fetch, then biomcp
    )
    enabled_tools: set[str] = field(
        default_factory=lambda: {
            "biomcp",
            "ctgov_v2",
            "who_ictrp",
        }
    )

    def ensure_dirs(self) -> None:
        self.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
        self.cache_db.parent.mkdir(parents=True, exist_ok=True)
