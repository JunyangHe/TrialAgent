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
    """Runtime configuration for pipeline execution."""

    output_jsonl: Path = Path("artifacts/trials.jsonl")
    checkpoint_db: Path = Path("artifacts/langgraph_checkpoints.sqlite")
    cache_db: Path = Path("artifacts/trialagent_cache.sqlite")
    default_target_k: int = 1_000_000  # default to return as many trials as found
    default_discovery_page_size: int = 1_000_000
    max_discovery_attempts: int = 1_000_000
    max_pagination_pages: int = 100
    max_fetch_count: int = 1_000_000  # fetch all candidates
    use_llm_react: bool = False
    react_max_repair_loops: int = 2
    react_normalize_batch_size: int = 30
    llm_factory: Callable[[], Any] | None = None
    tool_order: list[str] | None = None
    fetch_tool_order: list[str] = field(default_factory=lambda: ["ctgov_v2", "biomcp"])
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
