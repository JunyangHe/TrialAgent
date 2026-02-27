from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class RuntimeConfig:
    """Runtime configuration for the agent.

    IMPORTANT: wire your LLM client in `llm_factory` before running graph execution.
    """

    output_jsonl: Path = Path("artifacts/trials.jsonl")
    checkpoint_db: Path = Path("artifacts/langgraph_checkpoints.sqlite")
    cache_db: Path = Path("artifacts/trialagent_cache.sqlite")
    default_target_k: int = 25
    max_discovery_attempts: int = 8
    max_fetch_count: int = 50
    max_enrichment_count: int = 30
    llm_factory: Callable[[], Any] | None = None
    enabled_tools: set[str] = field(
        default_factory=lambda: {
            "biomcp",
            "ctgov_v2",
            "aact",
            "who_ictrp",
            "trialstreamer",
        }
    )

    def ensure_dirs(self) -> None:
        self.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
        self.cache_db.parent.mkdir(parents=True, exist_ok=True)
