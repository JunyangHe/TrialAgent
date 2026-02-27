from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from trial_agent.models import TrialRecord


def write_trials_jsonl(path: Path, records: list[TrialRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
