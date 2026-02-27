from __future__ import annotations

import argparse

from trial_agent.config import RuntimeConfig
from trial_agent.graph.builder import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TrialAgent prototype pipeline")
    parser.add_argument("request", help="Seed input (e.g., disease, drug, sponsor)")
    args = parser.parse_args()

    config = RuntimeConfig()
    # IMPORTANT: configure config.llm_factory when you want LLM-backed Parse/Plan/Critic behavior.
    state = run_pipeline(args.request, config)
    print(f"Wrote {len(state.trial_records)} records to {config.output_jsonl}")


if __name__ == "__main__":
    main()
