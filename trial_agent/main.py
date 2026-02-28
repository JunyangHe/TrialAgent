from __future__ import annotations

import argparse

from trial_agent.config import RuntimeConfig
from trial_agent.graph.builder import build_langgraph_app, run_pipeline
from langchain_openai import ChatOpenAI


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TrialAgent prototype pipeline")
    parser.add_argument("request", help="Seed input (e.g., disease, drug, sponsor)")
    parser.add_argument(
        "--langgraph",
        action="store_true",
        help="Run via LangGraph (state graph with conditional edges) instead of regular loop",
    )
    args = parser.parse_args()

    config = RuntimeConfig()
    config.llm_factory = lambda: ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key=os.environ.get("OPENAI_API_KEY", "")
    )

    if args.langgraph:
        graph = build_langgraph_app(config)
        state = graph.invoke({"user_request": args.request})
        trial_count = len(state["trial_records"]) if isinstance(state, dict) else len(state.trial_records)
    else:
        state = run_pipeline(args.request, config)
        trial_count = len(state.trial_records)

    print(f"Wrote {trial_count} records to {config.output_jsonl}")


if __name__ == "__main__":
    main()
