from __future__ import annotations

import argparse

from trial_agent.config import RuntimeConfig
from trial_agent.graph.builder import run_pipeline


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y"}:
        return True
    if lowered in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TrialAgent prototype pipeline")
    parser.add_argument("request", help="Seed input (e.g., disease, drug, sponsor)")
    parser.add_argument(
        "--llm",
        type=_parse_bool,
        nargs="?",
        const=True,
        default=False,
        help="Enable full LLM ReAct pipeline (default: false). If false, only parser uses LLM.",
    )
    parser.add_argument(
        "--langgraph",
        action="store_true",
        help="Run via LangGraph (state graph with conditional edges) instead of regular loop",
    )
    args = parser.parse_args()

    config = RuntimeConfig()
    config.use_llm_react = args.llm

    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        ChatOpenAI = None  # type: ignore[assignment]

    if ChatOpenAI is not None:
        # Configure LLM model, currently hardcoded for OpenAI GPT-4o
        # config.llm_factory = lambda: ChatOpenAI(
        #     model="gpt-4o-mini",
        #     temperature=0,
        #     api_key=os.environ.get("OPENAI_API_KEY", "")
        # )
        config.llm_factory = lambda: ChatOpenAI(
            model="gpt-oss:20b",
            temperature=0,
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )

    state = run_pipeline(args.request, config, langgraph=args.langgraph)
    errs = state.errors if hasattr(state, "errors") else state.get("errors", [])
    if errs:
        for e in errs:
            print(f"Error: {e}")
    trial_count = len(state["trial_records"]) if isinstance(state, dict) else len(state.trial_records)

    print(f"Wrote {trial_count} records to {config.output_jsonl}")
    run_summary = (
        (state.get("memory", {}) if isinstance(state, dict) else state.memory).get("run_summary")
    )
    if run_summary:
        print(f"Run summary: {run_summary}")


if __name__ == "__main__":
    main()
