# TrialAgent

Prototype framework for an AI-assisted clinical trial extraction agent using a ReAct-style loop and a LangGraph-first architecture.

## What is implemented

- Modular graph nodes for: parse, plan, discover, observe, fetch, normalize, enrich, validate, and JSONL export.
- Five tool adapters exposed in the framework (stubs ready for API wiring):
  - BioMCP
  - ClinicalTrials.gov API v2
  - AACT
  - WHO ICTRP
  - Trialstreamer
- Graph state model with memory, query history, candidate set, raw records, normalized records, quality report, runtime budgets, and error log.
- Fallback policy hooks implemented as deterministic replan/repair behaviors:
  1. low-yield fallback,
  2. overbroad fallback,
  3. ambiguity fallback,
  4. missing-fields fallback,
  5. no-evidence fallback.
- JSONL output writer for canonical `TrialRecord` artifacts (`artifacts/trials.jsonl`).

## Where to configure the LLM

The project intentionally does **not** configure an LLM client yet.

1. Set `RuntimeConfig.llm_factory` in your entrypoint (e.g. `trial_agent/main.py`).
2. Replace TODO sections in:
   - `trial_agent/graph/nodes.py::parse_request`
   - `trial_agent/graph/nodes.py::plan_queries`
   - optionally `trial_agent/graph/nodes.py::observe_discover`

These are the planner/critic/normalizer roles intended for ReAct behavior.

## Run the prototype pipeline

```bash
python -m trial_agent.main "EGFR mutant non-small cell lung cancer"
```

This runs the deterministic skeleton pipeline and writes `artifacts/trials.jsonl`.

## LangGraph mode

If you want an actual LangGraph app object:

- install optional dependency: `pip install .[graph]`
- use `trial_agent.graph.build_langgraph_app(config)`

If LangGraph is unavailable, `run_pipeline(...)` still works as a local deterministic fallback.

## Notes

- Tool adapters currently return empty payloads; wire each adapter to real endpoints next.
- `Trialstreamer` is enrichment-only by design in this architecture.
- Missing fields are preserved as null/empty and surfaced via `quality_flags`.
