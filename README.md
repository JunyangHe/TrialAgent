# TrialAgent

Clinical trial search and normalization agent with two execution modes:

- Rule-based pipeline (default): deterministic orchestration, LLM used only for request parsing.
- LLM ReAct pipeline (`--llm true`): LLM-guided planning, critic, normalization/QC, and repair loops.

The pipeline writes normalized trial rows to `artifacts/trials.jsonl`.

## Implemented functionality

- Graph state + node pipeline for:
  - parse request
  - plan queries
  - discover candidates
  - observe/critic loop
  - fetch full records
  - normalize records
  - validate quality
  - repair/replan (ReAct mode)
  - finalize + JSONL export
- Tool adapters currently wired and used:
  - BioMCP (`biomcp`)
  - ClinicalTrials.gov API v2 (`ctgov_v2`)
  - WHO ICTRP (`who_ictrp`)
- Deterministic fallback hooks:
  1. low-yield
  2. overbroad

## Installation

```bash
pip install -r requirements.txt
```

## Ollama setup (local LLM)

The CLI can use a local LLM via [Ollama](https://ollama.com) for parsing and ReAct. The default configuration in `trial_agent/main.py` points to `gpt-oss:20b` on `http://localhost:11434/v1`.

1. **Install Ollama**:

   - macOS (curl):
     ```bash
     curl -fsSL https://ollama.com/install.sh | sh
     ```
   - Windows (PowerShell):
     ```powershell
     irm https://ollama.com/install.ps1 | iex
     ```

   Then start the server:
   ```bash
   ollama serve
   ```

2. **Pull the model** (20B for typical machines; 120B if you have ≥60GB VRAM):

   ```bash
   ollama pull gpt-oss:20b
   ```

3. **Verify** Ollama is running and the model works:

   ```bash
   ollama run gpt-oss:20b "Hello"
   ```

   Or test the API:

   ```bash
   curl http://localhost:11434/v1/models
   ```

TrialAgent will call this local model for argument parser and when using `--llm true`.

## Usage

Rule mode (default):

```bash
python -m trial_agent.main "EGFR mutant non-small cell lung cancer"
```

ReAct mode:

```bash
python -m trial_agent.main "EGFR mutant non-small cell lung cancer" --llm true
```

LangGraph mode (rule pipeline only):

```bash
python -m trial_agent.main "EGFR mutant non-small cell lung cancer" --langgraph
```

### Options

| Option | Description |
|--------|-------------|
| `--target K` | Target number of trials to retrieve (default: 100000). Use a smaller value (e.g. `--target 500`) for quicker runs. |

## Troubleshooting

### SSL certificate error when running ctgov_v2

If you see an error like `[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: self signed certificate in certificate chain` when the `ctgov_v2` adapter fetches from ClinicalTrials.gov (e.g. during `act_discover` or fetch), Python’s CA store is likely incomplete. This can happen on macOS with python.org-installed Python.

**Fix:**

1. Install `certifi` (if pip also fails with SSL, use `--trusted-host` to bypass verification):

   ```bash
   pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org certifi
   ```

2. Point Python to certifi’s CA bundle before running the agent:

   ```bash
   export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")
   export REQUESTS_CA_BUNDLE=$SSL_CERT_FILE
   ```

   Or add these lines to your shell profile (`~/.zshrc` or `~/.bashrc`) so they apply to every session.

Then run the agent again in the same shell.

## Configuration

Main runtime knobs live in `trial_agent/config.py` (`RuntimeConfig`):

- `default_target_k` — target number of trials (overridable via `--target K` on the CLI)
- `enabled_tools`
- `tool_order` / `fetch_tool_order`
- `default_discovery_page_size`
- `max_discovery_attempts`
- `max_pagination_pages`
- `max_fetch_count`
- `use_llm_react`, `react_max_repair_loops`, `react_normalize_batch_size`
- `llm_factory`

The CLI currently wires `ChatOpenAI` in `trial_agent/main.py` via `config.llm_factory`.

## Project structure

```text
trial_agent/
├── __init__.py
├── main.py              # CLI entrypoint
├── config.py            # Runtime configuration
├── state.py             # GraphState and GraphRuntime
├── models.py            # SearchSpec, Plan, TrialRecord, etc.
├── graph/
│   ├── __init__.py
│   ├── builder.py       # Pipeline and LangGraph construction
│   └── nodes.py         # All graph node logic (parse, plan, act, observe, etc.)
├── tools/
│   ├── __init__.py
│   ├── base.py          # ToolAdapter, HTTPToolAdapter, ToolResult
│   └── adapters.py      # ctgov_v2, biomcp, who_ictrp adapters
├── policies/
│   ├── __init__.py
│   └── fallbacks.py     # Low-yield, overbroad, ambiguity, missing-fields fallbacks
└── io/
    ├── __init__.py
    └── jsonl_writer.py  # Write TrialRecords to JSONL
```

## Output

Trial records are written to `artifacts/trials.jsonl` (one JSON object per line, JSONL format). A run summary is stored in memory and printed by the CLI (`candidates`, `raw_records`, `trial_records`, `sources_used`, `errors`).

### JSON object structure (per line)

Each line is a single JSON object with the following fields. All fields may be `null` or empty if not available from the source.

| Field | Type | Description |
|-------|------|-------------|
| `trial_key` | string | Primary identifier (e.g. NCT number like `NCT04545710`) |
| `identifiers` | object | Mappings such as `primary`, `nct`, `registry` |
| `condition` | array of strings | Medical conditions studied |
| `interventions` | array of strings | Drugs, procedures, or other interventions |
| `molecular_targets` | array of strings | Biomolecular targets (if extracted) |
| `sponsor` | string \| null | Lead sponsor or organization |
| `clinical_status` | string \| null | Recruitment status (e.g. `RECRUITING`, `ACTIVE_NOT_RECRUITING`, `COMPLETED`, `TERMINATED`) |
| `phase` | string \| null | Study phase (e.g. `PHASE1`, `PHASE2`, `NA` for observational) |
| `study_type` | string \| null | `INTERVENTIONAL` or `OBSERVATIONAL` |
| `start_date` | string \| null | Trial start date |
| `primary_completion_date` | string \| null | Primary completion date |
| `outcomes` | object | `primaryOutcomes` and `secondaryOutcomes` arrays with `measure`, `description`, `timeFrame` |
| `summary` | string \| null | Brief or detailed study description |
| `locations` | array of strings | Study sites (e.g. facility, city, state, country) |
| `evidence_snippets` | array | Placeholder for extracted evidence (usually empty) |
| `provenance` | object | `source` (e.g. `ctgov_v2`, `biomcp`, `who_ictrp`) and optional metadata |
| `quality_flags` | array of strings | Validation notes (e.g. `missing_interventions`) |
