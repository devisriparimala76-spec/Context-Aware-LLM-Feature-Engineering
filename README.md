# CAFE — A Context-Aware LLM Framework for Automated Feature Engineering in Tabular Machine Learning

CAFE implements the full propose → execute → evaluate → feedback loop described
in the project brief: an LLM reads a dataset's *description* (not just its
column names), proposes one new engineered feature at a time as executable
pandas code, the code is run in a sandbox, a downstream classifier is
retrained with stratified cross-validation, and the resulting ROC-AUC delta
is fed back into the LLM's next prompt — automatically, with zero manual
feature engineering.

## Why this design

This project is a from-scratch reimplementation of the architecture behind
CAAFE (Hollmann et al., 2023) and its "context-aware automated feature
engineering" line of research, built to run standalone, be auditable end to
end, and work identically whether the "LLM" is a real model or an offline
stand-in (see [`MockProvider`](src/llm_client.py) below) — useful for
demoing, testing, and CI in network-restricted environments.

## Architecture

```
                 ┌─────────────────────────┐
                 │   DatasetContext         │   <- context_builder.py
                 │ (description, dtypes,    │      "Context-Aware": the LLM sees
                 │  stats, sample rows,     │      *meaning*, not just column names
                 │  feature history)        │
                 └────────────┬─────────────┘
                              │ rendered prompt
                              ▼
                 ┌─────────────────────────┐
                 │      LLMProvider         │   <- llm_client.py
                 │ (OpenAI / Anthropic /    │      pluggable: swap providers via
                 │  Gemini / Mock)          │      env var, same JSON contract
                 └────────────┬─────────────┘
                              │ {feature_name, code, explanation}
                              ▼
                 ┌─────────────────────────┐
                 │   execute_feature_code   │   <- code_executor.py
                 │ (sandboxed exec, no      │      restricted builtins, no I/O,
                 │  filesystem/network)     │      runs on a dataframe COPY
                 └────────────┬─────────────┘
                     success  │  failure -> fed back as a rejected candidate
                              ▼
                 ┌─────────────────────────┐
                 │     evaluate_roc_auc     │   <- evaluator.py
                 │ (RandomForest, 5-fold    │      the objective referee:
                 │  stratified CV)          │      keep only if ΔAUC > min_delta
                 └────────────┬─────────────┘
                              │ score + delta
                              ▼
                 ┌─────────────────────────┐
                 │        CAFEPipeline      │   <- pipeline.py
                 │  loops, logs every       │      full run_report.json / .md,
                 │  iteration, stops on     │      final feature-engineered
                 │  patience or max_iters   │      dataset saved to disk
                 └─────────────────────────┘
```

## Quickstart

```bash
pip install -r requirements.txt

# Offline demo (no API key needed — uses MockProvider automatically):
python examples/run_demo.py

# With a real LLM:
export ANTHROPIC_API_KEY=sk-ant-...
export CAFE_LLM_PROVIDER=anthropic     # or: openai / gemini
python examples/run_demo.py
```

Output lands in `outputs/`:
- `run_report.md` — human-readable iteration log (what was proposed, kept, why)
- `run_report.json` — same data, machine-readable
- `final_dataset.csv` — the original dataset plus every accepted feature

## Using it on your own dataset

```python
import pandas as pd
from src.pipeline import CAFEPipeline, save_report

df = pd.read_csv("your_data.csv")

pipeline = CAFEPipeline(
    df=df,
    target_column="your_target",
    description="One or two sentences: what does this dataset represent, "
                 "what is being predicted, what domain is it from?",
)
report = pipeline.run()
save_report(report, "outputs/")
```

The `description` field is what makes the framework *context-aware* — a
generic AutoFE tool (e.g. Featuretools) only sees numbers and column names;
this framework is explicitly told what the data *means*, so it can reason
about domain-appropriate transformations (BMI for health data, utilization
ratio for credit data, heat index for weather data, etc.) instead of blindly
generating every possible arithmetic combination.

## Honest scope note on the reference demo

The committed `outputs/run_report.md` in this repo was generated with
`MockProvider` — a deterministic, keyword-driven, **offline** stand-in for a
real LLM (see the big comment block at the top of `src/llm_client.py`). It
exists so the full pipeline is provably runnable and testable without any
API key or network access. It is **not** a benchmark of GPT-4/Claude/Gemini
feature-engineering quality. The `LLMProvider` interface is a genuine,
tested integration point for real models (`OpenAIProvider`,
`AnthropicProvider`, `GeminiProvider`) — swap one in with an API key and the
rest of the pipeline (execution, evaluation, logging) is unchanged. The
accompanying paper (`paper/`) is written and scoped accordingly: as a system/
architecture contribution with a reproducible proof-of-concept run, not as a
large-scale empirical LLM benchmark.

## Project layout

```
context_aware_fe/
├── src/
│   ├── context_builder.py   # dataset -> LLM-readable context ("context-aware")
│   ├── llm_client.py        # pluggable LLM providers + offline MockProvider
│   ├── code_executor.py     # sandboxed execution of LLM-generated code
│   ├── evaluator.py         # cross-validated ROC-AUC (the accept/reject referee)
│   └── pipeline.py          # the propose->execute->evaluate->feedback loop
├── data/
│   └── make_sample_dataset.py  # synthetic offline demo dataset generator
├── examples/
│   └── run_demo.py          # end-to-end runnable demo
├── dashboard/
│   └── app.py                # Phase 2: Streamlit dashboard (upload your own dataset)
├── tests/
│   ├── test_pipeline.py             # pytest suite
│   └── run_tests_standalone.py      # no-pytest-required runner
├── outputs/                 # generated run reports (produced by run_demo.py)
├── paper/                   # IEEE-style paper describing this project
└── requirements.txt
```

## Phase 2 — Interactive Dashboard

`dashboard/app.py` is a Streamlit dashboard built on top of the exact same pipeline —
no separate logic, no duplicated code. It lets you:

- Upload your own dataset (CSV or Excel) instead of editing Python
- Pick the target column and describe the dataset from dropdowns/text boxes
- Choose a provider (offline Mock, or OpenAI/Anthropic/Gemini with your own API key,
  entered as a password field — never written to disk)
- Watch the propose → execute → evaluate → feedback loop run **live**, iteration by
  iteration, with the generated code, the accept/reject decision, and a running
  ROC-AUC chart
- Download the final feature-engineered dataset and the run report when it's done

### Run it

```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```

This opens a browser tab (usually `http://localhost:8501`). Upload a dataset, fill in
the target column and description in the sidebar, and click **Run CAFE**.

### How it connects to the rest of the project

The dashboard doesn't reimplement anything — it imports `CAFEPipeline` directly from
`src/pipeline.py` and passes it a `progress_callback` function. That callback is a
small, tested contract (`src/pipeline.py::CAFEPipeline.run`, covered by
`test_progress_callback_fires_expected_event_sequence` in both test suites) that fires
four event types — `baseline`, `iteration`, `stopped`, `done` — which the dashboard
uses to update its live log and chart. Anything else built on top of CAFE (a CLI, a
different UI, a notebook) can hook into the same callback the same way.

## Running tests

```bash
pytest tests/ -v
# or, without pytest installed:
python tests/run_tests_standalone.py
```

## Limitations (stated plainly, not hidden)

- `MockProvider` is a template-matching heuristic, not a language model — it
  demonstrates the *pipeline*, not LLM reasoning quality.
- The code sandbox restricts builtins but is not a hardened security
  boundary; do not run untrusted LLM output against production data stores.
- Feature acceptance uses a single train/CV split family (stratified k-fold);
  a held-out test set is recommended before deploying any accepted feature
  set in production.
- Non-numeric proposed features are ordinal-encoded for modeling, which is a
  simplification versus target encoding or one-hot for high-cardinality cases.
