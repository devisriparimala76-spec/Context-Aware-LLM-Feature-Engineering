"""
dashboard/app.py
-----------------
Phase 2: an interactive dashboard around the CAFE pipeline.

The person uploads their own tabular dataset (CSV or Excel), picks the
target column, describes the dataset in a sentence or two, chooses an
LLM provider (or the offline Mock provider — no API key needed), and
watches the propose -> execute -> evaluate -> feedback loop run live,
iteration by iteration, with a running ROC-AUC chart and a downloadable
final dataset + run report at the end.

Run it with:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make the project's src/ package importable regardless of where streamlit is launched from
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import CAFEPipeline, RunReport  # noqa: E402
from src.llm_client import (  # noqa: E402
    MockProvider, OpenAIProvider, AnthropicProvider, GeminiProvider,
)

st.set_page_config(page_title="CAFE — Context-Aware Feature Engineering", page_icon="🧪", layout="wide")

# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
if "report" not in st.session_state:
    st.session_state.report = None
if "final_df" not in st.session_state:
    st.session_state.final_df = None
if "auc_history" not in st.session_state:
    st.session_state.auc_history = []  # list of {"label": str, "auc": float}

# --------------------------------------------------------------------------- #
# Sidebar — data upload + configuration
# --------------------------------------------------------------------------- #
st.sidebar.title("🧪 CAFE Dashboard")
st.sidebar.caption("Context-Aware LLM Feature Engineering — Phase 2")

st.sidebar.header("1. Upload your dataset")
uploaded_file = st.sidebar.file_uploader(
    "CSV or Excel file", type=["csv", "xlsx", "xls"],
    help="Your data stays in this session only — nothing is uploaded anywhere else.",
)

df = None
if uploaded_file is not None:
    try:
        if uploaded_file.name.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
        if df is not None and df.shape[0] == 0:
            st.sidebar.error("The uploaded file has no rows.")
            df = None
    except Exception as e:
        st.sidebar.error(f"Could not read file: {e}")
        df = None

st.sidebar.header("2. Choose an LLM provider")
provider_choice = st.sidebar.selectbox(
    "Provider",
    ["Offline (Mock — no API key needed)", "OpenAI", "Anthropic (Claude)", "Google (Gemini)"],
    help="The offline provider is deterministic and free — great for trying the dashboard out. "
         "Pick a real provider once you have an API key for genuine LLM reasoning.",
)
api_key = None
model_override = None
if provider_choice != "Offline (Mock — no API key needed)":
    api_key = st.sidebar.text_input(
        f"{provider_choice} API key", type="password",
        help="Used only for this session; never saved to disk.",
    )
    _default_models = {
        "OpenAI": "gpt-4o",
        "Anthropic (Claude)": "claude-sonnet-5",
        "Google (Gemini)": "gemini-flash-latest",
    }
    model_override = st.sidebar.text_input(
        "Model name",
        value=_default_models.get(provider_choice, ""),
        help="If you get a 404 / 'model not found' error, the provider has likely renamed "
             "or retired this model — check their current model list and type the correct "
             "name here. No code changes needed.",
    )

st.sidebar.header("3. Pipeline settings")
max_iterations = st.sidebar.slider("Max iterations", min_value=1, max_value=20, value=8)
cv_folds = st.sidebar.slider("Cross-validation folds", min_value=3, max_value=10, value=5)
patience = st.sidebar.slider("Patience (non-improving iterations before stopping)", min_value=1, max_value=10, value=3)
min_delta = st.sidebar.number_input(
    "Minimum ROC-AUC improvement to accept a feature", min_value=0.0, max_value=0.05,
    value=0.0005, step=0.0001, format="%.4f",
)

# --------------------------------------------------------------------------- #
# Main area
# --------------------------------------------------------------------------- #
st.title("Context-Aware LLM Feature Engineering")
st.write(
    "Upload a tabular dataset on the left, describe what it represents, and CAFE will "
    "automatically propose, test, and keep only the engineered features that measurably "
    "improve a cross-validated model — with every decision logged and explained."
)

# Everything below is deliberately nested inside this single `if df is not None:`
# block (rather than relying on st.stop() as a bail-out) so there is no code path
# that can reach df.shape / df.columns / etc. while df is still None.
if df is None:
    st.info("⬅️ Upload a CSV or Excel file in the sidebar to get started.")

else:
    st.subheader("Dataset preview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Missing values", f"{int(df.isna().sum().sum()):,}")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Configure the run")
    col_left, col_right = st.columns([1, 1])
    with col_left:
        target_column = st.selectbox("Target column (what you're predicting)", options=list(df.columns))
    with col_right:
        description = st.text_area(
            "Describe the dataset in a sentence or two",
            placeholder="e.g. Predict whether a patient will be readmitted within 30 days. "
                        "This is a hospital screening dataset; columns are patient vitals and history.",
            height=100,
        )

    # Basic guardrails before allowing a run
    problems = []
    target_nunique = df[target_column].nunique(dropna=True)
    if target_nunique < 2:
        problems.append("The target column needs at least two distinct values to compute ROC-AUC.")
    if target_nunique > 2:
        st.warning(
            "The target column has more than 2 unique values. The current evaluator is built for "
            "binary classification (ROC-AUC); results with a multi-class or continuous target may "
            "not be meaningful. Consider selecting a binary column."
        )
    if not description.strip():
        problems.append("Please add a short description of the dataset — this is what makes CAFE 'context-aware'.")
    if provider_choice != "Offline (Mock — no API key needed)" and not api_key:
        problems.append(f"Please enter your {provider_choice} API key, or switch to the Offline provider.")

    run_clicked = st.button("🚀 Run CAFE", type="primary", disabled=bool(problems))
    for p in problems:
        st.caption(f"⚠️ {p}")

    def build_provider():
        kwargs = {"api_key": api_key}
        if model_override:  # only override if the field isn't empty
            kwargs["model"] = model_override
        if provider_choice == "OpenAI":
            return OpenAIProvider(**kwargs)
        if provider_choice == "Anthropic (Claude)":
            return AnthropicProvider(**kwargs)
        if provider_choice == "Google (Gemini)":
            return GeminiProvider(**kwargs)
        return MockProvider()

    # ----------------------------------------------------------------------- #
    # Run the pipeline with live progress
    # ----------------------------------------------------------------------- #
    if run_clicked:
        st.session_state.auc_history = []
        st.session_state.report = None
        st.session_state.final_df = None

        provider = None
        try:
            provider = build_provider()
        except Exception as e:
            st.error(f"Could not initialize {provider_choice} provider: {e}")

        if provider is not None:
            status_box = st.status("Starting CAFE pipeline...", expanded=True)
            chart_placeholder = st.empty()
            log_placeholder = st.container()

            pipeline = CAFEPipeline(
                df=df,
                target_column=target_column,
                description=description,
                provider=provider,
                cv_folds=cv_folds,
                max_iterations=max_iterations,
                patience=patience,
                min_delta=min_delta,
            )

            def on_progress(event: dict):
                etype = event["type"]
                if etype == "baseline":
                    score = event["score"]
                    st.session_state.auc_history.append({"label": "Baseline", "auc": score})
                    status_box.write(f"📊 Baseline {cv_folds}-fold ROC-AUC: **{score:.4f}**")
                elif etype == "iteration":
                    rec = event["record"]
                    label = f"Iter {rec.iteration} ({rec.feature_name})"
                    if rec.score_after is not None:
                        st.session_state.auc_history.append({"label": label, "auc": rec.score_after})
                    icon = "✅" if rec.status == "accepted" else ("❌" if rec.status == "rejected_no_improvement" else "⚠️")
                    delta_str = f"{rec.delta:+.4f}" if rec.delta is not None else "n/a"
                    with log_placeholder:
                        st.write(f"{icon} **Iteration {rec.iteration}: `{rec.feature_name}`** — {rec.status} (Δ={delta_str})")
                        st.caption(rec.explanation if rec.status != "rejected_execution_error" else f"Execution error: {rec.error}")
                        with st.expander("Show generated code"):
                            st.code(rec.code, language="python")
                    status_box.write(f"Iteration {rec.iteration}: {rec.feature_name} — {rec.status} ({delta_str})")
                    hist_df = pd.DataFrame(st.session_state.auc_history)
                    if not hist_df.empty:
                        chart_placeholder.line_chart(hist_df.set_index("label")["auc"])
                elif etype == "stopped":
                    status_box.write(f"⏹️ Stopped: {event['reason']}")
                elif etype == "done":
                    status_box.update(label="CAFE run complete", state="complete", expanded=False)

            try:
                with st.spinner("Running..."):
                    report = pipeline.run(verbose=False, progress_callback=on_progress)
                st.session_state.report = report
                st.session_state.final_df = pipeline.final_df
            except Exception as e:
                status_box.update(label="CAFE run failed", state="error", expanded=True)
                st.error(
                    f"The pipeline stopped because the LLM provider raised an error:\n\n"
                    f"`{type(e).__name__}: {e}`\n\n"
                    f"This is usually either an invalid/expired API key, or a model name the "
                    f"provider no longer supports (try a different value in the 'Model name' "
                    f"field in the sidebar), or a rate limit. Nothing else in the app crashed — "
                    f"fix the setting above and click **Run CAFE** again."
                )

    # ----------------------------------------------------------------------- #
    # Results (persist across reruns via session_state)
    # ----------------------------------------------------------------------- #
    report: RunReport | None = st.session_state.report
    if report is not None and st.session_state.final_df is not None:
        st.divider()
        st.subheader("Results")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Baseline ROC-AUC", f"{report.baseline_auc:.4f}")
        m2.metric("Final ROC-AUC", f"{report.final_auc:.4f}",
                  delta=f"{report.final_auc - report.baseline_auc:+.4f}")
        m3.metric("Iterations run", len(report.iterations))
        m4.metric("Features accepted", len(report.accepted_features))

        hist_df = pd.DataFrame(st.session_state.auc_history)
        if not hist_df.empty:
            st.line_chart(hist_df.set_index("label")["auc"])

        st.markdown("**Accepted features:** " + (", ".join(f"`{f}`" for f in report.accepted_features) or "_none_"))

        st.subheader("Full iteration log")
        log_rows = [{
            "#": r.iteration, "Feature": r.feature_name, "Status": r.status,
            "Δ ROC-AUC": f"{r.delta:+.4f}" if r.delta is not None else "n/a",
            "Explanation": r.explanation,
        } for r in report.iterations]
        st.dataframe(pd.DataFrame(log_rows), use_container_width=True)

        st.subheader("Final feature-engineered dataset")
        st.dataframe(st.session_state.final_df.head(20), use_container_width=True)

        st.subheader("Download")
        d1, d2, d3 = st.columns(3)
        with d1:
            csv_bytes = st.session_state.final_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Final dataset (CSV)", csv_bytes, file_name="final_dataset.csv", mime="text/csv")
        with d2:
            json_bytes = json.dumps(report.to_json(), indent=2, default=str).encode("utf-8")
            st.download_button("⬇️ Run report (JSON)", json_bytes, file_name="run_report.json", mime="application/json")
        with d3:
            md_lines = [
                "# CAFE Run Report",
                f"- Provider: **{report.provider}**",
                f"- Target column: `{report.target_column}`",
                f"- Baseline ROC-AUC: **{report.baseline_auc:.4f}**",
                f"- Final ROC-AUC: **{report.final_auc:.4f}**",
                f"- Accepted features: {', '.join(report.accepted_features) or 'none'}",
            ]
            st.download_button("⬇️ Run report (Markdown)", "\n".join(md_lines).encode("utf-8"),
                                file_name="run_report.md", mime="text/markdown")
