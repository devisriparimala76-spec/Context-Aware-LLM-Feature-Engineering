"""
context_builder.py
-------------------
Builds the natural-language "context" that makes the LLM's feature
suggestions domain-aware instead of purely statistical.

This is the module that operationalizes the "Context-Aware" part of the
project title: instead of handing the LLM a bag of anonymous columns,
we hand it column names, dtypes, sample rows, summary statistics, the
target variable, and a free-text dataset description (task, domain,
units) supplied by the user.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class DatasetContext:
    """Container for everything the LLM is told about the dataset."""

    description: str
    target_column: str
    dataframe: pd.DataFrame
    already_accepted: list = field(default_factory=list)   # [{name, code, explanation, delta}]
    already_rejected: list = field(default_factory=list)   # [{name, code, explanation, delta}]

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def render(self, max_sample_rows: int = 5) -> str:
        """Render the full context block sent to the LLM at each iteration."""
        df = self.dataframe
        feature_cols = [c for c in df.columns if c != self.target_column]

        dtypes_block = "\n".join(
            f"  - {col} ({df[col].dtype})" for col in feature_cols
        )

        # Cheap, robust summary stats (works for numeric + categorical columns)
        stats_lines = []
        for col in feature_cols:
            series = df[col]
            if pd.api.types.is_numeric_dtype(series):
                stats_lines.append(
                    f"  - {col}: min={series.min():.3g}, max={series.max():.3g}, "
                    f"mean={series.mean():.3g}, missing={series.isna().sum()}"
                )
            else:
                top = series.value_counts().head(3).to_dict()
                stats_lines.append(f"  - {col}: top values={top}, missing={series.isna().sum()}")
        stats_block = "\n".join(stats_lines)

        sample_block = df[feature_cols + [self.target_column]].head(max_sample_rows).to_string(index=False)

        history_block = self._render_history()

        return textwrap.dedent(f"""
        DATASET DESCRIPTION (domain / prediction task, provided by the user):
        \"\"\"{self.description}\"\"\"

        TARGET COLUMN: {self.target_column}

        FEATURE COLUMNS AND DTYPES:
        {dtypes_block}

        SUMMARY STATISTICS:
        {stats_block}

        SAMPLE ROWS:
        {sample_block}

        {history_block}
        """).strip()

    def _render_history(self) -> str:
        if not self.already_accepted and not self.already_rejected:
            return "FEATURE HISTORY: (none yet — this is the first iteration)"

        lines = ["FEATURE HISTORY (do not repeat these; build on what worked, avoid what failed):"]
        for item in self.already_accepted:
            lines.append(
                f"  [ACCEPTED] {item['name']}: {item['explanation']} "
                f"(ROC-AUC change: {item['delta']:+.4f})"
            )
        for item in self.already_rejected:
            lines.append(
                f"  [REJECTED] {item['name']}: {item['explanation']} "
                f"(ROC-AUC change: {item['delta']:+.4f})"
            )
        return "\n".join(lines)

    def detect_domain_keywords(self) -> list[str]:
        """
        Cheap keyword-based domain detection, used by the MockProvider
        (see llm_client.py) to decide which template family of features
        to propose when no real LLM API key is configured. A real LLM
        provider does not need this — it infers domain directly from
        `description` and column names via the prompt.
        """
        text = (self.description + " " + " ".join(self.dataframe.columns)).lower()
        domains = []
        if any(k in text for k in ["patient", "disease", "diabet", "health", "blood", "hospital", "clinical"]):
            domains.append("healthcare")
        if any(k in text for k in ["loan", "credit", "bank", "transaction", "balance", "income", "debt"]):
            domains.append("finance")
        if any(k in text for k in ["temperature", "humidity", "rainfall", "weather", "climate"]):
            domains.append("weather")
        if any(k in text for k in ["price", "sale", "customer", "product", "revenue"]):
            domains.append("retail")
        return domains or ["generic"]
