"""
pipeline.py
-----------
The orchestrator. Implements the loop described in the project title:

    Dataset -> LLM understands dataset -> LLM generates feature
    -> Python executes feature -> Model retrains -> Accuracy checked
    -> Feedback sent back to LLM -> LLM creates a better feature -> Repeat

Every iteration is logged to a structured, JSON-serializable history so
the run is auditable end-to-end (which feature, which code, why kept or
discarded, and the exact score delta) — no black-box behavior.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from .code_executor import execute_feature_code
from .context_builder import DatasetContext
from .evaluator import evaluate_roc_auc
from .llm_client import LLMProvider, get_provider


@dataclass
class IterationRecord:
    iteration: int
    feature_name: str
    code: str
    explanation: str
    status: str            # "accepted" | "rejected_no_improvement" | "rejected_execution_error"
    score_before: float
    score_after: float | None
    delta: float | None
    error: str | None = None


@dataclass
class RunReport:
    provider: str
    target_column: str
    dataset_description: str
    baseline_auc: float
    final_auc: float
    iterations: list = field(default_factory=list)   # list[IterationRecord]
    accepted_features: list = field(default_factory=list)

    def to_json(self) -> dict:
        d = asdict(self)
        return d


class CAFEPipeline:
    """Context-Aware Feature Engineering pipeline."""

    def __init__(
        self,
        df: pd.DataFrame,
        target_column: str,
        description: str,
        provider: LLMProvider | None = None,
        cv_folds: int = 5,
        max_iterations: int = 8,
        patience: int = 3,
        min_delta: float = 0.0005,
        random_state: int = 42,
    ):
        self.df = df.copy()
        self.target_column = target_column
        self.description = description
        self.provider = provider or get_provider()
        self.cv_folds = cv_folds
        self.max_iterations = max_iterations
        self.patience = patience
        self.min_delta = min_delta
        self.random_state = random_state

    def run(self, verbose: bool = True, progress_callback=None) -> RunReport:
        """
        progress_callback, if given, is called with a dict after key stages:
          {"type": "baseline", "score": float}
          {"type": "iteration", "record": IterationRecord}
          {"type": "stopped", "reason": str}
          {"type": "done", "report": RunReport}
        This is what the Phase 2 dashboard (dashboard/app.py) uses to render
        live progress without needing to know anything about the pipeline
        internals — it only reacts to these events.
        """
        current_df = self.df.copy()
        context = DatasetContext(
            description=self.description,
            target_column=self.target_column,
            dataframe=current_df,
        )

        baseline = evaluate_roc_auc(current_df, self.target_column, self.cv_folds, self.random_state)
        if verbose:
            print(f"[pipeline] Baseline ROC-AUC ({self.cv_folds}-fold CV): {baseline:.4f}")
        if progress_callback:
            progress_callback({"type": "baseline", "score": baseline})

        report = RunReport(
            provider=self.provider.name,
            target_column=self.target_column,
            dataset_description=self.description,
            baseline_auc=baseline,
            final_auc=baseline,
        )

        current_score = baseline
        no_improve_streak = 0

        for i in range(1, self.max_iterations + 1):
            context.dataframe = current_df
            prompt_context = context.render()

            proposal = self.provider.propose_feature(prompt_context)
            feature_name = proposal["feature_name"]
            code = proposal["code"]
            explanation = proposal["explanation"]

            if feature_name == "no_further_features" or code.strip().startswith("pass"):
                if verbose:
                    print(f"[pipeline] Iteration {i}: provider signaled no further candidates. Stopping.")
                if progress_callback:
                    progress_callback({"type": "stopped", "reason": "provider signaled no further candidates"})
                break

            new_df, error = execute_feature_code(current_df, code, feature_name)

            if error is not None:
                record = IterationRecord(
                    iteration=i, feature_name=feature_name, code=code, explanation=explanation,
                    status="rejected_execution_error", score_before=current_score,
                    score_after=None, delta=None, error=error,
                )
                context.already_rejected.append({
                    "name": feature_name, "explanation": f"EXECUTION FAILED: {error}", "delta": 0.0
                })
                if verbose:
                    print(f"[pipeline] Iteration {i}: '{feature_name}' FAILED to execute -> {error}")
                report.iterations.append(record)
                if progress_callback:
                    progress_callback({"type": "iteration", "record": record})
                no_improve_streak += 1
                if no_improve_streak >= self.patience:
                    if verbose:
                        print(f"[pipeline] No improvement for {self.patience} iterations. Stopping.")
                    if progress_callback:
                        progress_callback({"type": "stopped", "reason": f"no improvement for {self.patience} iterations"})
                    break
                continue

            new_score = evaluate_roc_auc(new_df, self.target_column, self.cv_folds, self.random_state)
            delta = new_score - current_score

            if delta > self.min_delta:
                current_df = new_df
                current_score = new_score
                no_improve_streak = 0
                status = "accepted"
                context.already_accepted.append({"name": feature_name, "explanation": explanation, "delta": delta})
                report.accepted_features.append(feature_name)
                if verbose:
                    print(f"[pipeline] Iteration {i}: '{feature_name}' ACCEPTED "
                          f"(ROC-AUC {current_score - delta:.4f} -> {current_score:.4f}, Δ={delta:+.4f})")
            else:
                status = "rejected_no_improvement"
                context.already_rejected.append({"name": feature_name, "explanation": explanation, "delta": delta})
                no_improve_streak += 1
                if verbose:
                    print(f"[pipeline] Iteration {i}: '{feature_name}' REJECTED "
                          f"(Δ={delta:+.4f}, below min_delta={self.min_delta})")

            report.iterations.append(IterationRecord(
                iteration=i, feature_name=feature_name, code=code, explanation=explanation,
                status=status, score_before=current_score - (delta if status == "accepted" else 0),
                score_after=new_score, delta=delta,
            ))
            if progress_callback:
                progress_callback({"type": "iteration", "record": report.iterations[-1]})

            if no_improve_streak >= self.patience:
                if verbose:
                    print(f"[pipeline] No improvement for {self.patience} consecutive iterations. Stopping.")
                if progress_callback:
                    progress_callback({"type": "stopped", "reason": f"no improvement for {self.patience} consecutive iterations"})
                break

        report.final_auc = current_score
        self.final_df = current_df
        if progress_callback:
            progress_callback({"type": "done", "report": report})
        return report


def save_report(report: RunReport, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "run_report.json").write_text(
        json.dumps(report.to_json(), indent=2, default=str), encoding="utf-8"
    )

    lines = [
        "# Context-Aware LLM Feature Engineering — Run Report",
        f"- LLM provider: **{report.provider}**",
        f"- Target column: `{report.target_column}`",
        f"- Baseline ROC-AUC: **{report.baseline_auc:.4f}**",
        f"- Final ROC-AUC: **{report.final_auc:.4f}**",
        f"- Net improvement: **{report.final_auc - report.baseline_auc:+.4f}**",
        f"- Accepted features ({len(report.accepted_features)}): {', '.join(report.accepted_features) or 'none'}",
        "",
        "## Iteration log",
        "",
        "| # | Feature | Status | Δ ROC-AUC | Explanation |",
        "|---|---------|--------|-----------|-------------|",
    ]
    for rec in report.iterations:
        delta_str = f"{rec.delta:+.4f}" if rec.delta is not None else "n/a"
        lines.append(f"| {rec.iteration} | `{rec.feature_name}` | {rec.status} | {delta_str} | {rec.explanation} |")

    (out_dir / "run_report.md").write_text("\n".join(lines), encoding="utf-8")
