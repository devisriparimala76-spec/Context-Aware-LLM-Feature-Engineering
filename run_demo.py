"""
run_demo.py
-----------
End-to-end demonstration of the Context-Aware LLM Feature Engineering
pipeline on a synthetic healthcare (diabetes-style) dataset.

    python examples/run_demo.py

By default this uses whatever LLM provider is configured via environment
variables (ANTHROPIC_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY), or the
offline MockProvider if none are set — see src/llm_client.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.make_sample_dataset import make_diabetes_like_dataset  # noqa: E402
from src.pipeline import CAFEPipeline, save_report  # noqa: E402

DATASET_DESCRIPTION = (
    "Predict whether a patient will be diagnosed with diabetes (Outcome=1) "
    "within the study period. This is a clinical/healthcare dataset. Columns "
    "represent patient vitals and history collected at a routine screening."
)


def main():
    df = make_diabetes_like_dataset(n=600, seed=7)

    pipeline = CAFEPipeline(
        df=df,
        target_column="Outcome",
        description=DATASET_DESCRIPTION,
        cv_folds=5,
        max_iterations=8,
        patience=3,
        min_delta=0.0005,
    )

    report = pipeline.run(verbose=True)

    out_dir = Path(__file__).resolve().parents[1] / "outputs"
    save_report(report, out_dir)

    # persist the final, feature-engineered dataset too (traceability)
    pipeline.final_df.to_csv(out_dir / "final_dataset.csv", index=False)

    print("\n=== SUMMARY ===")
    print(f"Provider:        {report.provider}")
    print(f"Baseline AUC:    {report.baseline_auc:.4f}")
    print(f"Final AUC:       {report.final_auc:.4f}")
    print(f"Net improvement: {report.final_auc - report.baseline_auc:+.4f}")
    print(f"Accepted features: {report.accepted_features}")
    print(f"\nFull report written to: {out_dir}/run_report.md and run_report.json")


if __name__ == "__main__":
    main()
