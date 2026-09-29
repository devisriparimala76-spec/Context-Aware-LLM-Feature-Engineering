"""
Standalone test runner that does not require pytest to be installed.
Exercises the same checks as test_pipeline.py, for use in restricted/offline
environments. Prefer `pytest tests/` when pytest is available.
"""
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.make_sample_dataset import make_diabetes_like_dataset
from src.code_executor import execute_feature_code
from src.context_builder import DatasetContext
from src.evaluator import evaluate_roc_auc
from src.llm_client import MockProvider
from src.pipeline import CAFEPipeline

sample_df = make_diabetes_like_dataset(n=200, seed=1)


def test_dataset_generation_shape():
    assert sample_df.shape[0] == 200
    assert "Outcome" in sample_df.columns
    assert set(sample_df["Outcome"].unique()) <= {0, 1}


def test_context_builder_renders_key_sections():
    ctx = DatasetContext(description="Predict diabetes.", target_column="Outcome", dataframe=sample_df)
    rendered = ctx.render()
    assert "TARGET COLUMN: Outcome" in rendered
    assert "FEATURE COLUMNS AND DTYPES" in rendered
    assert "SAMPLE ROWS" in rendered


def test_code_executor_accepts_valid_feature():
    code = "df['BMI'] = df['Weight'] / (df['Height'] ** 2)"
    new_df, error = execute_feature_code(sample_df, code, "BMI")
    assert error is None
    assert "BMI" in new_df.columns
    assert new_df["BMI"].notna().all()


def test_code_executor_rejects_bad_code():
    code = "df['Bad'] = df['NonexistentColumn'] * 2"
    new_df, error = execute_feature_code(sample_df, code, "Bad")
    assert new_df is None
    assert error is not None


def test_code_executor_sandboxed_from_filesystem():
    code = "import os\ndf['x'] = 1"
    new_df, error = execute_feature_code(sample_df, code, "x")
    assert new_df is None
    assert error is not None


def test_evaluator_returns_reasonable_auc():
    auc = evaluate_roc_auc(sample_df, "Outcome", cv_folds=3)
    assert 0.0 <= auc <= 1.0


def test_mock_provider_returns_contract_compliant_dict():
    ctx = DatasetContext(description="Predict diabetes onset.", target_column="Outcome", dataframe=sample_df)
    provider = MockProvider(seed=0)
    proposal = provider.propose_feature(ctx.render())
    assert set(proposal.keys()) == {"feature_name", "code", "explanation"}


def test_full_pipeline_runs_and_improves_or_matches_baseline():
    pipeline = CAFEPipeline(
        df=sample_df, target_column="Outcome",
        description="Predict diabetes onset. Healthcare screening dataset.",
        provider=MockProvider(seed=0), cv_folds=3, max_iterations=4, patience=2,
    )
    report = pipeline.run(verbose=False)
    assert report.final_auc >= report.baseline_auc - 1e-9
    assert isinstance(report.accepted_features, list)


def test_progress_callback_fires_expected_event_sequence():
    """Regression test for the dashboard integration (dashboard/app.py) — the callback
    contract it depends on must keep emitting baseline -> iteration* -> stopped/done."""
    events = []
    pipeline = CAFEPipeline(
        df=sample_df, target_column="Outcome",
        description="Predict diabetes onset. Healthcare screening dataset.",
        provider=MockProvider(seed=0), cv_folds=3, max_iterations=4, patience=2,
    )
    report = pipeline.run(verbose=False, progress_callback=lambda e: events.append(e["type"]))

    assert events[0] == "baseline"
    assert events[-1] == "done"
    assert events.count("iteration") == len(report.iterations)
    for etype in events:
        assert etype in {"baseline", "iteration", "stopped", "done"}


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL  {t.__name__} -> {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
