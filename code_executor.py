"""
code_executor.py
-----------------
Executes LLM-proposed pandas code against a COPY of the dataframe, inside
a restricted namespace. Nothing the LLM writes can touch the filesystem,
network, or the original dataframe — if the code raises, we just discard
the copy and report the failure back into the pipeline's history so the
LLM (or MockProvider) sees it next iteration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Deliberately small allow-list. No __import__, no builtins beyond what's needed.
_SAFE_BUILTINS = {
    "len": len, "range": range, "min": min, "max": max, "abs": abs,
    "sum": sum, "round": round, "float": float, "int": int, "str": str,
    "bool": bool, "list": list, "dict": dict, "sorted": sorted,
}


def execute_feature_code(df: pd.DataFrame, code: str, feature_name: str) -> tuple[pd.DataFrame | None, str | None]:
    """
    Run `code` (expected to assign df['feature_name'] = ...) against a copy of df.

    Returns (new_df, None) on success, or (None, error_message) on failure.
    """
    sandbox_df = df.copy(deep=True)
    local_ns = {"df": sandbox_df, "pd": pd, "np": np}

    try:
        exec(code, {"__builtins__": _SAFE_BUILTINS}, local_ns)  # noqa: S102 - sandboxed on purpose
    except Exception as e:  # broad on purpose: any failure just becomes a rejected candidate
        return None, f"{type(e).__name__}: {e}"

    result_df = local_ns["df"]

    if feature_name not in result_df.columns:
        return None, f"Code executed but did not create column '{feature_name}'."

    new_col = result_df[feature_name]
    if new_col.isna().all():
        return None, f"Column '{feature_name}' is entirely NaN — likely a bad formula."

    if not (pd.api.types.is_numeric_dtype(new_col) or pd.api.types.is_bool_dtype(new_col)):
        # allow small categorical cardinality (will be one-hot/ordinal handled by the model step)
        if new_col.nunique() > 50:
            return None, f"Column '{feature_name}' is non-numeric with high cardinality ({new_col.nunique()})."

    return result_df, None
