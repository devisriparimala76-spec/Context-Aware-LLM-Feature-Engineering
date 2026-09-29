"""
evaluator.py
------------
Trains the downstream classifier and measures whether an engineered
feature actually helps, via stratified k-fold cross-validated ROC-AUC —
the same acceptance criterion used in the original CAAFE-style loop.

Nothing here is LLM-specific: this module is the objective, deterministic
"referee" that decides whether an LLM's (or MockProvider's) suggestion
is kept or discarded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


def _build_pipeline(random_state: int = 42) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=200, max_depth=None, random_state=random_state, n_jobs=-1
        )),
    ])


def _encode_non_numeric(X: pd.DataFrame) -> pd.DataFrame:
    """LLM-proposed categorical/binned features get ordinal-encoded before modeling."""
    X = X.copy()
    non_numeric = X.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X[non_numeric] = enc.fit_transform(X[non_numeric].astype(str))
    return X


def evaluate_roc_auc(df: pd.DataFrame, target_column: str, cv_folds: int = 5, random_state: int = 42) -> float:
    """Return mean cross-validated ROC-AUC for the given dataframe/target."""
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X = _encode_non_numeric(X)

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    pipeline = _build_pipeline(random_state=random_state)

    scores = cross_val_score(pipeline, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
    return float(np.mean(scores))
