"""
Generates a synthetic, diabetes-style tabular dataset so the demo runs
fully offline with no external download. Column names are deliberately
realistic (Age, Weight, Height, Sugar, Insulin, BloodPressure,
Pregnancies) so the MockProvider's healthcare templates — and any real
LLM — have genuine domain signal to reason about, and BMI /
Sugar-Insulin-Ratio are true (if noisy) drivers of the label, so an
LLM discovering them is a meaningful, verifiable test of the pipeline.
"""

import numpy as np
import pandas as pd


def make_diabetes_like_dataset(n: int = 600, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.integers(21, 75, size=n)
    height_m = rng.normal(1.65, 0.09, size=n).clip(1.4, 2.0)
    weight_kg = rng.normal(70, 15, size=n).clip(40, 140)
    pregnancies = rng.integers(0, 6, size=n)
    blood_pressure = rng.normal(75, 12, size=n).clip(50, 120)
    sugar = rng.normal(110, 30, size=n).clip(60, 250)
    insulin = rng.normal(80, 40, size=n).clip(0, 300)

    bmi = weight_kg / (height_m ** 2)
    sugar_insulin_ratio = sugar / (insulin + 1)

    # True (noisy) generative process the model/LLM has to *rediscover*
    logit = (
        0.09 * (bmi - 25)
        + 0.55 * (sugar_insulin_ratio - 1.2)
        + 0.02 * (age - 40)
        + 0.15 * pregnancies
        - 0.01 * (blood_pressure - 75)
        + rng.normal(0, 1.0, size=n)
    )
    prob = 1 / (1 + np.exp(-logit))
    outcome = (rng.uniform(size=n) < prob).astype(int)

    df = pd.DataFrame({
        "Age": age,
        "Height": height_m.round(2),
        "Weight": weight_kg.round(1),
        "Pregnancies": pregnancies,
        "BloodPressure": blood_pressure.round(1),
        "Sugar": sugar.round(1),
        "Insulin": insulin.round(1),
        "Outcome": outcome,
    })
    return df


if __name__ == "__main__":
    df = make_diabetes_like_dataset()
    df.to_csv("sample_diabetes.csv", index=False)
    print(f"Wrote sample_diabetes.csv with shape {df.shape}, positive rate={df['Outcome'].mean():.2%}")
