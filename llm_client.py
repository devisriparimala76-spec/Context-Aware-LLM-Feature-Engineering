"""
llm_client.py
-------------
Pluggable LLM layer. One `LLMProvider` interface, several backends.

Swap providers with an environment variable — no other code changes:

    export CAFE_LLM_PROVIDER=anthropic   # or openai, gemini, mock
    export ANTHROPIC_API_KEY=sk-...

If no provider / API key is configured, the framework automatically
falls back to `MockProvider`, a deterministic, offline, keyword-driven
"pseudo-LLM" that follows the exact same JSON contract a real LLM
would. This keeps the pipeline fully runnable (and unit-testable) in
network-restricted environments, and makes it obvious in every log line
that it is a stand-in, not a real language model.

CONTRACT every provider must satisfy
-------------------------------------
`propose_feature(context: str) -> dict` must return:
    {
        "feature_name": str,
        "code": str,          # single pandas expression assigning df["<feature_name>"]
        "explanation": str,   # human-readable rationale, shown to the auditor/reviewer
    }
"""

from __future__ import annotations

import json
import os
import random
import re
from abc import ABC, abstractmethod


SYSTEM_PROMPT = """You are an expert data scientist doing automated feature engineering.
You will be shown a dataset's description, columns, dtypes, statistics, sample rows,
and the history of features already tried (accepted/rejected) with their measured
impact on model ROC-AUC.

Propose exactly ONE new engineered feature that is:
  - grounded in domain meaning (use the dataset description, not just column names),
  - different from every feature already tried,
  - expressible as a single pandas assignment using only existing columns.

Respond with STRICT JSON only, no markdown fences, no commentary, matching:
{"feature_name": "...", "code": "df['feature_name'] = ...", "explanation": "..."}
"""


class LLMProvider(ABC):
    name: str = "base"

    @abstractmethod
    def propose_feature(self, context: str) -> dict:
        ...


# --------------------------------------------------------------------------- #
# Real providers — used automatically if the corresponding SDK + API key
# are present. Each is a thin, isolated adapter so a real network call is
# never more than one function away from the rest of the pipeline.
# --------------------------------------------------------------------------- #
class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, model: str = "gpt-4o", api_key: str | None = None):
        import openai  # imported lazily so the package is optional

        self.client = openai.OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
        self.model = model

    def propose_feature(self, context: str) -> dict:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": context},
            ],
            temperature=0.7,
        )
        return _parse_json_response(resp.choices[0].message.content)


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-5", api_key: str | None = None):
        import anthropic  # imported lazily so the package is optional

        self.client = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])
        self.model = model

    def propose_feature(self, context: str) -> dict:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=800,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": context}],
        )
        text = "".join(block.text for block in resp.content if block.type == "text")
        return _parse_json_response(text)


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, model: str = "gemini-flash-latest", api_key: str | None = None):
        import google.generativeai as genai  # imported lazily so the package is optional

        genai.configure(api_key=api_key or os.environ["GOOGLE_API_KEY"])
        self.model = genai.GenerativeModel(model, system_instruction=SYSTEM_PROMPT)

    def propose_feature(self, context: str) -> dict:
        resp = self.model.generate_content(context)
        return _parse_json_response(resp.text)


def _parse_json_response(raw: str) -> dict:
    """LLMs occasionally wrap JSON in markdown fences despite instructions; strip them."""
    cleaned = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    data = json.loads(cleaned)
    for key in ("feature_name", "code", "explanation"):
        if key not in data:
            raise ValueError(f"LLM response missing required key: {key}")
    return data


# --------------------------------------------------------------------------- #
# MockProvider — offline stand-in used for the reference demo in this repo
# and in unit tests. It is intentionally simple (keyword lookup, not
# reasoning) precisely so nobody mistakes it for a real result: it exists
# to prove the PIPELINE works end-to-end, not to prove an LLM is smart.
# --------------------------------------------------------------------------- #
class MockProvider(LLMProvider):
    name = "mock"

    # Candidate features per detected domain. Each entry is checked against
    # the dataframe's actual columns before being proposed, so it degrades
    # gracefully on datasets that don't match.
    TEMPLATES = {
        "healthcare": [
            {
                "requires": ["Weight", "Height"],
                "feature_name": "BMI",
                "code": "df['BMI'] = df['Weight'] / (df['Height'] ** 2)",
                "explanation": "Body Mass Index is a standard clinical derived measure "
                               "combining weight and height, commonly predictive in "
                               "metabolic-disease datasets.",
            },
            {
                "requires": ["Age"],
                "feature_name": "Age_Group",
                "code": "df['Age_Group'] = pd.cut(df['Age'], bins=[0,30,45,60,120], "
                        "labels=[0,1,2,3]).astype(float)",
                "explanation": "Binning age into clinically meaningful groups "
                               "(young adult / middle-age / senior / elderly) can "
                               "surface non-linear risk effects a raw age column hides.",
            },
            {
                "requires": ["Sugar", "Insulin"],
                "feature_name": "Sugar_Insulin_Ratio",
                "code": "df['Sugar_Insulin_Ratio'] = df['Sugar'] / (df['Insulin'] + 1)",
                "explanation": "The ratio of blood sugar to insulin is a proxy used in "
                               "clinical literature for insulin resistance, more "
                               "informative than either raw value alone.",
            },
        ],
        "finance": [
            {
                "requires": ["Balance", "CreditLimit"],
                "feature_name": "Utilization_Ratio",
                "code": "df['Utilization_Ratio'] = df['Balance'] / (df['CreditLimit'] + 1)",
                "explanation": "Credit utilization ratio (balance / limit) is one of the "
                               "strongest known predictors of credit risk.",
            },
            {
                "requires": ["Income", "Debt"],
                "feature_name": "Debt_to_Income",
                "code": "df['Debt_to_Income'] = df['Debt'] / (df['Income'] + 1)",
                "explanation": "Debt-to-income ratio is a standard underwriting signal "
                               "for repayment capacity.",
            },
        ],
        "weather": [
            {
                "requires": ["Temperature", "Humidity"],
                "feature_name": "Heat_Index_Proxy",
                "code": "df['Heat_Index_Proxy'] = df['Temperature'] + 0.05 * df['Humidity']",
                "explanation": "A simplified heat-index proxy combining temperature and "
                               "humidity captures perceived heat better than either alone.",
            },
        ],
        "generic": [],  # filled dynamically below
    }

    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)

    def propose_feature(self, context: str) -> dict:
        # The mock "reads" context the same crude way an LLM prompt would be
        # grounded: by scanning the rendered context text for domain and
        # column-name cues. This keeps the demo honest about being a stand-in.
        domain = self._infer_domain(context)
        rejected_or_accepted = self._already_tried(context)
        columns = self._columns_mentioned(context)

        candidates = [
            t for t in self.TEMPLATES.get(domain, [])
            if all(col in columns for col in t["requires"])
            and t["feature_name"] not in rejected_or_accepted
        ]

        if not candidates:
            # Generic numeric-interaction fallback so the loop never stalls
            numeric_cols = [c for c in columns if c not in rejected_or_accepted][:2]
            if len(numeric_cols) >= 2:
                a, b = numeric_cols[0], numeric_cols[1]
                name = f"{a}_x_{b}_interaction"
                if name in rejected_or_accepted:
                    return self._noop(context)
                return {
                    "feature_name": name,
                    "code": f"df['{name}'] = df['{a}'] * df['{b}']",
                    "explanation": f"Generic interaction term between {a} and {b}, "
                                   "proposed as a fallback once domain-specific "
                                   "templates were exhausted.",
                }
            return self._noop(context)

        choice = self._rng.choice(candidates)
        return {k: choice[k] for k in ("feature_name", "code", "explanation")}

    @staticmethod
    def _infer_domain(context: str) -> str:
        text = context.lower()
        if any(k in text for k in ["patient", "diabet", "health", "blood", "hospital"]):
            return "healthcare"
        if any(k in text for k in ["loan", "credit", "balance", "income", "debt"]):
            return "finance"
        if any(k in text for k in ["temperature", "humidity", "rainfall", "weather"]):
            return "weather"
        return "generic"

    @staticmethod
    def _columns_mentioned(context: str) -> list[str]:
        match = re.search(r"FEATURE COLUMNS AND DTYPES:\n(.*?)\n\n", context, re.DOTALL)
        if not match:
            return []
        return [line.strip("  - ").split(" (")[0] for line in match.group(1).splitlines()]

    @staticmethod
    def _already_tried(context: str) -> set:
        names = set(re.findall(r"\[(?:ACCEPTED|REJECTED)\] ([\w_]+):", context))
        return names

    def _noop(self, context: str) -> dict:
        return {
            "feature_name": "no_further_features",
            "code": "pass  # MockProvider exhausted its template library for this dataset",
            "explanation": "No further domain-matched or generic candidate features "
                           "remain to propose.",
        }


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def get_provider(name: str | None = None) -> LLMProvider:
    """
    Resolve which provider to use, in order of preference:
    explicit `name` arg > CAFE_LLM_PROVIDER env var > first available API key > MockProvider.
    """
    name = (name or os.environ.get("CAFE_LLM_PROVIDER", "")).lower()

    if name == "openai" or (not name and os.environ.get("OPENAI_API_KEY")):
        try:
            return OpenAIProvider()
        except Exception as e:
            print(f"[llm_client] OpenAI provider unavailable ({e}); falling back to MockProvider.")

    if name == "anthropic" or (not name and os.environ.get("ANTHROPIC_API_KEY")):
        try:
            return AnthropicProvider()
        except Exception as e:
            print(f"[llm_client] Anthropic provider unavailable ({e}); falling back to MockProvider.")

    if name == "gemini" or (not name and os.environ.get("GOOGLE_API_KEY")):
        try:
            return GeminiProvider()
        except Exception as e:
            print(f"[llm_client] Gemini provider unavailable ({e}); falling back to MockProvider.")

    if name and name != "mock":
        print(f"[llm_client] Unknown/unavailable provider '{name}'; falling back to MockProvider.")

    print("[llm_client] Using MockProvider (offline, deterministic, keyword-driven). "
          "Set ANTHROPIC_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY to use a real LLM.")
    return MockProvider()
