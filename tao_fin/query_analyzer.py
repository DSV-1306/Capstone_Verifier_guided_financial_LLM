"""
Query analyzer.

Classifies a free-text financial question by difficulty and task type before
retrieval/reasoning runs. This is deliberately cheap and rule-based rather than
an LLM call -- classification doesn't need a model, and keeping it deterministic
means it's free to run and easy to unit test. It mirrors the same classification
used in your deployed web app (regex-based), kept consistent across both halves
of the project on purpose.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_CAUSAL = re.compile(r"\b(why|driver|cause|explain|due to|because)\b", re.I)
_MARGIN = re.compile(r"\b(margin|operating income|profitability)\b", re.I)
_RATIO = re.compile(r"\b(ratio|calculate|percent)\b", re.I)
_COMPARISON = re.compile(r"\b(versus|vs\.?|compared|change|movement)\b", re.I)


@dataclass
class QueryAnalysis:
    difficulty: str  # "single-fact" | "multi-step-causal"
    task_type: str  # "margin-driver" | "ratio-calculation" | "period-comparison" | "trend-explanation" | "financial-review"
    rationale: str


def analyze_query(question: str) -> QueryAnalysis:
    causal = bool(_CAUSAL.search(question))
    margin = bool(_MARGIN.search(question))
    ratio = bool(_RATIO.search(question))
    comparison = bool(_COMPARISON.search(question))

    difficulty = "multi-step-causal" if (causal or comparison) else "single-fact"

    if margin:
        task_type = "margin-driver"
    elif ratio:
        task_type = "ratio-calculation"
    elif comparison:
        task_type = "period-comparison"
    elif causal:
        task_type = "trend-explanation"
    else:
        task_type = "financial-review"

    rationale = (
        "The question asks for a causal explanation, so numeric movement and "
        "explicit source language must both reconcile before release."
        if causal
        else "The question is handled as a bounded financial review with "
        "deterministic formula checks."
    )
    return QueryAnalysis(difficulty=difficulty, task_type=task_type, rationale=rationale)
