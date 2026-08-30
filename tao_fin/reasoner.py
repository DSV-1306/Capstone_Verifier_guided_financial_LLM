"""
The Reasoner: the component that was entirely absent from the deployed Manus
app. This is what actually calls an LLM to produce candidate causal reasoning
grounded in retrieved evidence -- as opposed to the deployed app, where the
"conclusion" was 100% string-templated from arithmetic plus whatever the user
typed into a manual field.

Requires ANTHROPIC_API_KEY in the environment to make real calls. The client
is dependency-injected so tests can supply a fake and verify parsing/prompt
logic without a real key or a paid call.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Sequence

SYSTEM_PROMPT = """You are a financial-reasoning engine. You are given a \
question, a set of retrieved evidence passages from a company's SEC filing, \
and a deterministically pre-computed calculation ledger. Produce a causal \
explanation for the numeric movement, using ONLY the retrieved evidence -- \
never invent a driver that isn't stated in the passages.

Respond with ONLY a JSON object (no markdown fences, no preamble) of the form:
{
  "conclusion": "<1-3 sentence causal explanation>",
  "claims": [
    {"text": "<a single factual/causal claim from the conclusion>",
     "supporting_passage_id": "<id of the passage that supports it, or null \
if you cannot find explicit support>"}
  ]
}

Every claim in "claims" must correspond to a sentence-level assertion in your \
conclusion. If no passage explicitly supports a causal claim you want to make, \
either drop the claim or set supporting_passage_id to null -- do not guess.
"""


@dataclass
class Claim:
    text: str
    supporting_passage_id: str | None


@dataclass
class ReasoningResult:
    conclusion: str
    claims: list[Claim] = field(default_factory=list)
    raw_response: str = ""


CompletionFn = Callable[[str, str], str]  # (system, user) -> raw text response


def default_completion_fn() -> CompletionFn:
    """Real Anthropic API call. Requires ANTHROPIC_API_KEY in the environment."""
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def complete(system: str, user: str) -> str:
        response = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=600,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in response.content if block.type == "text")

    return complete


def _build_user_prompt(question: str, ticker: str, passages: Sequence, ledger_summary: str) -> str:
    evidence_block = "\n\n".join(f"[{p.id}] {p.excerpt}" for p in passages) or "(no evidence retrieved)"
    return (
        f"Question: {question}\n"
        f"Company: {ticker}\n\n"
        f"Calculation ledger (already independently computed, do not recompute):\n"
        f"{ledger_summary}\n\n"
        f"Retrieved evidence passages:\n{evidence_block}"
    )


def _parse_response(raw: str) -> ReasoningResult:
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Reasoner did not return valid JSON: {raw!r}") from exc
    claims = [
        Claim(text=c["text"], supporting_passage_id=c.get("supporting_passage_id"))
        for c in payload.get("claims", [])
    ]
    return ReasoningResult(conclusion=payload["conclusion"], claims=claims, raw_response=raw)


def reason(
    question: str,
    ticker: str,
    passages: Sequence,
    ledger_summary: str,
    completion_fn: CompletionFn | None = None,
) -> ReasoningResult:
    """Generate grounded causal reasoning over retrieved evidence.

    This is a real model call (or an injected fake in tests) -- not a
    template. The verifier downstream is responsible for checking whether
    each returned claim's `supporting_passage_id` actually supports it.
    """
    completion_fn = completion_fn or default_completion_fn()
    user_prompt = _build_user_prompt(question, ticker, passages, ledger_summary)
    raw = completion_fn(SYSTEM_PROMPT, user_prompt)
    return _parse_response(raw)
