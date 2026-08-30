"""
Verifier: checks a reasoner's candidate output on four independent gates.

This mirrors the four gates in your deployed app (calculations / evidence /
logic / sanity) but the evidence gate here checks *semantic* grounding via
a similarity score against the specific passage the reasoner cited, rather
than a plain substring match -- so a claim that paraphrases its source still
passes, while a claim with no real support still fails.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from .calculator import LedgerItem, margins_are_plausible
from .reasoner import ReasoningResult
from .retrieval import CrossEncoderFn, Passage


@dataclass
class Gate:
    id: str
    label: str
    passed: bool
    explanation: str


def calculations_gate(ledger: Sequence[LedgerItem]) -> Gate:
    mismatches = [item for item in ledger if item.status == "MISMATCH"]
    passed = len(mismatches) == 0
    return Gate(
        "calculations",
        "Independent calculations",
        passed,
        "Every stated value matches the deterministic recomputation."
        if passed
        else f"{len(mismatches)} value(s) diverge from the deterministic recomputation: "
        + ", ".join(m.label for m in mismatches),
    )


def evidence_gate(
    result: ReasoningResult,
    passages: Sequence[Passage],
    cross_encoder: CrossEncoderFn,
    threshold: float = 0.05,
) -> Gate:
    """A claim passes only if it cites a real passage AND that passage's
    cross-encoder relevance score to the claim clears `threshold`.

    `threshold` must be tuned against whichever cross-encoder you actually
    plug in -- ms-marco-MiniLM scores are unnormalized logits with a
    different scale than a fake test scorer. Re-tune this against a labeled
    validation set before trusting it at 10/10 for the paper; treat 0.05 as
    a starting point, not a validated constant. This is exactly the kind of
    threshold a journal reviewer will ask you to justify empirically.
    """
    if not result.claims:
        return Gate("evidence", "Explicit source support", False, "The reasoner made no checkable claims.")

    by_id = {p.id: p for p in passages}
    unsupported = []
    for claim in result.claims:
        passage = by_id.get(claim.supporting_passage_id or "")
        if passage is None:
            unsupported.append(claim.text)
            continue
        score = float(cross_encoder([(claim.text, passage.excerpt)])[0])
        if score < threshold:
            unsupported.append(claim.text)

    passed = len(unsupported) == 0
    return Gate(
        "evidence",
        "Explicit source support",
        passed,
        "Every claim is grounded in a retrieved passage above the relevance threshold."
        if passed
        else f"{len(unsupported)} claim(s) lack adequate source support: " + "; ".join(unsupported),
    )


def logic_gate(movement: float, conclusion_text: str, window: int = 6, epsilon: float = 0.015) -> Gate:
    """Direction sanity: does the conclusion's stated *margin* direction match
    the sign of the independently computed movement?

    Checks only a small word-window around each mention of "margin", not the
    whole sentence. A correct explanation routinely cites sub-component
    trends in the opposite direction within the very same sentence (e.g.
    "revenue GREW and expenses ROSE faster, compressing operating MARGIN") --
    scanning the whole sentence for contradiction words flags that as a false
    positive, since "grew"/"rose" describe revenue/expenses, not margin. This
    windowed version was arrived at after that exact false positive turned up
    in eval/run_eval.py's smoke test; it is still a heuristic, not a parser,
    and should be validated against labeled examples before trusting it for
    a paper -- an LLM-based self-consistency check is the more robust
    long-term replacement for this gate.
    """
    direction_word = "expand" if movement > epsilon else "contract" if movement < -epsilon else "unchanged"
    words = re.findall(r"[a-zA-Z']+", conclusion_text.lower())
    margin_windows = [
        words[max(0, i - window) : i + window]
        for i, w in enumerate(words)
        if w == "margin" or w == "margins"
    ] or [words]

    def any_contradiction(terms: tuple[str, ...]) -> bool:
        return any(any(term in w for term in terms) for win in margin_windows for w in win)

    contradicts_expand = direction_word == "expand" and any_contradiction(("declin", "contract", "fell", "decreas"))
    contradicts_contract = direction_word == "contract" and any_contradiction(("expand", "increas", "grew", "rose"))
    passed = not (contradicts_expand or contradicts_contract)
    return Gate(
        "logic",
        "Logical consistency",
        passed,
        "The conclusion's stated direction matches the independently computed movement."
        if passed
        else "The conclusion's stated direction conflicts with the computed movement's sign.",
    )


def sanity_gate(current_margin: float, prior_margin: float) -> Gate:
    passed = margins_are_plausible(current_margin, prior_margin)
    return Gate(
        "sanity",
        "Input completeness & financial sanity",
        passed,
        "Computed margins fall within the plausible -100%..100% range."
        if passed
        else "A computed margin falls outside the plausible range -- check inputs for unit errors.",
    )


def run_all_gates(
    ledger: Sequence[LedgerItem],
    result: ReasoningResult,
    passages: Sequence[Passage],
    cross_encoder: CrossEncoderFn,
    current_margin: float,
    prior_margin: float,
    movement: float,
) -> list[Gate]:
    return [
        calculations_gate(ledger),
        evidence_gate(result, passages, cross_encoder),
        logic_gate(movement, result.conclusion),
        sanity_gate(current_margin, prior_margin),
    ]
