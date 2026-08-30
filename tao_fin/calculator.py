"""
Deterministic financial calculator.

This never touches an LLM. It exists so the TAO controller has a ground-truth
source of arithmetic to check the reasoner's claims against, independent of
whatever the language model says.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LedgerItem:
    label: str
    formula: str
    result: float
    status: str  # "MATCHED" | "MISMATCH" | "NOT_PROVIDED"


def _round(x: float) -> float:
    return round(x, 2)


def compute_margin_ledger(
    current_revenue: float,
    current_operating_income: float,
    prior_revenue: float,
    prior_operating_income: float,
    stated_current_margin: float | None = None,
    stated_prior_margin: float | None = None,
    stated_movement: float | None = None,
    epsilon: float = 0.015,
) -> list[LedgerItem]:
    """Recompute operating margin and its period-over-period movement.

    Every value here is deterministic arithmetic. If `stated_*` values are
    supplied (e.g. what the reasoner claimed), we flag MISMATCH rather than
    silently trusting them -- that's what feeds the verifier's
    'independent calculations' gate.
    """
    current_margin = _round((current_operating_income / current_revenue) * 100)
    prior_margin = _round((prior_operating_income / prior_revenue) * 100)
    movement = _round(current_margin - prior_margin)

    def status_for(stated: float | None, computed: float) -> str:
        if stated is None:
            return "NOT_PROVIDED"
        return "MATCHED" if abs(stated - computed) <= epsilon else "MISMATCH"

    return [
        LedgerItem(
            "Current operating margin",
            f"{current_operating_income:,.0f} / {current_revenue:,.0f} x 100",
            current_margin,
            status_for(stated_current_margin, current_margin),
        ),
        LedgerItem(
            "Prior operating margin",
            f"{prior_operating_income:,.0f} / {prior_revenue:,.0f} x 100",
            prior_margin,
            status_for(stated_prior_margin, prior_margin),
        ),
        LedgerItem(
            "Operating-margin movement",
            f"{current_margin:.2f}% - {prior_margin:.2f}%",
            movement,
            status_for(stated_movement, movement),
        ),
    ]


def margins_are_plausible(*margins: float, bound: float = 100.0) -> bool:
    """Financial sanity check: operating margin should realistically sit in (-100%, 100%)."""
    return all(abs(m) <= bound for m in margins)
