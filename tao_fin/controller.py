"""
The TAO controller: given the verifier's gate results, decides whether to
stop, revise the reasoning, retrieve more evidence, or recalculate.

This is pure decision logic with no side effects, which is what makes it
possible to unit-test every branch (STOP/REVISE/RETRIEVE/RECALCULATE)
directly against hand-built gate combinations -- see tests/test_controller.py.
"""
from __future__ import annotations

from dataclasses import dataclass

from .verifier import Gate

MAX_ITERATIONS = 3
Action = str  # "STOP" | "REVISE" | "RETRIEVE" | "RECALCULATE"


def choose_action(gates: list[Gate]) -> Action:
    by_id = {g.id: g for g in gates}
    if all(g.passed for g in gates):
        return "STOP"
    if not by_id["logic"].passed or not by_id["sanity"].passed:
        return "REVISE"
    if not by_id["evidence"].passed:
        return "RETRIEVE"
    if not by_id["calculations"].passed:
        return "RECALCULATE"
    return "STOP"  # unreachable if gate set is exhaustive; safe fallback


@dataclass
class IterationRecord:
    iteration: int
    action: Action
    gates: list[Gate]
    verifier_score: float  # fraction of gates passed, 0..1


def confidence_from_trace(
    trace: list[IterationRecord],
    max_iterations: int = MAX_ITERATIONS,
) -> float:
    """Confidence derived from (a) how many gates passed on the final
    iteration and (b) how much of the iteration budget was actually needed.
    Never a flat number -- a run that resolves cleanly on iteration 1 scores
    higher than one that grinds through all 3 iterations and still fails.
    """
    final = trace[-1]
    gate_fraction = final.verifier_score
    iterations_used = len(trace)
    saved_credit = (max_iterations - iterations_used) * 7
    action_penalty = {"STOP": 0, "REVISE": 4, "RETRIEVE": 7, "RECALCULATE": 5}
    penalty = sum(action_penalty[r.action] for r in trace)
    raw = 50 + gate_fraction * 32 + saved_credit - penalty
    return max(0.0, min(99.0, round(raw, 1)))


def compute_saved_percent(trace: list[IterationRecord], max_iterations: int = MAX_ITERATIONS) -> float:
    return round(((max_iterations - len(trace)) / max_iterations) * 100, 1)
