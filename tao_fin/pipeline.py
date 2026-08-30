"""
End-to-end pipeline: wires the Query Analyzer, Calculator, Retriever,
Reasoner, Verifier, and TAO Controller together into one controlled run.

This is the piece that didn't exist anywhere in the deployed Manus app --
there, "evidence" and "numbers" were static inputs and the "conclusion" was a
string template. Here, retrieval is live, the Reasoner is a real LLM call,
and the controller genuinely loops the Reasoner/Retriever based on which
gate failed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from .calculator import LedgerItem, compute_margin_ledger
from .controller import MAX_ITERATIONS, IterationRecord, choose_action, compute_saved_percent, confidence_from_trace
from .query_analyzer import QueryAnalysis, analyze_query
from .reasoner import CompletionFn, ReasoningResult, reason
from .retrieval import CrossEncoderFn, EmbedderFn, Passage
from .verifier import Gate, run_all_gates

RetrieveFn = Callable[[bool], Sequence[Passage]]  # (expanded) -> passages


@dataclass
class AnalysisResult:
    question: str
    ticker: str
    query_analysis: QueryAnalysis
    ledger: list[LedgerItem]
    conclusion: str
    claims: list
    gates: list[Gate]
    release_status: str  # "VERIFIED" | "NEEDS_REVIEW"
    confidence: float
    trace: list[IterationRecord] = field(default_factory=list)
    compute_saved_percent: float = 0.0


def _ledger_summary(ledger: Sequence[LedgerItem]) -> str:
    return "\n".join(f"- {item.label}: {item.formula} = {item.result}" for item in ledger)


def _feedback_note(gates: Sequence[Gate]) -> str:
    failed = [g.explanation for g in gates if not g.passed]
    return "\n\nThe previous attempt failed verification: " + " ".join(failed) if failed else ""


def run_controlled_analysis(
    question: str,
    ticker: str,
    current_revenue: float,
    current_operating_income: float,
    prior_revenue: float,
    prior_operating_income: float,
    retrieve_fn: RetrieveFn,
    embedder: EmbedderFn,
    cross_encoder: CrossEncoderFn,
    completion_fn: CompletionFn | None = None,
    max_iterations: int = MAX_ITERATIONS,
) -> AnalysisResult:
    query_analysis = analyze_query(question)
    ledger = compute_margin_ledger(current_revenue, current_operating_income, prior_revenue, prior_operating_income)
    current_margin = ledger[0].result
    prior_margin = ledger[1].result
    movement = ledger[2].result
    ledger_summary = _ledger_summary(ledger)

    trace: list[IterationRecord] = []
    passages = list(retrieve_fn(False))
    result: ReasoningResult | None = None
    gates: list[Gate] = []
    feedback = ""
    expanded = False

    for i in range(1, max_iterations + 1):
        result = reason(question + feedback, ticker, passages, ledger_summary, completion_fn)
        gates = run_all_gates(ledger, result, passages, cross_encoder, current_margin, prior_margin, movement)
        score = sum(1 for g in gates if g.passed) / len(gates)
        action = choose_action(gates)
        trace.append(IterationRecord(iteration=i, action=action, gates=gates, verifier_score=score))

        if action == "STOP" or i == max_iterations:
            break
        if action == "REVISE":
            feedback = _feedback_note(gates)
        elif action == "RETRIEVE" and not expanded:
            expanded = True
            passages = list(retrieve_fn(True))
        elif action == "RECALCULATE":
            # Ledger is already deterministic/independent in this design, so a
            # recompute is a no-op here -- included so the branch is still
            # observable in the trace for parity with the app's controller.
            ledger = compute_margin_ledger(current_revenue, current_operating_income, prior_revenue, prior_operating_income)

    all_pass = all(g.passed for g in gates)
    return AnalysisResult(
        question=question,
        ticker=ticker,
        query_analysis=query_analysis,
        ledger=ledger,
        conclusion=result.conclusion if result else "",
        claims=result.claims if result else [],
        gates=gates,
        release_status="VERIFIED" if all_pass else "NEEDS_REVIEW",
        confidence=confidence_from_trace(trace, max_iterations),
        trace=trace,
        compute_saved_percent=compute_saved_percent(trace, max_iterations),
    )


def run_fully_automated_analysis(
    question: str,
    ticker: str,
    user_agent: str,
    embedder: EmbedderFn,
    cross_encoder: CrossEncoderFn,
    completion_fn: CompletionFn | None = None,
    max_iterations: int = MAX_ITERATIONS,
) -> AnalysisResult:
    """The fully automated entry point your architecture actually described:
    just a ticker and a free-text question. No manually typed revenue,
    operating income, or evidence excerpt anywhere.

    This is what neither the deployed Manus app nor earlier versions of this
    pipeline could do -- both still required numbers to be typed in by hand
    even in "auto"/retrieval-ready mode. Here, `xbrl.fetch_annual_figures`
    supplies the numbers and `retrieval.fetch_filing_html` + `chunk_filing`
    + `rank_passages_hybrid` supply the evidence, both live from SEC EDGAR.
    """
    from .retrieval import chunk_filing, fetch_filing_html, rank_passages_hybrid
    from .xbrl import fetch_annual_figures

    figures = fetch_annual_figures(ticker, user_agent)
    filing_html, identity = fetch_filing_html(ticker, user_agent)
    all_passages = chunk_filing(filing_html)

    def retrieve_fn(expanded: bool):
        query = f"{question} (expanded search)" if expanded else question
        ranked = rank_passages_hybrid(all_passages, query, embedder, cross_encoder, top_k=5 if expanded else 3)
        return [Passage(id=r.id, excerpt=r.excerpt) for r in ranked]

    return run_controlled_analysis(
        question=question,
        ticker=ticker,
        current_revenue=figures["current_revenue"],
        current_operating_income=figures["current_operating_income"],
        prior_revenue=figures["prior_revenue"],
        prior_operating_income=figures["prior_operating_income"],
        retrieve_fn=retrieve_fn,
        embedder=embedder,
        cross_encoder=cross_encoder,
        completion_fn=completion_fn,
        max_iterations=max_iterations,
    )
