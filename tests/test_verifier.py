from tao_fin.calculator import compute_margin_ledger
from tao_fin.reasoner import Claim, ReasoningResult
from tao_fin.retrieval import Passage
from tao_fin.verifier import evidence_gate, logic_gate, run_all_gates, sanity_gate


PASSAGES = [
    Passage(id="p1", excerpt="Gross margin improved due to favorable product mix and lower component costs."),
]


def test_evidence_gate_passes_when_claim_matches_cited_passage(fake_cross_encoder):
    result = ReasoningResult(
        conclusion="Margin expanded due to favorable product mix.",
        claims=[Claim(text="favorable product mix improved margin", supporting_passage_id="p1")],
    )
    gate = evidence_gate(result, PASSAGES, fake_cross_encoder, threshold=0.1)
    assert gate.passed


def test_evidence_gate_fails_on_ungrounded_claim(fake_cross_encoder):
    result = ReasoningResult(
        conclusion="Margin expanded because of a new government subsidy.",
        claims=[Claim(text="a new government subsidy expanded margin", supporting_passage_id="p1")],
    )
    gate = evidence_gate(result, PASSAGES, fake_cross_encoder, threshold=0.1)
    assert not gate.passed


def test_evidence_gate_fails_when_no_passage_cited(fake_cross_encoder):
    result = ReasoningResult(conclusion="Margin expanded.", claims=[Claim(text="Margin expanded", supporting_passage_id=None)])
    gate = evidence_gate(result, PASSAGES, fake_cross_encoder, threshold=0.1)
    assert not gate.passed


def test_logic_gate_catches_direction_contradiction():
    gate = logic_gate(movement=0.46, conclusion_text="Operating margin declined sharply this year.")
    assert not gate.passed


def test_logic_gate_passes_when_direction_matches():
    gate = logic_gate(movement=0.46, conclusion_text="Operating margin expanded modestly this year.")
    assert gate.passed


def test_sanity_gate_flags_implausible_margin():
    assert not sanity_gate(current_margin=150.0, prior_margin=31.5).passed
    assert sanity_gate(current_margin=31.97, prior_margin=31.51).passed


def test_run_all_gates_four_gates_present(fake_cross_encoder):
    ledger = compute_margin_ledger(416161, 133050, 391035, 123216)
    result = ReasoningResult(
        conclusion="Operating margin expanded due to favorable product mix.",
        claims=[Claim(text="favorable product mix improved margin", supporting_passage_id="p1")],
    )
    gates = run_all_gates(ledger, result, PASSAGES, fake_cross_encoder, 31.97, 31.51, 0.46)
    assert {g.id for g in gates} == {"calculations", "evidence", "logic", "sanity"}
