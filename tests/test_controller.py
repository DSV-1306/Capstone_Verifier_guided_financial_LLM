from tao_fin.controller import choose_action, compute_saved_percent, confidence_from_trace, IterationRecord
from tao_fin.verifier import Gate


def gates(**overrides):
    base = {"calculations": True, "evidence": True, "logic": True, "sanity": True}
    base.update(overrides)
    return [Gate(k, k, v, "") for k, v in base.items()]


def test_stop_when_all_gates_pass():
    assert choose_action(gates()) == "STOP"


def test_revise_when_logic_fails():
    assert choose_action(gates(logic=False)) == "REVISE"


def test_revise_when_sanity_fails():
    assert choose_action(gates(sanity=False)) == "REVISE"


def test_retrieve_when_only_evidence_fails():
    assert choose_action(gates(evidence=False)) == "RETRIEVE"


def test_recalculate_when_only_calculations_fail():
    assert choose_action(gates(calculations=False)) == "RECALCULATE"


def test_revise_takes_priority_over_retrieve():
    """If logic AND evidence both fail, REVISE should win -- an incoherent
    conclusion needs fixing before more evidence would even help."""
    assert choose_action(gates(logic=False, evidence=False)) == "REVISE"


def test_confidence_is_not_flat_across_paths():
    clean = [IterationRecord(1, "STOP", gates(), 1.0)]
    grinding = [
        IterationRecord(1, "RETRIEVE", gates(evidence=False), 0.75),
        IterationRecord(2, "RETRIEVE", gates(evidence=False), 0.75),
        IterationRecord(3, "RETRIEVE", gates(evidence=False), 0.75),
    ]
    clean_confidence = confidence_from_trace(clean)
    grinding_confidence = confidence_from_trace(grinding)
    assert clean_confidence > grinding_confidence
    assert clean_confidence != grinding_confidence  # explicitly not a flat number


def test_compute_saved_percent_is_zero_when_cap_is_hit():
    trace = [IterationRecord(i, "RETRIEVE", gates(evidence=False), 0.75) for i in range(1, 4)]
    assert compute_saved_percent(trace) == 0.0


def test_compute_saved_percent_is_positive_on_early_stop():
    trace = [IterationRecord(1, "STOP", gates(), 1.0)]
    assert compute_saved_percent(trace) > 0.0
