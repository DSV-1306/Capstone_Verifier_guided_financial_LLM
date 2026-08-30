from tao_fin.calculator import compute_margin_ledger, margins_are_plausible


def test_apple_fy2025_matches_published_figures():
    """Sanity-checks the calculator against the real, verified Apple FY2025
    10-K figures (416,161 / 133,050 current; 391,035 / 123,216 prior) --
    the same figures shown supported in the deployed app's screenshots."""
    ledger = compute_margin_ledger(416161, 133050, 391035, 123216)
    current, prior, movement = ledger
    assert current.result == 31.97
    assert prior.result == 31.51
    assert movement.result == 0.46
    assert current.status == "NOT_PROVIDED"  # no stated value supplied to compare


def test_mismatch_is_flagged_not_silently_trusted():
    ledger = compute_margin_ledger(
        416161, 133050, 391035, 123216, stated_current_margin=50.0
    )
    assert ledger[0].status == "MISMATCH"


def test_margin_plausibility():
    assert margins_are_plausible(31.97, 31.51)
    assert not margins_are_plausible(150.0, 20.0)
