"""
End-to-end proof that the controller genuinely branches, using scripted fake
LLM responses in place of a real (paid, network-dependent) reasoner call.
This is the test the deployed Manus app has no equivalent of: it demonstrates
STOP, REVISE, and RETRIEVE are all reachable outcomes of the *same* pipeline
code, driven only by what the (fake) reasoner and (real) verifier logic
produce -- not hardcoded per scenario.
"""
import json

from tao_fin.pipeline import run_controlled_analysis
from tao_fin.retrieval import Passage

REAL_PASSAGE = Passage(
    id="p1",
    excerpt="Total net sales were $416,161 million for fiscal 2025, compared with $391,035 million for "
    "fiscal 2024. Operating income was $133,050 million, compared with $123,216 million. Gross margin "
    "improved due to favorable product mix and lower component costs, partially offset by higher operating expenses.",
)


def _retrieve_fn(expanded_result=None):
    def retrieve(expanded: bool):
        if expanded and expanded_result is not None:
            return expanded_result
        return [REAL_PASSAGE]

    return retrieve


def test_clean_case_stops_on_first_iteration(fake_embedder, fake_cross_encoder, make_scripted_completion_fn):
    grounded_response = json.dumps({
        "conclusion": "Operating margin expanded due to favorable product mix and lower component costs.",
        "claims": [{"text": "favorable product mix and lower component costs improved margin", "supporting_passage_id": "p1"}],
    })
    result = run_controlled_analysis(
        question="Why did operating margin change in FY2025 vs FY2024?",
        ticker="AAPL",
        current_revenue=416161, current_operating_income=133050,
        prior_revenue=391035, prior_operating_income=123216,
        retrieve_fn=_retrieve_fn(),
        embedder=fake_embedder, cross_encoder=fake_cross_encoder,
        completion_fn=make_scripted_completion_fn([grounded_response]),
    )
    assert result.trace[0].action == "STOP"
    assert len(result.trace) == 1
    assert result.release_status == "VERIFIED"
    assert result.compute_saved_percent > 0


def test_ungrounded_causal_claim_triggers_retrieve_then_can_resolve(fake_embedder, fake_cross_encoder, make_scripted_completion_fn):
    ungrounded = json.dumps({
        "conclusion": "Operating margin expanded due to a new government manufacturing subsidy.",
        "claims": [{"text": "a new government manufacturing subsidy expanded margin", "supporting_passage_id": "p1"}],
    })
    grounded = json.dumps({
        "conclusion": "Operating margin expanded due to favorable product mix and lower component costs.",
        "claims": [{"text": "favorable product mix and lower component costs improved margin", "supporting_passage_id": "p1"}],
    })
    expanded_passage = Passage(
        id="p2",
        excerpt="Management discussion notes the margin improvement was driven by favorable product mix "
        "and lower component costs, not by any government subsidy.",
    )
    result = run_controlled_analysis(
        question="Why did operating margin change in FY2025 vs FY2024?",
        ticker="AAPL",
        current_revenue=416161, current_operating_income=133050,
        prior_revenue=391035, prior_operating_income=123216,
        retrieve_fn=_retrieve_fn(expanded_result=[REAL_PASSAGE, expanded_passage]),
        embedder=fake_embedder, cross_encoder=fake_cross_encoder,
        completion_fn=make_scripted_completion_fn([ungrounded, grounded]),
    )
    assert result.trace[0].action == "RETRIEVE"
    assert len(result.trace) >= 2


def test_direction_contradiction_triggers_revise(fake_embedder, fake_cross_encoder, make_scripted_completion_fn):
    contradictory = json.dumps({
        "conclusion": "Operating margin declined sharply due to favorable product mix.",
        "claims": [{"text": "favorable product mix", "supporting_passage_id": "p1"}],
    })
    corrected = json.dumps({
        "conclusion": "Operating margin expanded due to favorable product mix and lower component costs.",
        "claims": [{"text": "favorable product mix and lower component costs improved margin", "supporting_passage_id": "p1"}],
    })
    result = run_controlled_analysis(
        question="Why did operating margin change in FY2025 vs FY2024?",
        ticker="AAPL",
        current_revenue=416161, current_operating_income=133050,
        prior_revenue=391035, prior_operating_income=123216,
        retrieve_fn=_retrieve_fn(),
        embedder=fake_embedder, cross_encoder=fake_cross_encoder,
        completion_fn=make_scripted_completion_fn([contradictory, corrected]),
    )
    assert result.trace[0].action == "REVISE"


def test_iteration_cap_is_respected(fake_embedder, fake_cross_encoder, make_scripted_completion_fn):
    always_ungrounded = json.dumps({
        "conclusion": "Operating margin expanded due to an unexplained one-time gain.",
        "claims": [{"text": "an unexplained one-time gain expanded margin", "supporting_passage_id": "p1"}],
    })
    result = run_controlled_analysis(
        question="Why did operating margin change in FY2025 vs FY2024?",
        ticker="AAPL",
        current_revenue=416161, current_operating_income=133050,
        prior_revenue=391035, prior_operating_income=123216,
        retrieve_fn=_retrieve_fn(expanded_result=[REAL_PASSAGE]),  # never actually resolves
        embedder=fake_embedder, cross_encoder=fake_cross_encoder,
        completion_fn=make_scripted_completion_fn([always_ungrounded]),
    )
    assert len(result.trace) == 3
    assert result.release_status == "NEEDS_REVIEW"
    assert result.compute_saved_percent == 0.0
