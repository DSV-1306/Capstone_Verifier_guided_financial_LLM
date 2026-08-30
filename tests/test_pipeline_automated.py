"""
Proves run_fully_automated_analysis actually wires XBRL numbers + live-style
retrieval + reasoner + verifier + controller together into one ticker+question
call -- by monkeypatching only the two network-touching functions
(fetch_annual_figures, fetch_filing_html), not the pipeline logic itself.
"""
import json

import tao_fin.retrieval as retrieval_module
import tao_fin.xbrl as xbrl_module
from tao_fin.pipeline import run_fully_automated_analysis

FAKE_FILING_HTML = """
<html><body>
<div><p>Total net sales were $416,161 million for fiscal 2025, compared with $391,035 million for
fiscal 2024. Operating income was $133,050 million, compared with $123,216 million. Gross margin
improved due to favorable product mix and lower component costs, partially offset by higher operating expenses.</p></div>
<div><p>The board also approved routine changes to the audit committee charter during fiscal 2025.</p></div>
</body></html>
"""


def test_fully_automated_analysis_uses_xbrl_numbers_and_live_filing_text(monkeypatch, fake_embedder, fake_cross_encoder, make_scripted_completion_fn):
    monkeypatch.setattr(
        xbrl_module, "fetch_annual_figures",
        lambda ticker, user_agent: {
            "current_revenue": 416161, "prior_revenue": 391035,
            "current_operating_income": 133050, "prior_operating_income": 123216,
            "current_fiscal_year": 2025, "prior_fiscal_year": 2024,
        },
    )
    monkeypatch.setattr(
        retrieval_module, "fetch_filing_html",
        lambda ticker, user_agent: (FAKE_FILING_HTML, {"ticker": ticker, "form": "10-K", "url": "https://sec.gov/fake"}),
    )
    grounded_response = json.dumps({
        "conclusion": "Operating margin expanded due to favorable product mix and lower component costs.",
        "claims": [{"text": "favorable product mix and lower component costs improved margin", "supporting_passage_id": "p1"}],
    })

    result = run_fully_automated_analysis(
        question="Why did operating margin change in FY2025 vs FY2024?",
        ticker="AAPL",
        user_agent="Test test@example.com",
        embedder=fake_embedder,
        cross_encoder=fake_cross_encoder,
        completion_fn=make_scripted_completion_fn([grounded_response]),
    )

    # Numbers came from (mocked) XBRL, not typed manually -- and they're right.
    assert result.ledger[0].result == 31.97
    assert result.ledger[2].result == 0.46
    # Evidence came from (mocked) live filing chunking + hybrid ranking, not
    # a hand-pasted excerpt -- the audit-committee paragraph should not surface.
    assert "audit committee" not in result.conclusion.lower()
