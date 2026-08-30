"""
Tests the XBRL parsing/selection logic against a hand-built companyfacts
payload shaped like SEC EDGAR's real response -- so the concept-fallback and
full-fiscal-year filtering logic is verified without needing live network
access to data.sec.gov (unavailable in this sandbox; see tao_fin/xbrl.py).
"""
from tao_fin.xbrl import _first_available, _latest_two_annual_facts

# Mirrors the real shape of data.sec.gov/api/xbrl/companyfacts/CIK*.json,
# trimmed to just what the parser reads. Includes a quarterly (Q3, ~92 day)
# datapoint mixed in with annual ones -- exactly the kind of noise that
# makes naive "take the last N entries" parsing unsafe.
REVENUES_UNITS = {
    "USD": [
        {"start": "2023-09-25", "end": "2024-09-28", "val": 391035000000, "fy": 2024, "fp": "FY", "form": "10-K", "filed": "2024-11-01"},
        {"start": "2024-07-01", "end": "2024-09-28", "val": 94930000000, "fy": 2024, "fp": "Q4", "form": "10-Q", "filed": "2024-08-01"},
        {"start": "2024-09-29", "end": "2025-09-27", "val": 416161000000, "fy": 2025, "fp": "FY", "form": "10-K", "filed": "2025-11-01"},
    ]
}


def test_latest_two_annual_facts_ignores_quarterly_noise():
    facts = _latest_two_annual_facts(REVENUES_UNITS)
    assert len(facts) == 2
    assert facts[0].fiscal_year == 2024
    assert facts[0].value == 391035000000
    assert facts[1].fiscal_year == 2025
    assert facts[1].value == 416161000000


def test_first_available_falls_back_through_concept_aliases():
    us_gaap = {"SalesRevenueNet": {"units": REVENUES_UNITS}}
    result = _first_available(us_gaap, ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"))
    assert result is not None
    assert result[-1].value == 416161000000


def test_first_available_returns_none_when_no_concept_matches():
    result = _first_available({}, ("Revenues",))
    assert result is None
