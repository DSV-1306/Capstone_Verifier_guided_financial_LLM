"""
Automatic numeric retrieval from SEC EDGAR's XBRL company-facts API.

This closes a gap that existed in BOTH the deployed Manus app and earlier
versions of this pipeline: even in "auto"/retrieval-ready mode, revenue and
operating income still had to be typed in by hand. Your original
architecture listed "XBRL facts (SEC data.sec.gov, free)" as its own data
source specifically so numbers wouldn't need manual entry -- this module is
that piece.

Same network caveat as retrieval.py: this needs to run somewhere with real
internet access to data.sec.gov, not this sandbox.
"""
from __future__ import annotations

from dataclasses import dataclass

import requests

from .retrieval import SEC_DATA

# GAAP concepts tried in order -- companies don't all tag the same concept,
# so fall back through the common alternatives rather than failing outright.
REVENUE_CONCEPTS = (
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet",
)
OPERATING_INCOME_CONCEPTS = ("OperatingIncomeLoss",)


@dataclass
class AnnualFact:
    fiscal_year: int
    value: float
    filed: str  # filing date the fact was reported in, for traceability


def _cik_for_ticker(ticker: str, user_agent: str) -> str:
    headers = {"User-Agent": user_agent}
    index = requests.get(f"{SEC_DATA}/files/company_tickers.json", headers=headers, timeout=20)
    index.raise_for_status()
    entry = next((v for v in index.json().values() if v["ticker"].upper() == ticker.upper()), None)
    if entry is None:
        raise ValueError(f"No SEC registrant found for ticker {ticker!r}.")
    return str(entry["cik_str"]).zfill(10)


def _latest_two_annual_facts(concept_units: dict, form_filter: tuple[str, ...] = ("10-K",)) -> list[AnnualFact]:
    """Pick full-fiscal-year (~365 day) datapoints from 10-K filings only --
    XBRL facts mix annual, quarterly, and YTD figures under one concept, and
    only full-year 10-K entries are comparable to each other."""
    usd_facts = concept_units.get("USD", [])
    annual = [
        f for f in usd_facts
        if f.get("form") in form_filter
        and f.get("fp") == "FY"
        and f.get("start") and f.get("end")
        and 350 <= (_days(f["end"]) - _days(f["start"])) <= 380
    ]
    by_year = {f["fy"]: f for f in sorted(annual, key=lambda f: f["end"])}
    latest_years = sorted(by_year.keys())[-2:]
    return [
        AnnualFact(fiscal_year=y, value=float(by_year[y]["val"]), filed=by_year[y]["filed"])
        for y in latest_years
    ]


def _days(date_str: str) -> int:
    from datetime import date
    y, m, d = (int(x) for x in date_str.split("-"))
    return date(y, m, d).toordinal()


def fetch_annual_figures(ticker: str, user_agent: str) -> dict:
    """Returns the two most recent fiscal years' revenue and operating
    income, auto-pulled from SEC EDGAR -- no manual number entry required.

    Return shape:
        {"current_revenue": ..., "prior_revenue": ...,
         "current_operating_income": ..., "prior_operating_income": ...,
         "current_fiscal_year": ..., "prior_fiscal_year": ...}
    """
    cik = _cik_for_ticker(ticker, user_agent)
    response = requests.get(
        f"{SEC_DATA}/api/xbrl/companyfacts/CIK{cik}.json",
        headers={"User-Agent": user_agent}, timeout=30,
    )
    response.raise_for_status()
    facts = response.json().get("facts", {}).get("us-gaap", {})

    revenue = _first_available(facts, REVENUE_CONCEPTS)
    operating_income = _first_available(facts, OPERATING_INCOME_CONCEPTS)
    if revenue is None or operating_income is None:
        raise ValueError(
            f"Could not locate both revenue and operating-income XBRL facts for {ticker!r}. "
            "This company may tag its filings with non-standard GAAP concepts -- "
            "fall back to manual entry for this ticker."
        )
    if len(revenue) < 2 or len(operating_income) < 2:
        raise ValueError(f"Fewer than 2 full fiscal years of 10-K data found for {ticker!r}.")

    return {
        "current_revenue": revenue[-1].value, "prior_revenue": revenue[-2].value,
        "current_operating_income": operating_income[-1].value, "prior_operating_income": operating_income[-2].value,
        "current_fiscal_year": revenue[-1].fiscal_year, "prior_fiscal_year": revenue[-2].fiscal_year,
    }


def _first_available(facts: dict, concepts: tuple[str, ...]) -> list[AnnualFact] | None:
    for concept in concepts:
        if concept in facts:
            annual = _latest_two_annual_facts(facts[concept].get("units", {}))
            if len(annual) >= 2:
                return annual
    return None
