from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Sequence

import httpx

from ai_trader.config import AppSettings, get_settings
from ai_trader.domain.events import ThirteenFHolding


class SECEdgarProvider:
    def __init__(self, *, settings: AppSettings | None = None, http_client: httpx.Client | None = None) -> None:
        self._settings = settings or get_settings()
        self._client = http_client or httpx.Client(timeout=30)

    def fetch_holdings(
        self,
        manager_ciks: Sequence[str],
        filed_after: date,
        filed_before: date,
    ) -> Sequence[ThirteenFHolding]:
        holdings: list[ThirteenFHolding] = []
        for cik in manager_ciks:
            submissions = self._submissions(cik)
            recent = submissions.get("filings", {}).get("recent", {})
            forms = recent.get("form", [])
            filing_dates = recent.get("filingDate", [])
            acceptance_datetimes = recent.get("acceptanceDateTime", [])
            accession_numbers = recent.get("accessionNumber", [])
            primary_documents = recent.get("primaryDocument", [])
            for idx, form in enumerate(forms):
                if form != "13F-HR":
                    continue
                filing_date = _parse_date(acceptance_datetimes[idx] if idx < len(acceptance_datetimes) else "")
                if filing_date is None or filing_date < filed_after or filing_date > filed_before:
                    continue
                holdings.append(
                    ThirteenFHolding(
                        manager_name=submissions.get("name", f"CIK {cik}"),
                        cik=str(cik).zfill(10),
                        accession_number=accession_numbers[idx] if idx < len(accession_numbers) else None,
                        filing_date=filing_date,
                        report_period=_parse_date(filing_dates[idx]) or filing_date,
                        ticker="UNKNOWN",
                        issuer="13F filing metadata record",
                        market_value_usd=Decimal("0"),
                        shares=Decimal("0"),
                        source_url=self._archive_url(cik, accession_numbers, primary_documents, idx),
                    )
                )
        return tuple(holdings)

    def _submissions(self, cik: str) -> dict:
        response = self._client.get(
            f"https://data.sec.gov/submissions/CIK{str(cik).zfill(10)}.json",
            headers={"User-Agent": self._settings.sec_edgar_user_agent},
        )
        response.raise_for_status()
        return response.json()

    def _archive_url(self, cik: str, accessions: list[str], documents: list[str], idx: int) -> str | None:
        if idx >= len(accessions) or idx >= len(documents):
            return None
        accession = accessions[idx].replace("-", "")
        return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{documents[idx]}"


def _parse_date(value: str) -> date | None:
    if not value:
        return None
    return date.fromisoformat(value[:10])
