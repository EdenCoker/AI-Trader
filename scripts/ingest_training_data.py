"""
Ingest all available training data into LocalTrainingExample JSONL.

Sources ingested:
  - Training Data/congress-trading-all.xlsx   (insider/congressional trades)
  - Training Data/contracts-recent.xlsx        (government contracts per ticker)
  - Training Data/fear-and-greed.xlsx          (market sentiment index)
  - Training Data/lobbying-recent.xlsx         (lobbying activity per ticker)
  - Training Data/wsb-all.xlsx                 (WSB retail sentiment)
  - Training Data/patents-recent.xlsx          (patent filings per ticker)
  - Quiver API: live congressional trades      (if available)
  - Polygon API: historical daily OHLCV        (price returns per ticker/date)
  - RSS news feeds: Reuters, Bloomberg, FT     (headlines → narrative signal)

Each training example is a (SignalBundle, TradePlan, pnl_pct) triple
where pnl_pct is the actual forward 30-day price return from Polygon.

Usage:
    python scripts/ingest_training_data.py --out training_examples.jsonl
    python scripts/ingest_training_data.py --out training_examples.jsonl --tickers MSFT AAPL NVDA
"""
from __future__ import annotations

import argparse
import datetime
import logging
import math
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx
import pandas as pd

# Ensure src/ is on the path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ai_trader.config import get_settings
from ai_trader.domain.signals import Signal, SignalBundle, SignalDirection
from ai_trader.intelligence.trade_plan import TradePlan, horizon_class_for_days
from ai_trader.training.data import LocalTrainingExample

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("ingest")

TRAINING_DATA = Path("Training Data")
POLYGON_BASE = "https://api.polygon.io/v2/aggs/ticker"
SEC_FULL_TEXT_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_COMPANY_TICKERS_EXCHANGE_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
FINRA_SHORT_INTEREST_URL = "https://api.finra.org/data/group/otcmarket/name/consolidatedShortInterest"
FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"
RSS_FEEDS = [
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.bbci.co.uk/news/business/rss.xml",
    "https://rss.ft.com/rss/companies",
]
HOLDING_DAYS = 30  # forward return window for pnl_pct
CYCLICAL_TICKERS = {
    "AA", "AAL", "BA", "CAT", "CCL", "DAL", "DE", "DIS", "F", "FDX", "GM",
    "HD", "JPM", "LUV", "MAR", "NCLH", "RCL", "UAL", "UPS", "WYNN",
}
INDUSTRIAL_TICKERS = {
    "BA", "CAT", "DE", "DOV", "EMR", "ETN", "FDX", "GE", "HON", "MMM",
    "NOC", "RTX", "UNP", "UPS",
}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

XL = {"engine": "openpyxl"}  # xlsx reader (openpyxl is always available)


def load_fear_greed() -> pd.DataFrame:
    """Return DataFrame indexed by date with column 'fear_greed'."""
    log.info("  loading fear-and-greed.xlsx …")
    df = pd.read_excel(TRAINING_DATA / "fear-and-greed.xlsx", parse_dates=["Date"], **XL)
    df = df.rename(columns={"Date": "date", "Index": "fear_greed"})
    df["date"] = df["date"].dt.date
    return df.set_index("date").sort_index()


def load_wsb() -> pd.DataFrame:
    """Return DataFrame with columns: ticker, date, wsb_sentiment, wsb_mentions."""
    log.info("  loading wsb-all.xlsx …")
    df = pd.read_excel(TRAINING_DATA / "wsb-all.xlsx", parse_dates=["Datetime"], **XL)
    df = df.rename(columns={
        "Ticker": "ticker",
        "Datetime": "date",
        "Sentiment": "wsb_sentiment",
        "Mentions": "wsb_mentions",
    })
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["ticker"] = df["ticker"].str.upper()
    return df[["ticker", "date", "wsb_sentiment", "wsb_mentions"]].drop_duplicates(["ticker", "date"])


def load_congress() -> pd.DataFrame:
    """Return DataFrame: ticker, date, congress_buy, congress_sell, congress_amount."""
    path = TRAINING_DATA / "congress-trading-all.xlsx"
    log.info("  loading congress-trading-all.xlsx …")
    df = pd.read_excel(path, **XL)
    # File may be an API error blob — check for expected columns
    if "Ticker" not in df.columns and "ticker" not in df.columns:
        log.warning("congress-trading-all.xlsx appears empty or error page — fetching from Quiver API instead")
        return _fetch_congress_from_api()
    df = df.rename(columns=lambda c: c.strip())
    ticker_col = "Ticker" if "Ticker" in df.columns else "ticker"
    date_col = next((c for c in df.columns if "date" in c.lower() or "filed" in c.lower()), None)
    if date_col is None:
        return pd.DataFrame(columns=["ticker", "date", "congress_buy", "congress_sell", "congress_amount"])
    df["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.date
    df["ticker"] = df[ticker_col].astype(str).str.upper()
    trans_col = next((c for c in df.columns if "transaction" in c.lower() or "type" in c.lower()), None)
    amt_col = next((c for c in df.columns if "amount" in c.lower() or "range" in c.lower()), None)
    df["congress_buy"] = 0
    df["congress_sell"] = 0
    df["congress_amount"] = 0.0
    if trans_col:
        df["congress_buy"] = df[trans_col].astype(str).str.lower().str.contains("purchase|buy").astype(int)
        df["congress_sell"] = df[trans_col].astype(str).str.lower().str.contains("sale|sell").astype(int)
    if amt_col:
        df["congress_amount"] = pd.to_numeric(df[amt_col], errors="coerce").fillna(0)
    out = df.groupby(["ticker", "date"]).agg(
        congress_buy=("congress_buy", "sum"),
        congress_sell=("congress_sell", "sum"),
        congress_amount=("congress_amount", "sum"),
    ).reset_index()
    return out


def _quiver_paginate(url: str, headers: dict, params: dict | None = None, page_size: int = 500, max_pages: int = 40) -> list[dict]:
    """Paginate through a Quiver endpoint, returning all records (capped at max_pages)."""
    all_rows: list[dict] = []
    page = 1
    while True:
        if page > max_pages:
            log.info("  Quiver page cap (%d) reached for %s — stopping", max_pages, url)
            break
        p = {**(params or {}), "page": page, "page_size": page_size}
        try:
            r = httpx.get(url, headers=headers, params=p, timeout=60)
            if r.status_code == 404:
                log.warning("Quiver 404 at %s (page %d) — endpoint may not be available on your tier", url, page)
                break
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break
            all_rows.extend(batch if isinstance(batch, list) else [])
            log.info("  page %d: %d rows (total so far: %d)", page, len(batch), len(all_rows))
            if len(batch) < page_size:
                break  # last page
            page += 1
        except Exception as exc:
            log.warning("Quiver paginate error at %s page %d: %s", url, page, exc)
            break
    return all_rows


def _fetch_congress_from_api() -> pd.DataFrame:
    """Pull congressional trades from Quiver API with full pagination."""
    settings = get_settings()
    if settings.quiver_api_key is None:
        log.warning("No QUIVER_API_KEY — congressional trades unavailable")
        return pd.DataFrame(columns=["ticker", "date", "congress_buy", "congress_sell", "congress_amount"])
    headers = {"Authorization": f"Bearer {settings.quiver_api_key.get_secret_value()}"}
    log.info("  fetching congress trades from Quiver API (paginated) …")
    rows = _quiver_paginate("https://api.quiverquant.com/beta/bulk/congresstrading", headers)
    log.info("Quiver congress API: %d total rows fetched", len(rows))
    # Also pull house + senate live endpoints (most recent, Tier 1)
    for live_url, label in [
        ("https://api.quiverquant.com/beta/live/congresstrading", "live/congresstrading"),
        ("https://api.quiverquant.com/beta/live/housetrading", "live/housetrading"),
        ("https://api.quiverquant.com/beta/live/senatetrading", "live/senatetrading"),
    ]:
        try:
            r = httpx.get(live_url, headers=headers, timeout=30)
            r.raise_for_status()
            extra = r.json() if isinstance(r.json(), list) else []
            rows.extend(extra)
            log.info("Quiver %s: %d rows", label, len(extra))
        except Exception as exc:
            log.warning("Quiver %s failed: %s", label, exc)
    records = []
    for item in rows:
        ticker = str(item.get("Ticker") or item.get("ticker") or "").upper()
        raw_date = item.get("FiledAfterDate") or item.get("ReportDate") or item.get("FiledDate") or item.get("TransactionDate")
        if not ticker or not raw_date:
            continue
        try:
            d = datetime.date.fromisoformat(str(raw_date)[:10])
        except ValueError:
            continue
        tx = str(item.get("Transaction") or "").lower()
        amount = float(item.get("RangeHigh") or item.get("AmountHigh") or 0)
        records.append({
            "ticker": ticker,
            "date": d,
            "congress_buy": 1 if "purchase" in tx else 0,
            "congress_sell": 1 if "sale" in tx else 0,
            "congress_amount": amount,
        })
    if not records:
        return pd.DataFrame(columns=["ticker", "date", "congress_buy", "congress_sell", "congress_amount"])
    df = pd.DataFrame(records)
    return df.groupby(["ticker", "date"]).agg(
        congress_buy=("congress_buy", "sum"),
        congress_sell=("congress_sell", "sum"),
        congress_amount=("congress_amount", "sum"),
    ).reset_index()


def _fetch_lobbying_from_api() -> pd.DataFrame:
    """Pull all lobbying records from Quiver /beta/live/lobbying (paginated)."""
    settings = get_settings()
    if settings.quiver_api_key is None:
        return pd.DataFrame(columns=["ticker", "date", "lobby_amount"])
    headers = {"Authorization": f"Bearer {settings.quiver_api_key.get_secret_value()}"}
    log.info("  fetching live lobbying from Quiver API (paginated) …")
    rows = _quiver_paginate(
        "https://api.quiverquant.com/beta/live/lobbying",
        headers,
        params={"all": "true"},
    )
    log.info("Quiver live/lobbying: %d total rows fetched", len(rows))
    records = []
    for item in rows:
        ticker = str(item.get("Ticker") or item.get("ticker") or "").upper()
        raw_date = item.get("Date") or item.get("date") or item.get("Filed") or item.get("Period")
        amount = float(item.get("Amount") or item.get("amount") or 0)
        if not ticker or not raw_date:
            continue
        try:
            d = datetime.date.fromisoformat(str(raw_date)[:10])
        except ValueError:
            continue
        records.append({"ticker": ticker, "date": d, "lobby_amount": amount})
    if not records:
        return pd.DataFrame(columns=["ticker", "date", "lobby_amount"])
    df = pd.DataFrame(records)
    return df.groupby(["ticker", "date"]).agg(lobby_amount=("lobby_amount", "sum")).reset_index()


def _fetch_govcontracts_from_api() -> pd.DataFrame:
    """Pull gov contract records from Quiver /beta/live/govcontractsall (paginated)."""
    settings = get_settings()
    if settings.quiver_api_key is None:
        return pd.DataFrame(columns=["ticker", "date", "contract_amount"])
    headers = {"Authorization": f"Bearer {settings.quiver_api_key.get_secret_value()}"}
    log.info("  fetching live gov contracts from Quiver API (paginated) …")
    rows = _quiver_paginate("https://api.quiverquant.com/beta/live/govcontractsall", headers)
    log.info("Quiver live/govcontractsall: %d total rows fetched", len(rows))
    records = []
    for item in rows:
        ticker = str(item.get("Ticker") or item.get("ticker") or "").upper()
        raw_date = item.get("Date") or item.get("date") or item.get("SignedDate")
        amount = float(item.get("Amount") or item.get("amount") or 0)
        if not ticker or not raw_date:
            continue
        try:
            d = datetime.date.fromisoformat(str(raw_date)[:10])
        except ValueError:
            continue
        records.append({"ticker": ticker, "date": d, "contract_amount": amount})
    if not records:
        return pd.DataFrame(columns=["ticker", "date", "contract_amount"])
    df = pd.DataFrame(records)
    return df.groupby(["ticker", "date"]).agg(contract_amount=("contract_amount", "sum")).reset_index()


def load_lobbying() -> pd.DataFrame:
    """Return per-ticker/date lobbying amount — Excel file merged with live API data."""
    frames = []
    excel_path = TRAINING_DATA / "lobbying-recent.xlsx"
    if excel_path.exists():
        log.info("  loading lobbying-recent.xlsx …")
        df = pd.read_excel(excel_path, parse_dates=["Date"], **XL)
        df = df.rename(columns={"Ticker": "ticker", "Date": "date", "Amount": "lobby_amount"})
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["ticker"] = df["ticker"].astype(str).str.upper()
        frames.append(df[["ticker", "date", "lobby_amount"]])
    # Supplement with live API data
    api_df = _fetch_lobbying_from_api()
    if not api_df.empty:
        frames.append(api_df)
    if not frames:
        return pd.DataFrame(columns=["ticker", "date", "lobby_amount"])
    combined = pd.concat(frames, ignore_index=True)
    out = combined.groupby(["ticker", "date"]).agg(lobby_amount=("lobby_amount", "sum")).reset_index()
    log.info("  lobbying: %d ticker/date rows (Excel + API)", len(out))
    return out


def load_contracts() -> pd.DataFrame:
    """Return per-ticker/date government contract amount — Excel merged with live API data."""
    frames = []
    excel_path = TRAINING_DATA / "contracts-recent.xlsx"
    if excel_path.exists():
        log.info("  loading contracts-recent.xlsx …")
        df = pd.read_excel(excel_path, parse_dates=["Date"], **XL)
        df = df.rename(columns={"Ticker": "ticker", "Date": "date", "Amount": "contract_amount"})
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["ticker"] = df["ticker"].astype(str).str.upper()
        frames.append(df[["ticker", "date", "contract_amount"]])
    # Supplement with live API data
    api_df = _fetch_govcontracts_from_api()
    if not api_df.empty:
        frames.append(api_df)
    if not frames:
        return pd.DataFrame(columns=["ticker", "date", "contract_amount"])
    combined = pd.concat(frames, ignore_index=True)
    out = combined.groupby(["ticker", "date"]).agg(contract_amount=("contract_amount", "sum")).reset_index()
    log.info("  contracts: %d ticker/date rows (Excel + API)", len(out))
    return out


def load_patents() -> pd.DataFrame:
    """Return per-ticker/month patent count."""
    log.info("  loading patents-recent.xlsx (large file, may take a moment) …")
    df = pd.read_excel(TRAINING_DATA / "patents-recent.xlsx", **XL)
    ticker_col = "compu_ticker" if "compu_ticker" in df.columns else df.columns[0]
    date_col = "pubdate" if "pubdate" in df.columns else None
    if date_col is None:
        return pd.DataFrame(columns=["ticker", "date", "patent_count"])
    df["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.date
    df["ticker"] = df[ticker_col].astype(str).str.upper()
    out = df.groupby(["ticker", "date"]).size().reset_index(name="patent_count")
    return out


def load_ibkr_executions(settings) -> pd.DataFrame:
    """Pull all IBKR execution fills (read-only) and return per-ticker/date aggregates.

    Returns a DataFrame with columns: ticker, date, ibkr_buy, ibkr_sell, ibkr_qty, ibkr_price.
    Returns an empty DataFrame if TWS/IB Gateway is unreachable or ib_insync is not installed.
    """
    empty = pd.DataFrame(columns=["ticker", "date", "ibkr_buy", "ibkr_sell", "ibkr_qty", "ibkr_price"])
    try:
        from ib_insync import IB
    except ImportError:
        log.warning("ib_insync not installed — skipping IBKR executions (pip install ib_insync)")
        return empty

    ib = IB()
    try:
        ib.connect(
            host=settings.ibkr_host,
            port=settings.ibkr_port,
            clientId=settings.ibkr_client_id + 10,  # distinct clientId to avoid conflicts
            timeout=8.0,
            readonly=True,
            account=settings.ibkr_account or "",
        )
    except Exception as exc:
        log.warning("IBKR connect failed — skipping executions: %s", exc)
        return empty

    try:
        fills = ib.reqExecutions()
        log.info("IBKR: %d execution fills fetched", len(fills))
    except Exception as exc:
        log.warning("IBKR reqExecutions failed: %s", exc)
        fills = []
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass

    records = []
    for fill in fills:
        try:
            ticker = str(fill.contract.symbol).upper()
            # ib_insync time format: "20240115  09:30:00 US/Eastern"
            raw_time = str(fill.execution.time).strip()
            d = datetime.date(int(raw_time[:4]), int(raw_time[4:6]), int(raw_time[6:8]))
            side = str(fill.execution.side).upper()  # "BOT" or "SLD"
            qty = float(fill.execution.shares)
            price = float(fill.execution.price)
            records.append({
                "ticker": ticker,
                "date": d,
                "ibkr_buy": 1 if side == "BOT" else 0,
                "ibkr_sell": 1 if side == "SLD" else 0,
                "ibkr_qty": qty,
                "ibkr_price": price,
            })
        except Exception as exc:
            log.debug("Skipping fill record: %s", exc)

    if not records:
        log.info("No usable IBKR execution fills found")
        return empty

    df = pd.DataFrame(records)
    out = df.groupby(["ticker", "date"]).agg(
        ibkr_buy=("ibkr_buy", "sum"),
        ibkr_sell=("ibkr_sell", "sum"),
        ibkr_qty=("ibkr_qty", "sum"),
        ibkr_price=("ibkr_price", "mean"),
    ).reset_index()
    log.info("IBKR executions: %d ticker/date rows", len(out))
    return out


def load_insider_trades(
    tickers: list[str],
    start: datetime.date,
    end: datetime.date,
) -> pd.DataFrame:
    """Fetch Form 4 transactions from EDGAR and aggregate by ticker/disclosure date."""

    columns = [
        "ticker",
        "date",
        "insider_buy_qty",
        "insider_sell_qty",
        "insider_net_qty",
        "insider_value_usd",
        "insider_officer_count",
        "insider_director_count",
    ]
    settings = get_settings()
    if "set SEC_EDGAR_USER_AGENT" in settings.sec_edgar_user_agent:
        log.warning("SEC_EDGAR_USER_AGENT not configured -- skipping Form 4 insider trades")
        return pd.DataFrame(columns=columns)

    ticker_map = _sec_company_ticker_map(settings)
    rows: list[dict] = []
    for ticker in tickers:
        ticker = ticker.upper()
        cik = ticker_map.get(ticker)
        filings = _sec_full_text_form4_hits(ticker, start, end, settings)
        if not filings and cik:
            filings = _sec_submission_form4_hits(cik, start, end, settings)
        # Cap to most-recent 50 filings per ticker to avoid multi-hour SEC crawls
        for filing in filings[:50]:
            rows.extend(_parse_form4_transactions(ticker, filing, settings))
        if rows:
            log.info("  Form 4 %s: %d transactions so far", ticker, len(rows))
        time.sleep(0.1)

    if not rows:
        return pd.DataFrame(columns=columns)
    df = pd.DataFrame(rows)
    df = df[(df["date"] >= start) & (df["date"] <= end)]
    if df.empty:
        return pd.DataFrame(columns=columns)
    out = df.groupby(["ticker", "date"]).agg(
        insider_buy_qty=("buy_qty", "sum"),
        insider_sell_qty=("sell_qty", "sum"),
        insider_net_qty=("net_qty", "sum"),
        insider_value_usd=("value_usd", "sum"),
        insider_officer_count=("is_officer", "sum"),
        insider_director_count=("is_director", "sum"),
    ).reset_index()
    return out[columns]


def load_fred_macro(start: datetime.date, end: datetime.date) -> pd.DataFrame:
    """Return daily macro context from FRED, forward-filled across trading days."""

    columns = [
        "date",
        "yield_spread_2_10",
        "cpi_mom",
        "ism_pmi",
        "unemployment_claims",
        "fed_funds_rate",
    ]
    settings = get_settings()
    if settings.fred_api_key is None:
        log.warning("FRED_API_KEY not configured -- skipping FRED macro context")
        return pd.DataFrame(columns=columns)
    series = {
        "yield_spread_2_10": "T10Y2Y",
        "cpi": "CPIAUCSL",
        "ism_pmi": "NAPM",
        "unemployment_claims": "ICSA",
        "fed_funds_rate": "FEDFUNDS",
    }
    frames = []
    for output_name, series_id in series.items():
        values = _fetch_fred_series(series_id, start, end, settings.fred_api_key.get_secret_value())
        if values.empty:
            continue
        frames.append(values.rename(columns={"value": output_name}).set_index("date"))
    if not frames:
        return pd.DataFrame(columns=columns)

    macro = pd.concat(frames, axis=1).sort_index()
    if "cpi" in macro:
        macro["cpi_mom"] = macro["cpi"].pct_change() * 100.0
        macro = macro.drop(columns=["cpi"])
    else:
        macro["cpi_mom"] = 0.0
    full_index = pd.date_range(start=start, end=end, freq="D").date
    macro = macro.reindex(full_index).ffill()
    macro.index.name = "date"
    out = macro.reset_index()
    for column in columns:
        if column not in out:
            out[column] = 0.0
    return out[columns].fillna(0.0)


def load_earnings_surprises(
    tickers: list[str],
    start: datetime.date,
    end: datetime.date,
) -> pd.DataFrame:
    columns = ["ticker", "date", "eps_actual", "eps_estimate", "eps_surprise_pct"]
    settings = get_settings()
    base_url = settings.fmp_base_url.rstrip("/")
    rows = []
    for ticker in tickers:
        for endpoint in (
            f"{base_url}/historical/earning_calendar/{ticker.upper()}",
            f"{base_url}/earning_calendar",
        ):
            try:
                params = {"symbol": ticker.upper()} if endpoint.endswith("earning_calendar") else None
                response = httpx.get(endpoint, params=params, timeout=20)
                if response.status_code in {401, 402, 403, 404}:
                    continue
                response.raise_for_status()
                payload = response.json()
            except Exception as exc:
                log.debug("FMP earnings fetch failed for %s via %s: %s", ticker, endpoint, exc)
                continue
            items = payload if isinstance(payload, list) else payload.get("historical", [])
            for item in items:
                row = _parse_earnings_item(ticker, item)
                if row and start <= row["date"] <= end:
                    rows.append(row)
            if rows:
                break
        time.sleep(0.05)
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).drop_duplicates(["ticker", "date"])[columns]


def load_options_put_call_ratios(tickers: list[str], as_of: datetime.date) -> pd.DataFrame:
    columns = ["ticker", "date", "put_call_ratio", "put_open_interest", "call_open_interest"]
    settings = get_settings()
    if settings.polygon_api_key is None:
        return pd.DataFrame(columns=columns)
    rows = []
    for ticker in tickers:
        url = f"https://api.polygon.io/v3/snapshot/options/{ticker.upper()}"
        try:
            response = httpx.get(
                url,
                params={"limit": 250, "apiKey": settings.polygon_api_key.get_secret_value()},
                timeout=30,
            )
            if response.status_code in {403, 404}:
                continue
            response.raise_for_status()
            results = response.json().get("results", [])
        except Exception as exc:
            log.debug("Polygon options snapshot failed for %s: %s", ticker, exc)
            continue
        put_oi = 0.0
        call_oi = 0.0
        for item in results:
            details = item.get("details") or {}
            contract_type = str(details.get("contract_type") or "").casefold()
            open_interest = float(item.get("open_interest") or 0)
            if contract_type == "put":
                put_oi += open_interest
            elif contract_type == "call":
                call_oi += open_interest
        if put_oi > 0 or call_oi > 0:
            rows.append(
                {
                    "ticker": ticker.upper(),
                    "date": as_of,
                    "put_call_ratio": put_oi / max(call_oi, 1.0),
                    "put_open_interest": put_oi,
                    "call_open_interest": call_oi,
                }
            )
        time.sleep(0.05)
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)[columns]


def load_13f_changes(
    tickers: list[str],
    start: datetime.date,
    end: datetime.date,
) -> pd.DataFrame:
    """Compare consecutive 13F information tables for well-known managers."""

    columns = [
        "ticker",
        "date",
        "institutional_delta_shares",
        "institutional_delta_pct",
        "institutional_manager",
        "institutional_market_value_usd",
    ]
    settings = get_settings()
    if "set SEC_EDGAR_USER_AGENT" in settings.sec_edgar_user_agent:
        return pd.DataFrame(columns=columns)
    name_map = _sec_company_name_ticker_map(settings)
    ticker_filter = {ticker.upper() for ticker in tickers}
    manager_ciks = {
        "Berkshire Hathaway": "1067983",
        "Duquesne Family Office": "1536411",
        "Soros Fund Management": "1029160",
        "Renaissance Technologies": "1037389",
        "Citadel Advisors": "1423053",
    }
    rows = []
    for manager_name, cik in manager_ciks.items():
        filings = _sec_13f_filings(cik, start - datetime.timedelta(days=140), end, settings)
        parsed = []
        for filing in filings[:6]:
            holdings = _fetch_13f_info_table(cik, filing, settings, name_map)
            if holdings:
                parsed.append((filing["filing_date"], holdings))
            time.sleep(0.1)
        parsed.sort(key=lambda item: item[0])
        for idx in range(1, len(parsed)):
            filing_date, current = parsed[idx]
            previous = parsed[idx - 1][1]
            if not (start <= filing_date <= end):
                continue
            for ticker, holding in current.items():
                if ticker not in ticker_filter:
                    continue
                previous_shares = previous.get(ticker, {}).get("shares")
                delta = holding["shares"] - (previous_shares or 0.0)
                delta_pct = None if not previous_shares else delta / previous_shares
                if previous_shares is not None and delta_pct is not None and delta_pct < 0.20:
                    continue
                if delta <= 0:
                    continue
                rows.append(
                    {
                        "ticker": ticker,
                        "date": filing_date,
                        "institutional_delta_shares": delta,
                        "institutional_delta_pct": 1.0 if previous_shares is None else delta_pct,
                        "institutional_manager": manager_name,
                        "institutional_market_value_usd": holding["market_value_usd"],
                    }
                )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)[columns]


def load_short_interest(
    tickers: list[str],
    start: datetime.date,
    end: datetime.date,
) -> pd.DataFrame:
    columns = [
        "ticker",
        "date",
        "short_interest_shares",
        "days_to_cover",
        "short_interest_change_pct",
    ]
    rows = []
    for ticker in tickers:
        payload = {
            "limit": 200,
            "fields": [
                "symbolCode",
                "settlementDate",
                "currentShortPositionQuantity",
                "daysToCoverQuantity",
                "changePercent",
            ],
            "compareFilters": [
                {"compareType": "equal", "fieldName": "symbolCode", "fieldValue": ticker.upper()},
            ],
        }
        try:
            response = httpx.post(FINRA_SHORT_INTEREST_URL, json=payload, timeout=30)
            response.raise_for_status()
            items = response.json()
        except Exception as exc:
            log.debug("FINRA short interest failed for %s: %s", ticker, exc)
            continue
        for item in items if isinstance(items, list) else []:
            try:
                settlement_date = datetime.date.fromisoformat(str(item["settlementDate"])[:10])
            except Exception:
                continue
            if not (start <= settlement_date <= end):
                continue
            rows.append(
                {
                    "ticker": str(item.get("symbolCode") or ticker).upper(),
                    "date": settlement_date,
                    "short_interest_shares": float(item.get("currentShortPositionQuantity") or 0),
                    "days_to_cover": float(item.get("daysToCoverQuantity") or 0),
                    "short_interest_change_pct": float(item.get("changePercent") or 0),
                }
            )
        time.sleep(0.05)
    if not rows:
        return pd.DataFrame(columns=columns)
    df = pd.DataFrame(rows).sort_values(["ticker", "date"])
    return df.drop_duplicates(["ticker", "date"], keep="last")[columns]


def _sec_headers(settings) -> dict[str, str]:
    return {
        "User-Agent": settings.sec_edgar_user_agent,
        "Accept-Encoding": "gzip, deflate",
        "Host": "www.sec.gov",
    }


def _sec_company_ticker_map(settings) -> dict[str, str]:
    try:
        response = httpx.get(
            SEC_COMPANY_TICKERS_URL,
            headers=_sec_headers(settings),
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        log.warning("SEC company ticker map unavailable: %s", exc)
        return {}
    return {
        str(item.get("ticker") or "").upper(): str(item.get("cik_str") or "").zfill(10)
        for item in payload.values()
        if item.get("ticker") and item.get("cik_str")
    }


def _sec_company_name_ticker_map(settings) -> dict[str, str]:
    try:
        response = httpx.get(
            SEC_COMPANY_TICKERS_EXCHANGE_URL,
            headers=_sec_headers(settings),
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return {}
    fields = payload.get("fields", [])
    data = payload.get("data", [])
    try:
        ticker_idx = fields.index("ticker")
        name_idx = fields.index("name")
    except ValueError:
        return {}
    return {
        _normalize_company_name(str(row[name_idx])): str(row[ticker_idx]).upper()
        for row in data
        if len(row) > max(ticker_idx, name_idx)
    }


def _sec_full_text_form4_hits(ticker: str, start: datetime.date, end: datetime.date, settings) -> list[dict]:
    params = {
        "q": f'"{ticker.upper()}"',
        "forms": "4",
        "startdt": start.isoformat(),
        "enddt": end.isoformat(),
        "from": 0,
        "size": 40,
    }
    try:
        response = httpx.get(
            SEC_FULL_TEXT_SEARCH_URL,
            params=params,
            headers={"User-Agent": settings.sec_edgar_user_agent},
            timeout=30,
        )
        if response.status_code in {403, 404}:
            return []
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        log.debug("SEC full-text Form 4 search failed for %s: %s", ticker, exc)
        return []
    hits = payload.get("hits", {}).get("hits", payload.get("hits", []))
    filings = []
    for hit in hits if isinstance(hits, list) else []:
        source = hit.get("_source", hit)
        form = str(source.get("form") or source.get("file_type") or "")
        if form and form != "4":
            continue
        url = (
            source.get("adsh")
            or source.get("file_url")
            or source.get("document_url")
            or source.get("url")
        )
        filing_date = _safe_date(source.get("file_date") or source.get("filedAt") or source.get("filingDate"))
        if not filing_date or not url:
            continue
        filings.append({"filing_date": filing_date, "url": _sec_filing_url_from_hit(url)})
    return filings


def _sec_submission_form4_hits(
    cik: str,
    start: datetime.date,
    end: datetime.date,
    settings,
) -> list[dict]:
    try:
        response = httpx.get(
            f"https://data.sec.gov/submissions/CIK{str(cik).zfill(10)}.json",
            headers={"User-Agent": settings.sec_edgar_user_agent},
            timeout=30,
        )
        response.raise_for_status()
        submissions = response.json()
    except Exception as exc:
        log.debug("SEC submissions unavailable for CIK %s: %s", cik, exc)
        return []
    recent = submissions.get("filings", {}).get("recent", {})
    filings = []
    for idx, form in enumerate(recent.get("form", [])):
        if form != "4":
            continue
        filing_date = _safe_date(_at(recent.get("filingDate", []), idx))
        if not filing_date or filing_date < start or filing_date > end:
            continue
        accession = _at(recent.get("accessionNumber", []), idx)
        document = _at(recent.get("primaryDocument", []), idx)
        if accession and document:
            filings.append(
                {
                    "filing_date": filing_date,
                    "url": _sec_archive_url(cik, accession, document),
                }
            )
    return filings


def _parse_form4_transactions(ticker: str, filing: dict, settings) -> list[dict]:
    url = filing.get("url")
    if not url:
        return []
    try:
        response = httpx.get(url, headers={"User-Agent": settings.sec_edgar_user_agent}, timeout=30)
        response.raise_for_status()
        root = ET.fromstring(response.content)
    except Exception as exc:
        log.debug("Form 4 parse failed for %s: %s", url, exc)
        return []
    issuer_ticker = _xml_text(root, ".//issuerTradingSymbol") or ticker
    is_officer = _xml_text(root, ".//reportingOwnerRelationship/isOfficer") == "1"
    is_director = _xml_text(root, ".//reportingOwnerRelationship/isDirector") == "1"
    rows = []
    for node in root.findall(".//nonDerivativeTransaction"):
        code = (_xml_text(node, ".//transactionCoding/transactionCode") or "").upper()
        if code not in {"P", "S"}:
            continue
        tx_date = _safe_date(_xml_text(node, ".//transactionDate/value")) or filing["filing_date"]
        shares = _float(_xml_text(node, ".//transactionShares/value"))
        price = _float(_xml_text(node, ".//transactionPricePerShare/value"))
        shares = 0.0 if math.isnan(shares) else shares
        price = 0.0 if math.isnan(price) else price
        value = shares * price if shares > 0 and price > 0 else 0.0
        rows.append(
            {
                "ticker": issuer_ticker.upper(),
                "date": filing["filing_date"],
                "transaction_date": tx_date,
                "buy_qty": shares if code == "P" else 0.0,
                "sell_qty": shares if code == "S" else 0.0,
                "net_qty": shares if code == "P" else -shares,
                "value_usd": value,
                "is_officer": 1 if is_officer else 0,
                "is_director": 1 if is_director else 0,
            }
        )
    return rows


def _fetch_fred_series(series_id: str, start: datetime.date, end: datetime.date, api_key: str) -> pd.DataFrame:
    try:
        response = httpx.get(
            FRED_OBSERVATIONS_URL,
            params={
                "series_id": series_id,
                "observation_start": start.isoformat(),
                "observation_end": end.isoformat(),
                "api_key": api_key,
                "file_type": "json",
            },
            timeout=30,
        )
        response.raise_for_status()
    except Exception as exc:
        log.debug("FRED %s failed: %s", series_id, exc)
        return pd.DataFrame(columns=["date", "value"])
    rows = []
    for item in response.json().get("observations", []):
        value = _float(item.get("value"))
        if math.isnan(value):
            continue
        observed = _safe_date(item.get("date"))
        if observed:
            rows.append({"date": observed, "value": value})
    return pd.DataFrame(rows, columns=["date", "value"])


def _parse_earnings_item(ticker: str, item: dict) -> dict | None:
    observed = _safe_date(item.get("date") or item.get("fiscalDateEnding"))
    actual = _float(item.get("eps") or item.get("epsActual") or item.get("actual"))
    estimate = _float(item.get("epsEstimated") or item.get("epsEstimate") or item.get("estimate"))
    if observed is None or estimate == 0 or math.isnan(actual) or math.isnan(estimate):
        return None
    surprise = (actual - estimate) / abs(estimate) * 100.0
    return {
        "ticker": ticker.upper(),
        "date": observed,
        "eps_actual": actual,
        "eps_estimate": estimate,
        "eps_surprise_pct": surprise,
    }


def _sec_13f_filings(cik: str, start: datetime.date, end: datetime.date, settings) -> list[dict]:
    try:
        response = httpx.get(
            f"https://data.sec.gov/submissions/CIK{str(cik).zfill(10)}.json",
            headers={"User-Agent": settings.sec_edgar_user_agent},
            timeout=30,
        )
        response.raise_for_status()
        recent = response.json().get("filings", {}).get("recent", {})
    except Exception:
        return []
    filings = []
    for idx, form in enumerate(recent.get("form", [])):
        if form != "13F-HR":
            continue
        filing_date = _safe_date(_at(recent.get("filingDate", []), idx))
        if not filing_date or filing_date < start or filing_date > end:
            continue
        accession = _at(recent.get("accessionNumber", []), idx)
        if accession:
            filings.append({"filing_date": filing_date, "accession": accession})
    return sorted(filings, key=lambda row: row["filing_date"], reverse=True)


def _fetch_13f_info_table(cik: str, filing: dict, settings, name_map: dict[str, str]) -> dict[str, dict]:
    accession = filing["accession"]
    base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}"
    try:
        index = httpx.get(
            f"{base}/index.json",
            headers={"User-Agent": settings.sec_edgar_user_agent},
            timeout=30,
        )
        index.raise_for_status()
        items = index.json().get("directory", {}).get("item", [])
    except Exception:
        return {}
    candidates = [
        item["name"]
        for item in items
        if str(item.get("name", "")).lower().endswith(".xml")
        and "info" in str(item.get("name", "")).lower()
    ]
    if not candidates:
        candidates = [
            item["name"]
            for item in items
            if str(item.get("name", "")).lower().endswith(".xml")
        ]
    for document in candidates:
        try:
            response = httpx.get(
                f"{base}/{document}",
                headers={"User-Agent": settings.sec_edgar_user_agent},
                timeout=30,
            )
            response.raise_for_status()
            root = ET.fromstring(response.content)
        except Exception:
            continue
        holdings: dict[str, dict] = {}
        for info_table in root.findall(".//{*}infoTable"):
            issuer = _xml_text_any(info_table, "nameOfIssuer")
            ticker = name_map.get(_normalize_company_name(issuer or ""))
            if not ticker:
                continue
            shares = _float(_xml_text_any(info_table, "sshPrnamt"))
            value_thousands = _float(_xml_text_any(info_table, "value"))
            shares = 0.0 if math.isnan(shares) else shares
            value_thousands = 0.0 if math.isnan(value_thousands) else value_thousands
            holdings[ticker] = {
                "shares": shares,
                "market_value_usd": value_thousands * 1000.0,
            }
        if holdings:
            return holdings
    return {}


def _price_context(prices: pd.DataFrame, as_of: datetime.date) -> dict[str, float | str]:
    if prices.empty:
        return {"price_level": "unknown", "price_momentum_20d": 0.0}
    available = prices[prices.index <= as_of]
    if available.empty:
        return {"price_level": "unknown", "price_momentum_20d": 0.0}
    window = available.tail(20)
    close = float(window.iloc[-1]["close"])
    low = float(window["close"].min())
    high = float(window["close"].max())
    first = float(window.iloc[0]["close"])
    momentum = (close - first) / first if first else 0.0
    level = "middle"
    if low > 0 and close <= low * 1.03:
        level = "support"
    elif high > 0 and close >= high * 0.97:
        level = "resistance"
    return {"price_level": level, "price_momentum_20d": momentum}


def _macro_for_date(macro_df: pd.DataFrame, as_of: datetime.date) -> dict[str, float]:
    if macro_df.empty:
        return {}
    sub = macro_df[macro_df["date"] <= as_of]
    if sub.empty:
        return {}
    row = sub.iloc[-1]
    return {
        "yield_spread_2_10": float(row.get("yield_spread_2_10", 0.0)),
        "cpi_mom": float(row.get("cpi_mom", 0.0)),
        "ism_pmi": float(row.get("ism_pmi", 0.0)),
        "unemployment_claims": float(row.get("unemployment_claims", 0.0)),
        "fed_funds_rate": float(row.get("fed_funds_rate", 0.0)),
    }


def _sec_archive_url(cik: str, accession: str, document: str) -> str:
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/{document}"


def _sec_filing_url_from_hit(value: str) -> str:
    if value.startswith("http"):
        return value
    if "/" in value:
        return f"https://www.sec.gov/Archives/{value.lstrip('/')}"
    return value


def _xml_text(root, path: str) -> str | None:
    node = root.find(path)
    if node is None or node.text is None:
        return None
    return node.text.strip()


def _xml_text_any(root, local_name: str) -> str | None:
    node = root.find(f".//{{*}}{local_name}")
    if node is None or node.text is None:
        return None
    return node.text.strip()


def _safe_date(value) -> datetime.date | None:
    if not value:
        return None
    text = str(value)[:10]
    try:
        return datetime.date.fromisoformat(text)
    except ValueError:
        return None


def _at(values, idx: int):
    return values[idx] if idx < len(values) else None


def _float(value) -> float:
    if value is None:
        return 0.0
    text = str(value).strip().replace(",", "")
    if text in {"", ".", "None", "nan"}:
        return math.nan
    try:
        return float(text)
    except ValueError:
        return 0.0


def _normalize_company_name(value: str) -> str:
    value = value.casefold()
    value = re.sub(r"\b(inc|incorporated|corp|corporation|co|company|class|plc|ltd|llc)\b", "", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


# ---------------------------------------------------------------------------
# Polygon price helpers
# ---------------------------------------------------------------------------

def fetch_polygon_prices(ticker: str, start: datetime.date, end: datetime.date, api_key: str) -> pd.DataFrame:
    """Return daily close prices from Polygon for ticker between start and end."""
    url = f"{POLYGON_BASE}/{ticker.upper()}/range/1/day/{start.isoformat()}/{end.isoformat()}"
    for attempt in range(7):
        try:
            r = httpx.get(url, params={"adjusted": "true", "sort": "asc", "limit": 5000, "apiKey": api_key}, timeout=30)
            if r.status_code == 429:
                wait = 12 * (attempt + 1)
                log.info("Rate limited, sleeping %ds …", wait)
                time.sleep(wait)
                continue
            r.raise_for_status()
            results = r.json().get("results", [])
            if not results:
                return pd.DataFrame(columns=["date", "close"])
            rows = [{"date": datetime.date.fromtimestamp(item["t"] / 1000), "close": item["c"]} for item in results]
            return pd.DataFrame(rows).set_index("date").sort_index()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (403, 404):
                log.warning("Polygon %s %s: %s", ticker, exc.response.status_code, exc)
                return pd.DataFrame(columns=["date", "close"])
            raise
    return pd.DataFrame(columns=["date", "close"])


def compute_forward_return(prices: pd.DataFrame, entry_date: datetime.date, holding_days: int) -> float | None:
    """Compute the pct return from entry_date over holding_days trading days."""
    dates = prices.index.tolist()
    if entry_date not in dates:
        # use next available
        future = [d for d in dates if d >= entry_date]
        if not future:
            return None
        entry_date = future[0]
    idx = dates.index(entry_date)
    exit_idx = min(idx + holding_days, len(dates) - 1)
    if exit_idx == idx:
        return None
    entry_price = prices.loc[entry_date, "close"]
    exit_price = prices.iloc[exit_idx]["close"]
    if entry_price == 0:
        return None
    return float((exit_price - entry_price) / entry_price)


# ---------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------

def build_signal_bundle(
    ticker: str,
    as_of: datetime.date,
    *,
    congress_buy: int = 0,
    congress_sell: int = 0,
    congress_amount: float = 0.0,
    lobby_amount: float = 0.0,
    contract_amount: float = 0.0,
    patent_count: int = 0,
    wsb_sentiment: float = 0.0,
    wsb_mentions: int = 0,
    fear_greed: float = 50.0,
    ibkr_buy: int = 0,
    ibkr_sell: int = 0,
    ibkr_qty: float = 0.0,
    ibkr_price: float = 0.0,
    insider_buy_qty: float = 0.0,
    insider_sell_qty: float = 0.0,
    insider_net_qty: float = 0.0,
    insider_value_usd: float = 0.0,
    insider_officer_count: int = 0,
    insider_director_count: int = 0,
    yield_spread_2_10: float | None = None,
    cpi_mom: float | None = None,
    ism_pmi: float | None = None,
    unemployment_claims: float | None = None,
    fed_funds_rate: float | None = None,
    eps_actual: float = 0.0,
    eps_estimate: float = 0.0,
    eps_surprise_pct: float = 0.0,
    put_call_ratio: float = 0.0,
    put_open_interest: float = 0.0,
    call_open_interest: float = 0.0,
    price_level: str = "unknown",
    price_momentum_20d: float = 0.0,
    institutional_delta_shares: float = 0.0,
    institutional_delta_pct: float = 0.0,
    institutional_manager: str = "",
    institutional_market_value_usd: float = 0.0,
    short_interest_shares: float = 0.0,
    days_to_cover: float = 0.0,
    short_interest_change_pct: float = 0.0,
    # yfinance fundamentals
    yf_pe_ratio: float = 0.0,
    yf_forward_pe: float = 0.0,
    yf_revenue_growth: float = 0.0,
    yf_earnings_growth: float = 0.0,
    yf_beta: float = 1.0,
    yf_analyst_upside_pct: float = 0.0,
    yf_short_ratio: float = 0.0,
    yf_profit_margin: float = 0.0,
    yf_debt_to_equity: float = 0.0,
    yf_roe: float = 0.0,
    yf_institutional_pct_held: float = 0.0,
    yf_fifty_two_week_high_pct: float = 0.0,
    yf_recommendation: str = "none",
    # yfinance options
    yf_put_call_ratio: float = 0.0,
    yf_implied_volatility_avg: float = 0.0,
    # Wikipedia pageviews
    wiki_pageviews: int = 0,
    # Reddit multi-subreddit
    reddit_mentions: int = 0,
    reddit_sentiment_score: float = 0.0,
    # Google News
    gnews_mentions: int = 0,
    gnews_sentiment: float = 0.0,
    # Crypto macro (BTC/ETH risk-on indicator)
    btc_price: float = 0.0,
    btc_7d_change_pct: float = 0.0,
    eth_price: float = 0.0,
    # CFTC positioning
    cftc_sp500_net_noncomm: float = 0.0,
    # BLS macro
    bls_nonfarm_payrolls: float = 0.0,
    bls_unemployment_rate: float = 0.0,
    bls_ppi_finished_goods: float = 0.0,
    # EIA energy
    eia_crude_wti: float = 0.0,
    eia_natural_gas: float = 0.0,
    # Alpha Vantage technicals
    av_rsi_14: float = 50.0,
    av_macd_signal: float = 0.0,
    # SEC XBRL
    xbrl_revenue: float = 0.0,
    xbrl_net_income: float = 0.0,
    xbrl_gross_margin_pct: float = 0.0,
    # USASpending contracts
    usa_contract_amount: float = 0.0,
    # Stocktwits trader sentiment
    st_bull_score: float = 0.0,
    st_total: int = 0,
    # Google Trends
    gtrends_interest: int = 0,
    # VIX level (from CBOE)
    vix_close: float = 0.0,
    vix_1m_avg: float = 0.0,
    # Extended FRED
    fred_hy_spread: float = 0.0,
    fred_m2_billions: float = 0.0,
    fred_mortgage30: float = 0.0,
    fred_treasury_10y: float = 0.0,
    # SEC 8-K events
    sec_8k_count: int = 0,
    # USD strength
    usd_strength_index: float = 0.0,
    # HN tech buzz
    hn_hits: int = 0,
    hn_sentiment: float = 0.0,
    # PatentsView
    pv_patent_count: int = 0,
    # World Bank macro
    wb_us_gdp_growth: float = 0.0,
    wb_us_inflation: float = 0.0,
    # OpenInsider
    oi_buy_count: int = 0,
    oi_sell_count: int = 0,
    oi_net_value: float = 0.0,
    # GDELT news tone
    gdelt_avg_tone: float = 0.0,
    gdelt_article_count: int = 0,
) -> SignalBundle:
    signals: list[Signal] = []

    # Congressional insider signal
    if congress_buy > 0 or congress_sell > 0:
        net = congress_buy - congress_sell
        direction = SignalDirection.LONG if net > 0 else SignalDirection.SHORT
        strength = min(abs(net) / max(congress_buy + congress_sell, 1), 1.0)
        confidence = min(congress_amount / 500_000, 1.0) if congress_amount > 0 else 0.5
        signals.append(Signal(
            name="congressional_insider",
            ticker=ticker,
            source="quiver",
            direction=direction,
            strength=round(strength, 4),
            confidence=round(confidence, 4),
            effective_date=as_of,
            horizon_days=HOLDING_DAYS,
            reasons=[f"{congress_buy} buys, {congress_sell} sells, ${congress_amount:,.0f} disclosed"],
        ))

    # Lobbying signal — heavy lobbying → company expects upcoming regulatory benefit
    if lobby_amount > 0:
        norm = min(lobby_amount / 5_000_000, 1.0)
        signals.append(Signal(
            name="lobbying_activity",
            ticker=ticker,
            source="quiver",
            direction=SignalDirection.LONG,
            strength=round(norm * 0.6, 4),
            confidence=0.4,
            effective_date=as_of,
            horizon_days=60,
            reasons=[f"${lobby_amount:,.0f} lobbying spend"],
        ))

    # Government contracts signal
    if contract_amount > 0:
        norm = min(contract_amount / 10_000_000, 1.0)
        signals.append(Signal(
            name="government_contract",
            ticker=ticker,
            source="quiver",
            direction=SignalDirection.LONG,
            strength=round(norm * 0.7, 4),
            confidence=0.55,
            effective_date=as_of,
            horizon_days=HOLDING_DAYS,
            reasons=[f"${contract_amount:,.0f} contract award"],
        ))

    # Patent filings — R&D momentum
    if patent_count > 0:
        norm = min(patent_count / 10, 1.0)
        signals.append(Signal(
            name="patent_filings",
            ticker=ticker,
            source="quiver",
            direction=SignalDirection.LONG,
            strength=round(norm * 0.4, 4),
            confidence=0.35,
            effective_date=as_of,
            horizon_days=90,
            reasons=[f"{patent_count} patents filed"],
        ))

    # WSB retail sentiment — contrarian or momentum depending on magnitude
    if wsb_mentions >= 10:
        wsb_norm = max(-1.0, min(1.0, wsb_sentiment))
        direction = SignalDirection.LONG if wsb_norm > 0 else SignalDirection.SHORT
        confidence = min(wsb_mentions / 1000, 0.6)
        signals.append(Signal(
            name="wsb_sentiment",
            ticker=ticker,
            source="reddit",
            direction=direction,
            strength=round(abs(wsb_norm) * 0.5, 4),
            confidence=round(confidence, 4),
            effective_date=as_of,
            horizon_days=7,
            reasons=[f"sentiment={wsb_sentiment:.3f}, mentions={wsb_mentions}"],
        ))

    # Fear & Greed macro signal
    fg_norm = (fear_greed - 50) / 50  # -1 (extreme fear) to +1 (extreme greed)
    direction = SignalDirection.LONG if fg_norm > 0 else SignalDirection.SHORT
    signals.append(Signal(
        name="fear_greed_macro",
        ticker=ticker,
        source="internal",
        direction=direction,
        strength=round(abs(fg_norm) * 0.5, 4),
        confidence=0.45,
        effective_date=as_of,
        horizon_days=14,
        reasons=[f"Fear & Greed Index={fear_greed:.1f}"],
    ))

    # IBKR execution signal — confirmed real orders placed by our own system
    if ibkr_buy > 0 or ibkr_sell > 0:
        net = ibkr_buy - ibkr_sell
        direction = SignalDirection.LONG if net > 0 else SignalDirection.SHORT
        strength = min(ibkr_qty / 1000, 1.0)
        signals.append(Signal(
            name="ibkr_execution",
            ticker=ticker,
            source="ibkr",
            direction=direction,
            strength=round(strength, 4),
            confidence=0.70,
            effective_date=as_of,
            horizon_days=HOLDING_DAYS,
            reasons=[
                f"{ibkr_buy} buy fills, {ibkr_sell} sell fills, "
                f"{ibkr_qty:.0f} shares @ ${ibkr_price:.2f}"
            ],
        ))

    # SEC Form 4 insider trade signal. Officers receive a higher score because
    # their transactions tend to be more tied to operating visibility.
    if insider_buy_qty > 0 or insider_sell_qty > 0:
        direction = SignalDirection.LONG if insider_net_qty > 0 else SignalDirection.SHORT
        signal_name = "insider_buy" if direction is SignalDirection.LONG else "insider_sell"
        role_bonus = 0.20 if insider_officer_count > 0 else (0.10 if insider_director_count > 0 else 0.0)
        value_score = min(insider_value_usd / 2_500_000, 1.0) if insider_value_usd > 0 else 0.0
        qty_score = min(abs(insider_net_qty) / 100_000, 1.0)
        strength = min(1.0, 0.35 + 0.35 * value_score + 0.20 * qty_score + role_bonus)
        confidence = min(0.85, 0.45 + 0.20 * value_score + role_bonus)
        signals.append(Signal(
            name=signal_name,
            ticker=ticker,
            source="sec_edgar",
            direction=direction,
            strength=round(strength, 4),
            confidence=round(confidence, 4),
            effective_date=as_of,
            horizon_days=20,
            reasons=[
                (
                    f"Form 4 net qty {insider_net_qty:,.0f}, "
                    f"value ${insider_value_usd:,.0f}, "
                    f"officers={insider_officer_count}, directors={insider_director_count}"
                )
            ],
            metadata={
                "insider_buy_qty": insider_buy_qty,
                "insider_sell_qty": insider_sell_qty,
                "insider_net_qty": insider_net_qty,
                "transaction_value_usd": insider_value_usd,
                "insider_officer_count": insider_officer_count,
                "insider_director_count": insider_director_count,
            },
        ))

    # FRED macro regime signal.
    if yield_spread_2_10 is not None and ism_pmi is not None:
        macro_direction = SignalDirection.NEUTRAL
        macro_strength = 0.0
        macro_reason = "neutral macro regime"
        if yield_spread_2_10 < 0 and ticker.upper() in CYCLICAL_TICKERS:
            macro_direction = SignalDirection.SHORT
            macro_strength = min(abs(yield_spread_2_10) / 1.5, 1.0) * 0.65
            macro_reason = "2Y/10Y curve inverted; cyclical short bias"
        elif ism_pmi > 55 and ticker.upper() in INDUSTRIAL_TICKERS:
            macro_direction = SignalDirection.LONG
            macro_strength = min((ism_pmi - 55) / 10, 1.0) * 0.65
            macro_reason = "ISM PMI above 55; industrial long bias"
        if macro_direction is not SignalDirection.NEUTRAL:
            signals.append(Signal(
                name="macro_regime",
                ticker=ticker,
                source="fred",
                direction=macro_direction,
                strength=round(macro_strength, 4),
                confidence=0.50,
                effective_date=as_of,
                horizon_days=60,
                reasons=[macro_reason],
                metadata={
                    "yield_spread_2_10": yield_spread_2_10,
                    "cpi_mom": cpi_mom,
                    "ism_pmi": ism_pmi,
                    "unemployment_claims": unemployment_claims,
                    "fed_funds_rate": fed_funds_rate,
                },
            ))

    # Earnings surprise and post-earnings drift.
    if eps_estimate and abs(eps_surprise_pct) >= 10:
        direction = SignalDirection.LONG if eps_surprise_pct > 0 else SignalDirection.SHORT
        signal_name = "earnings_beat" if direction is SignalDirection.LONG else "earnings_miss"
        signals.append(Signal(
            name=signal_name,
            ticker=ticker,
            source="fmp",
            direction=direction,
            strength=round(min(abs(eps_surprise_pct) / 50, 1.0), 4),
            confidence=0.55,
            effective_date=as_of,
            horizon_days=10,
            reasons=[f"EPS surprise {eps_surprise_pct:+.1f}% ({eps_actual:.2f} vs {eps_estimate:.2f})"],
            metadata={
                "eps_actual": eps_actual,
                "eps_estimate": eps_estimate,
                "eps_surprise_pct": eps_surprise_pct,
            },
        ))

    # Options put/call contrarian setup.
    if put_call_ratio > 0:
        options_direction = SignalDirection.NEUTRAL
        if put_call_ratio > 1.5 and price_level == "support":
            options_direction = SignalDirection.LONG
        elif put_call_ratio < 0.5 and price_level == "resistance":
            options_direction = SignalDirection.SHORT
        if options_direction is not SignalDirection.NEUTRAL:
            signals.append(Signal(
                name="options_put_call_contrarian",
                ticker=ticker,
                source="polygon",
                direction=options_direction,
                strength=round(min(abs(put_call_ratio - 1.0) / 2.0, 1.0), 4),
                confidence=0.45,
                effective_date=as_of,
                horizon_days=7,
                reasons=[f"put/call OI ratio={put_call_ratio:.2f}, price near {price_level}"],
                metadata={
                    "put_call_ratio": put_call_ratio,
                    "put_open_interest": put_open_interest,
                    "call_open_interest": call_open_interest,
                    "price_level": price_level,
                    "price_momentum_20d": price_momentum_20d,
                },
            ))

    # SEC 13F institutional accumulation.
    if institutional_delta_shares > 0 and (institutional_delta_pct >= 0.20 or institutional_delta_pct >= 1.0):
        signals.append(Signal(
            name="institutional_accumulation",
            ticker=ticker,
            source="sec_edgar",
            direction=SignalDirection.LONG,
            strength=round(min(0.35 + institutional_delta_pct, 1.0), 4),
            confidence=0.50,
            effective_date=as_of,
            horizon_days=63,
            reasons=[
                (
                    f"{institutional_manager or 'institutional manager'} increased 13F shares "
                    f"by {institutional_delta_shares:,.0f} ({institutional_delta_pct:+.1%})"
                )
            ],
            metadata={
                "institutional_delta_shares": institutional_delta_shares,
                "institutional_delta_pct": institutional_delta_pct,
                "institutional_manager": institutional_manager,
                "institutional_market_value_usd": institutional_market_value_usd,
            },
        ))

    # FINRA short interest squeeze setup.
    if days_to_cover > 10 and price_momentum_20d > 0:
        signals.append(Signal(
            name="short_squeeze",
            ticker=ticker,
            source="finra",
            direction=SignalDirection.LONG,
            strength=round(min(days_to_cover / 30, 1.0), 4),
            confidence=0.45,
            effective_date=as_of,
            horizon_days=21,
            reasons=[f"days-to-cover={days_to_cover:.1f} with 20d momentum {price_momentum_20d:+.1%}"],
            metadata={
                "short_interest_shares": short_interest_shares,
                "days_to_cover": days_to_cover,
                "short_interest_change_pct": short_interest_change_pct,
                "price_momentum_20d": price_momentum_20d,
            },
        ))

    # ----- New signals from expanded data sources -----

    # Yahoo Finance fundamental quality signal
    if yf_revenue_growth != 0 or yf_earnings_growth != 0:
        growth_score = (yf_revenue_growth + yf_earnings_growth) / 2
        direction = SignalDirection.LONG if growth_score > 0 else SignalDirection.SHORT
        strength = min(abs(growth_score), 1.0)
        signals.append(Signal(
            name="fundamental_growth",
            ticker=ticker,
            source="yahoo_finance",
            direction=direction,
            strength=round(strength * 0.6, 4),
            confidence=0.5,
            effective_date=as_of,
            horizon_days=60,
            reasons=[f"rev_growth={yf_revenue_growth:.1%} earnings_growth={yf_earnings_growth:.1%}"],
        ))

    # Analyst upside / consensus recommendation
    if abs(yf_analyst_upside_pct) >= 5:
        direction = SignalDirection.LONG if yf_analyst_upside_pct > 0 else SignalDirection.SHORT
        consensus_bonus = {
            "strongbuy": 0.20, "buy": 0.10, "hold": 0.0,
            "sell": -0.10, "strongsell": -0.20,
        }.get(yf_recommendation.replace(" ", "").lower(), 0.0)
        strength = min(abs(yf_analyst_upside_pct) / 40 + abs(consensus_bonus), 1.0)
        signals.append(Signal(
            name="analyst_consensus",
            ticker=ticker,
            source="yahoo_finance",
            direction=direction,
            strength=round(strength * 0.55, 4),
            confidence=0.50,
            effective_date=as_of,
            horizon_days=90,
            reasons=[f"analyst target upside={yf_analyst_upside_pct:+.1f}%, rec={yf_recommendation}"],
        ))

    # High short ratio with momentum = squeeze setup (YF version, complements FINRA)
    if yf_short_ratio > 8 and price_momentum_20d > 0:
        signals.append(Signal(
            name="short_ratio_squeeze",
            ticker=ticker,
            source="yahoo_finance",
            direction=SignalDirection.LONG,
            strength=round(min(yf_short_ratio / 20, 1.0) * 0.5, 4),
            confidence=0.40,
            effective_date=as_of,
            horizon_days=14,
            reasons=[f"yf short_ratio={yf_short_ratio:.1f}"],
        ))

    # YF options implied volatility — high IV = uncertainty, use as risk dampener
    if yf_implied_volatility_avg > 0.5:
        signals.append(Signal(
            name="high_implied_volatility",
            ticker=ticker,
            source="yahoo_finance",
            direction=SignalDirection.SHORT,
            strength=round(min(yf_implied_volatility_avg / 1.5, 1.0) * 0.3, 4),
            confidence=0.35,
            effective_date=as_of,
            horizon_days=7,
            reasons=[f"yf avg IV={yf_implied_volatility_avg:.2f}"],
        ))

    # Wikipedia pageviews — spike in attention can precede volatility
    if wiki_pageviews > 50_000:
        signals.append(Signal(
            name="public_attention_spike",
            ticker=ticker,
            source="wikipedia",
            direction=SignalDirection.LONG,
            strength=round(min(wiki_pageviews / 500_000, 1.0) * 0.3, 4),
            confidence=0.30,
            effective_date=as_of,
            horizon_days=7,
            reasons=[f"wikipedia pageviews={wiki_pageviews:,}"],
        ))

    # Reddit broad-market sentiment
    if reddit_mentions >= 5:
        direction = SignalDirection.LONG if reddit_sentiment_score > 0 else SignalDirection.SHORT
        signals.append(Signal(
            name="reddit_broad_sentiment",
            ticker=ticker,
            source="reddit",
            direction=direction,
            strength=round(abs(reddit_sentiment_score) * 0.4, 4),
            confidence=round(min(reddit_mentions / 200, 0.55), 4),
            effective_date=as_of,
            horizon_days=7,
            reasons=[f"reddit mentions={reddit_mentions}, sentiment={reddit_sentiment_score:.3f}"],
        ))

    # Google News sentiment
    if gnews_mentions >= 5:
        direction = SignalDirection.LONG if gnews_sentiment > 0 else SignalDirection.SHORT
        signals.append(Signal(
            name="news_sentiment",
            ticker=ticker,
            source="google_news",
            direction=direction,
            strength=round(abs(gnews_sentiment) * 0.4, 4),
            confidence=0.40,
            effective_date=as_of,
            horizon_days=5,
            reasons=[f"gnews items={gnews_mentions}, sentiment={gnews_sentiment:.3f}"],
        ))

    # BTC risk-on/risk-off macro signal
    if btc_7d_change_pct != 0:
        direction = SignalDirection.LONG if btc_7d_change_pct > 0 else SignalDirection.SHORT
        strength = min(abs(btc_7d_change_pct) / 30, 1.0)  # normalise over 30% weekly move
        signals.append(Signal(
            name="crypto_risk_regime",
            ticker=ticker,
            source="coingecko",
            direction=direction,
            strength=round(strength * 0.35, 4),
            confidence=0.35,
            effective_date=as_of,
            horizon_days=7,
            reasons=[f"BTC 7d change={btc_7d_change_pct:+.1f}%"],
        ))

    # CFTC net non-commercial positioning on S&P futures
    if cftc_sp500_net_noncomm != 0:
        direction = SignalDirection.LONG if cftc_sp500_net_noncomm > 0 else SignalDirection.SHORT
        norm = min(abs(cftc_sp500_net_noncomm) / 200_000, 1.0)
        signals.append(Signal(
            name="cftc_futures_positioning",
            ticker=ticker,
            source="cftc",
            direction=direction,
            strength=round(norm * 0.4, 4),
            confidence=0.40,
            effective_date=as_of,
            horizon_days=21,
            reasons=[f"CFTC S&P net_noncomm={cftc_sp500_net_noncomm:,.0f}"],
        ))

    # BLS employment — strong payrolls = risk-on
    if bls_nonfarm_payrolls > 0:
        # Payrolls in thousands; above 200k month is healthy
        direction = SignalDirection.LONG if bls_nonfarm_payrolls > 150 else SignalDirection.SHORT
        strength = min(abs(bls_nonfarm_payrolls - 150) / 300, 1.0)
        signals.append(Signal(
            name="employment_macro",
            ticker=ticker,
            source="bls",
            direction=direction,
            strength=round(strength * 0.35, 4),
            confidence=0.40,
            effective_date=as_of,
            horizon_days=30,
            reasons=[f"nonfarm payrolls={bls_nonfarm_payrolls:,.0f}k, UR={bls_unemployment_rate:.1f}%"],
        ))

    # EIA crude oil as energy-sector and macro signal
    if eia_crude_wti > 0 and ticker in CYCLICAL_TICKERS | INDUSTRIAL_TICKERS:
        # High oil price = headwind for non-energy cyclicals, tailwind for energy
        energy_tickers = {"XOM", "CVX", "OXY", "SLB", "COP", "HES", "DVN", "PSX", "VLO", "MPC"}
        if ticker in energy_tickers:
            direction = SignalDirection.LONG if eia_crude_wti > 70 else SignalDirection.SHORT
        else:
            direction = SignalDirection.SHORT if eia_crude_wti > 90 else SignalDirection.LONG
        signals.append(Signal(
            name="oil_price_macro",
            ticker=ticker,
            source="eia",
            direction=direction,
            strength=round(min(abs(eia_crude_wti - 70) / 50, 1.0) * 0.3, 4),
            confidence=0.35,
            effective_date=as_of,
            horizon_days=14,
            reasons=[f"WTI crude=${eia_crude_wti:.2f}/bbl"],
        ))

    # Alpha Vantage RSI overbought/oversold
    if av_rsi_14 > 0:
        if av_rsi_14 < 30:
            signals.append(Signal(
                name="rsi_oversold",
                ticker=ticker,
                source="alpha_vantage",
                direction=SignalDirection.LONG,
                strength=round((30 - av_rsi_14) / 30, 4),
                confidence=0.50,
                effective_date=as_of,
                horizon_days=14,
                reasons=[f"RSI14={av_rsi_14:.1f} (oversold)"],
            ))
        elif av_rsi_14 > 70:
            signals.append(Signal(
                name="rsi_overbought",
                ticker=ticker,
                source="alpha_vantage",
                direction=SignalDirection.SHORT,
                strength=round((av_rsi_14 - 70) / 30, 4),
                confidence=0.45,
                effective_date=as_of,
                horizon_days=10,
                reasons=[f"RSI14={av_rsi_14:.1f} (overbought)"],
            ))

    # SEC XBRL profit quality signal
    if xbrl_gross_margin_pct > 0:
        # High gross margin (>40%) = quality business moat
        if xbrl_gross_margin_pct > 40:
            signals.append(Signal(
                name="high_gross_margin",
                ticker=ticker,
                source="sec_xbrl",
                direction=SignalDirection.LONG,
                strength=round(min((xbrl_gross_margin_pct - 40) / 60, 1.0) * 0.4, 4),
                confidence=0.45,
                effective_date=as_of,
                horizon_days=90,
                reasons=[f"gross margin={xbrl_gross_margin_pct:.1f}%"],
            ))

    # USASpending.gov supplemental government contract signal
    if usa_contract_amount > 0:
        norm = min(usa_contract_amount / 50_000_000, 1.0)
        signals.append(Signal(
            name="usa_spending_contract",
            ticker=ticker,
            source="usaspending",
            direction=SignalDirection.LONG,
            strength=round(norm * 0.5, 4),
            confidence=0.50,
            effective_date=as_of,
            horizon_days=HOLDING_DAYS,
            reasons=[f"${usa_contract_amount:,.0f} USASpending.gov contract"],
        ))

    # ----- Round 2 signals -----

    # Stocktwits trader sentiment (market-participants only, higher quality than general social)
    if st_total >= 5:
        direction = SignalDirection.LONG if st_bull_score > 0 else SignalDirection.SHORT
        signals.append(Signal(
            name="stocktwits_sentiment",
            ticker=ticker,
            source="stocktwits",
            direction=direction,
            strength=round(abs(st_bull_score) * 0.55, 4),
            confidence=round(min(st_total / 100, 0.60), 4),
            effective_date=as_of,
            horizon_days=5,
            reasons=[f"st bull_score={st_bull_score:.3f}, msgs={st_total}"],
        ))

    # Google Trends — interest spike is attention signal; extremely high = crowded
    if gtrends_interest > 0:
        if gtrends_interest >= 75:
            # High interest: can be momentum or contrarian crowding
            direction = SignalDirection.SHORT  # mean-reversion bias at extremes
            strength = (gtrends_interest - 75) / 25
        elif gtrends_interest >= 40:
            direction = SignalDirection.LONG   # rising attention = momentum
            strength = (gtrends_interest - 40) / 35
        else:
            direction = SignalDirection.NEUTRAL
            strength = 0.0
        if strength > 0:
            signals.append(Signal(
                name="google_trends_attention",
                ticker=ticker,
                source="google_trends",
                direction=direction,
                strength=round(min(strength, 1.0) * 0.35, 4),
                confidence=0.38,
                effective_date=as_of,
                horizon_days=7,
                reasons=[f"gtrends interest={gtrends_interest}/100"],
            ))

    # VIX regime — fear gauge
    if vix_close > 0:
        if vix_close > 30:
            # Elevated fear = contrarian long opportunity (market oversold)
            direction = SignalDirection.LONG
            strength = min((vix_close - 30) / 50, 1.0)
            confidence = 0.45
        elif vix_close < 15:
            # Extreme complacency = potential short-term downside risk
            direction = SignalDirection.SHORT
            strength = min((15 - vix_close) / 10, 1.0)
            confidence = 0.35
        else:
            direction = SignalDirection.NEUTRAL
            strength = 0.0
            confidence = 0.0
        if strength > 0:
            signals.append(Signal(
                name="vix_regime",
                ticker=ticker,
                source="cboe",
                direction=direction,
                strength=round(strength * 0.40, 4),
                confidence=confidence,
                effective_date=as_of,
                horizon_days=14,
                reasons=[f"VIX={vix_close:.1f} (1m avg={vix_1m_avg:.1f})"],
            ))

    # High-yield credit spread — wide spread = credit stress = risk-off
    if fred_hy_spread > 0:
        if fred_hy_spread > 6.0:
            signals.append(Signal(
                name="credit_stress",
                ticker=ticker,
                source="fred",
                direction=SignalDirection.SHORT,
                strength=round(min((fred_hy_spread - 6.0) / 8.0, 1.0) * 0.45, 4),
                confidence=0.50,
                effective_date=as_of,
                horizon_days=30,
                reasons=[f"HY spread={fred_hy_spread:.2f}%"],
            ))
        elif fred_hy_spread < 3.5:
            signals.append(Signal(
                name="credit_benign",
                ticker=ticker,
                source="fred",
                direction=SignalDirection.LONG,
                strength=round(min((3.5 - fred_hy_spread) / 3.5, 1.0) * 0.35, 4),
                confidence=0.42,
                effective_date=as_of,
                horizon_days=30,
                reasons=[f"HY spread tight={fred_hy_spread:.2f}%"],
            ))

    # SEC 8-K material events — recent filing = news catalyst
    if sec_8k_count > 0:
        signals.append(Signal(
            name="material_event_8k",
            ticker=ticker,
            source="sec_edgar",
            direction=SignalDirection.LONG,  # ambiguous direction; treat as volatility signal
            strength=round(min(sec_8k_count / 3, 1.0) * 0.40, 4),
            confidence=0.38,
            effective_date=as_of,
            horizon_days=5,
            reasons=[f"{sec_8k_count} 8-K filings"],
        ))

    # USD strength — strong USD is headwind for multinationals (most large caps)
    if usd_strength_index > 0:
        # Roughly, USD index above 100 = strong, below 90 = weak
        if usd_strength_index > 105:
            signals.append(Signal(
                name="usd_strength_headwind",
                ticker=ticker,
                source="open_exchange_rates",
                direction=SignalDirection.SHORT,
                strength=round(min((usd_strength_index - 105) / 20, 1.0) * 0.30, 4),
                confidence=0.35,
                effective_date=as_of,
                horizon_days=30,
                reasons=[f"USD strength index={usd_strength_index:.2f}"],
            ))

    # Hacker News buzz — relevant for tech tickers
    if hn_hits >= 5:
        direction = SignalDirection.LONG if hn_sentiment >= 0 else SignalDirection.SHORT
        signals.append(Signal(
            name="tech_community_buzz",
            ticker=ticker,
            source="hacker_news",
            direction=direction,
            strength=round(min(hn_hits / 50, 1.0) * 0.35, 4),
            confidence=0.35,
            effective_date=as_of,
            horizon_days=7,
            reasons=[f"hn_hits={hn_hits}, sentiment={hn_sentiment:.3f}"],
        ))

    # PatentsView patent grants — R&D output quality signal
    if pv_patent_count > 0:
        norm = min(pv_patent_count / 20, 1.0)
        signals.append(Signal(
            name="patent_grant_momentum",
            ticker=ticker,
            source="patentsview",
            direction=SignalDirection.LONG,
            strength=round(norm * 0.40, 4),
            confidence=0.38,
            effective_date=as_of,
            horizon_days=90,
            reasons=[f"{pv_patent_count} patents granted (PatentsView)"],
        ))

    # World Bank GDP growth — strong economy = risk-on
    if wb_us_gdp_growth != 0:
        if wb_us_gdp_growth >= 3.0:
            signals.append(Signal(
                name="gdp_growth_tailwind",
                ticker=ticker,
                source="world_bank",
                direction=SignalDirection.LONG,
                strength=round(min((wb_us_gdp_growth - 3.0) / 5.0, 1.0) * 0.35, 4),
                confidence=0.40,
                effective_date=as_of,
                horizon_days=90,
                reasons=[f"US GDP growth={wb_us_gdp_growth:.1f}%"],
            ))
        elif wb_us_gdp_growth < 1.0:
            signals.append(Signal(
                name="gdp_slowdown_risk",
                ticker=ticker,
                source="world_bank",
                direction=SignalDirection.SHORT,
                strength=round(min((1.0 - wb_us_gdp_growth) / 3.0, 1.0) * 0.35, 4),
                confidence=0.40,
                effective_date=as_of,
                horizon_days=90,
                reasons=[f"US GDP growth={wb_us_gdp_growth:.1f}%"],
            ))

    # OpenInsider cluster buys — multiple insiders buying = strong conviction
    if oi_buy_count > 0 or oi_sell_count > 0:
        net = oi_buy_count - oi_sell_count
        direction = SignalDirection.LONG if net > 0 else SignalDirection.SHORT
        strength = min(abs(net) / max(oi_buy_count + oi_sell_count, 1), 1.0)
        value_score = min(abs(oi_net_value) / 1_000_000, 1.0) if oi_net_value != 0 else 0.0
        signals.append(Signal(
            name="insider_cluster",
            ticker=ticker,
            source="openinsider",
            direction=direction,
            strength=round((0.5 * strength + 0.5 * value_score) * 0.60, 4),
            confidence=0.55,
            effective_date=as_of,
            horizon_days=20,
            reasons=[f"oi buys={oi_buy_count} sells={oi_sell_count} net_val=${oi_net_value:,.0f}"],
        ))

    # GDELT news tone — global media sentiment
    if gdelt_article_count >= 10:
        direction = SignalDirection.LONG if gdelt_avg_tone > 0 else SignalDirection.SHORT
        signals.append(Signal(
            name="global_news_tone",
            ticker=ticker,
            source="gdelt",
            direction=direction,
            strength=round(min(abs(gdelt_avg_tone) / 5.0, 1.0) * 0.35, 4),
            confidence=0.35,
            effective_date=as_of,
            horizon_days=7,
            reasons=[f"gdelt tone={gdelt_avg_tone:.2f}, articles={gdelt_article_count}"],
        ))

    return SignalBundle(ticker=ticker, as_of=as_of, signals=tuple(signals))


def build_trade_plan(bundle: SignalBundle, as_of: datetime.date) -> TradePlan:
    direction = bundle.direction
    conviction = round(bundle.conviction, 4)
    size_mult = round(min(1.0 + conviction, 2.0), 2)
    if direction == SignalDirection.NEUTRAL:
        size_mult = 0.0
    holding_days = _dominant_horizon_days(bundle)
    horizon_class = horizon_class_for_days(holding_days)
    stop_loss = {"short": 0.08, "medium": 0.12, "long": 0.15}[horizon_class]
    exit_trigger = (
        f"Exit if price falls more than {stop_loss:.0%} below entry or "
        f"{holding_days}-day horizon expires."
    )
    return TradePlan(
        ticker=bundle.ticker,
        as_of=as_of,
        direction=direction,
        conviction=conviction,
        size_multiplier=size_mult,
        holding_period_days=holding_days,
        horizon_class=horizon_class,
        exit_trigger=exit_trigger,
    )


def _dominant_horizon_days(bundle: SignalBundle) -> int:
    if not bundle.signals:
        return HOLDING_DAYS
    dominant = max(bundle.signals, key=lambda signal: signal.strength * signal.confidence)
    return dominant.horizon_days


# ---------------------------------------------------------------------------
# New data source loaders
# ---------------------------------------------------------------------------

# Additional RSS feeds to cast a wider news net
EXTRA_RSS_FEEDS = [
    "https://feeds.marketwatch.com/marketwatch/topstories/",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    "https://www.investing.com/rss/news.rss",
    "https://seekingalpha.com/feed.xml",
    "https://prnewswire.com/rss/news-releases-list.rss",
    "https://www.businesswire.com/rss/home/?rss=G1",
]


def load_yfinance_fundamentals(tickers: list[str]) -> pd.DataFrame:
    """
    Fetch fundamental data from Yahoo Finance (no API key required).
    Returns DataFrame with columns: ticker, pe_ratio, revenue_growth, beta,
    analyst_target, dividend_yield, short_ratio, fifty_two_week_high_pct.
    """
    try:
        import yfinance as yf
    except ImportError:
        log.warning("yfinance not installed — skipping Yahoo Finance fundamentals")
        return pd.DataFrame()

    records = []
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            info = t.info or {}
            # Safe numeric extraction helper
            def _n(key: str, default: float = 0.0) -> float:
                v = info.get(key)
                if v is None or not isinstance(v, (int, float)):
                    return default
                return float(v)

            current_price = _n("currentPrice") or _n("regularMarketPrice")
            target_price = _n("targetMeanPrice")
            high_52w = _n("fiftyTwoWeekHigh")

            records.append({
                "ticker": ticker,
                "pe_ratio": _n("trailingPE"),
                "forward_pe": _n("forwardPE"),
                "revenue_growth": _n("revenueGrowth"),  # YoY
                "earnings_growth": _n("earningsGrowth"),
                "beta": _n("beta", 1.0),
                "analyst_target": target_price,
                "analyst_upside_pct": (
                    (target_price / current_price - 1.0) * 100
                    if current_price > 0 and target_price > 0 else 0.0
                ),
                "dividend_yield": _n("dividendYield") * 100 if info.get("dividendYield") else 0.0,
                "short_ratio": _n("shortRatio"),
                "profit_margin": _n("profitMargins") * 100 if info.get("profitMargins") else 0.0,
                "debt_to_equity": _n("debtToEquity"),
                "roe": _n("returnOnEquity") * 100 if info.get("returnOnEquity") else 0.0,
                "institutional_pct_held": _n("heldPercentInstitutions") * 100,
                "fifty_two_week_high_pct": (
                    (current_price / high_52w - 1.0) * 100
                    if current_price > 0 and high_52w > 0 else 0.0
                ),
                "recommendation": str(info.get("recommendationKey", "none")),
            })
            log.info("  yfinance: fetched fundamentals for %s", ticker)
        except Exception as exc:
            log.warning("  yfinance %s failed: %s", ticker, exc)

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def load_yfinance_options_sentiment(tickers: list[str], as_of: datetime.date) -> pd.DataFrame:
    """
    Fetch options chain from Yahoo Finance and compute put/call ratio.
    Supplements the Polygon options data.
    Returns DataFrame: ticker, yf_put_call_ratio, yf_implied_volatility_avg.
    """
    try:
        import yfinance as yf
    except ImportError:
        log.warning("yfinance not installed — skipping YF options")
        return pd.DataFrame()

    records = []
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            expirations = t.options
            if not expirations:
                continue
            # Use the nearest expiry
            chain = t.option_chain(expirations[0])
            puts_vol = chain.puts["volume"].fillna(0).sum()
            calls_vol = chain.calls["volume"].fillna(0).sum()
            pc_ratio = (puts_vol / calls_vol) if calls_vol > 0 else 1.0
            avg_iv = float(
                pd.concat([chain.puts["impliedVolatility"], chain.calls["impliedVolatility"]])
                .dropna().mean() or 0.0
            )
            records.append({
                "ticker": ticker,
                "date": as_of,
                "yf_put_call_ratio": round(pc_ratio, 4),
                "yf_implied_volatility_avg": round(avg_iv, 4),
            })
        except Exception as exc:
            log.warning("  yfinance options %s failed: %s", ticker, exc)

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def load_reddit_multi_subreddit(tickers: list[str], settings) -> pd.DataFrame:
    """
    Fetch sentiment across multiple investing subreddits using PRAW.
    Subreddits: stocks, investing, options, SecurityAnalysis, dividends.
    Returns DataFrame: ticker, date, reddit_mentions, reddit_sentiment_score.
    """
    if not (settings.reddit_client_id and settings.reddit_client_secret):
        log.info("Reddit credentials not set — skipping multi-subreddit sentiment")
        return pd.DataFrame()

    try:
        import praw
    except ImportError:
        log.warning("praw not installed — skipping Reddit multi-subreddit")
        return pd.DataFrame()

    subreddits = ["stocks", "investing", "options", "SecurityAnalysis", "dividends", "algotrading"]
    records = []
    try:
        reddit = praw.Reddit(
            client_id=settings.reddit_client_id.get_secret_value(),
            client_secret=settings.reddit_client_secret.get_secret_value(),
            user_agent=settings.reddit_user_agent,
        )
        for ticker in tickers:
            mention_count = 0
            positive = 0
            negative = 0
            for sub_name in subreddits:
                try:
                    sub = reddit.subreddit(sub_name)
                    for post in sub.search(ticker, limit=20, sort="new", time_filter="month"):
                        mention_count += 1
                        text = (post.title + " " + (post.selftext or "")).lower()
                        pos_words = ["buy", "bullish", "long", "moon", "calls", "undervalued", "growth"]
                        neg_words = ["sell", "bearish", "short", "puts", "overvalued", "crash", "dump"]
                        pos = sum(1 for w in pos_words if w in text)
                        neg = sum(1 for w in neg_words if w in text)
                        positive += pos
                        negative += neg
                except Exception as exc:
                    log.debug("  reddit sub %s search failed for %s: %s", sub_name, ticker, exc)

            if mention_count > 0:
                sentiment = (positive - negative) / max(positive + negative, 1)
                records.append({
                    "ticker": ticker,
                    "date": datetime.date.today(),
                    "reddit_mentions": mention_count,
                    "reddit_sentiment_score": round(sentiment, 4),
                })
    except Exception as exc:
        log.warning("Reddit multi-subreddit failed: %s", exc)

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def load_wikipedia_pageviews(tickers: list[str], start: datetime.date, end: datetime.date) -> pd.DataFrame:
    """
    Fetch Wikipedia pageview counts as a proxy for public interest.
    Maps tickers to company names for Wikipedia article lookup.
    Returns DataFrame: ticker, date, wiki_pageviews.
    """
    TICKER_TO_WIKI = {
        "AAPL": "Apple_Inc.", "MSFT": "Microsoft", "NVDA": "Nvidia", "AMZN": "Amazon_(company)",
        "GOOGL": "Alphabet_Inc.", "META": "Meta_Platforms", "TSLA": "Tesla,_Inc.",
        "JPM": "JPMorgan_Chase", "BAC": "Bank_of_America", "GS": "Goldman_Sachs",
        "BA": "Boeing", "CAT": "Caterpillar_Inc.", "DE": "John_Deere", "GE": "GE_Aerospace",
        "F": "Ford_Motor_Company", "GM": "General_Motors", "DAL": "Delta_Air_Lines",
        "AAL": "American_Airlines", "UAL": "United_Airlines", "LUV": "Southwest_Airlines",
        "DIS": "The_Walt_Disney_Company", "NFLX": "Netflix", "ROKU": "Roku,_Inc.",
        "CCL": "Carnival_Corporation", "RCL": "Royal_Caribbean_Group",
        "MAR": "Marriott_International", "WYNN": "Wynn_Resorts", "NCLH": "Norwegian_Cruise_Line",
        "HD": "The_Home_Depot", "UPS": "United_Parcel_Service", "FDX": "FedEx",
        "RTX": "RTX_Corporation", "NOC": "Northrop_Grumman", "HON": "Honeywell",
        "MMM": "3M", "EMR": "Emerson_Electric", "ETN": "Eaton_Corporation",
        "DOV": "Dover_Corporation", "UNP": "Union_Pacific_Corporation",
    }
    records = []
    start_str = start.strftime("%Y%m%d")
    end_str = end.strftime("%Y%m%d")
    for ticker in tickers:
        article = TICKER_TO_WIKI.get(ticker)
        if not article:
            continue
        url = (
            f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
            f"/en.wikipedia/all-access/all-agents/{article}/daily/{start_str}/{end_str}"
        )
        try:
            r = httpx.get(url, timeout=15, headers={"User-Agent": "AI-Trader research bot"})
            if r.status_code != 200:
                continue
            items = r.json().get("items", [])
            for item in items:
                try:
                    dt = datetime.datetime.strptime(item["timestamp"], "%Y%m%d00").date()
                    records.append({
                        "ticker": ticker,
                        "date": dt,
                        "wiki_pageviews": int(item.get("views", 0)),
                    })
                except Exception:
                    pass
        except Exception as exc:
            log.warning("  Wikipedia pageviews %s failed: %s", ticker, exc)
        time.sleep(0.1)

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def load_usaspending_contracts(tickers: list[str], start: datetime.date, end: datetime.date) -> pd.DataFrame:
    """
    Fetch government contract awards from USASpending.gov (no API key required).
    Supplements the Quiver government contracts data.
    Returns DataFrame: ticker, date, usa_contract_amount.
    """
    TICKER_TO_RECIPIENT = {
        "BA": "Boeing", "RTX": "Raytheon", "NOC": "Northrop Grumman",
        "GE": "General Electric", "HON": "Honeywell", "CAT": "Caterpillar",
        "LMT": "Lockheed Martin", "GD": "General Dynamics",
    }
    records = []
    for ticker in tickers:
        recipient = TICKER_TO_RECIPIENT.get(ticker)
        if not recipient:
            continue
        try:
            payload = {
                "filters": {
                    "time_period": [{"start_date": start.isoformat(), "end_date": end.isoformat()}],
                    "award_type_codes": ["A", "B", "C", "D"],
                    "recipient_search_text": [recipient],
                },
                "fields": ["Award ID", "Recipient Name", "Award Amount", "Action Date"],
                "page": 1, "limit": 50, "sort": "Action Date", "order": "desc",
            }
            r = httpx.post(
                "https://api.usaspending.gov/api/v2/search/spending_by_award/",
                json=payload, timeout=30,
                headers={"Content-Type": "application/json"},
            )
            if r.status_code != 200:
                continue
            results = r.json().get("results", [])
            for award in results:
                try:
                    dt = datetime.date.fromisoformat(award["Action Date"])
                    amount = float(award.get("Award Amount") or 0)
                    records.append({"ticker": ticker, "date": dt, "usa_contract_amount": amount})
                except Exception:
                    pass
        except Exception as exc:
            log.warning("  USASpending %s failed: %s", ticker, exc)
        time.sleep(0.2)

    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df = df.groupby(["ticker", "date"])["usa_contract_amount"].sum().reset_index()
    return df


def load_cftc_positioning() -> pd.DataFrame:
    """
    Fetch CFTC Commitments of Traders (net non-commercial positioning) for key futures.
    Returns DataFrame indexed by date with columns for net positioning.
    Uses the public CSV: https://www.cftc.gov/dea/newcot/FinFutNet.txt
    """
    url = "https://www.cftc.gov/dea/newcot/FinFutNet.txt"
    try:
        r = httpx.get(url, timeout=30, headers={"User-Agent": "AI-Trader research bot"})
        r.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(r.text), header=0)
        # Standardize column names (CFTC CSV has many columns)
        df.columns = [c.strip() for c in df.columns]
        # Try to find date column and net positioning
        date_col = next((c for c in df.columns if "date" in c.lower()), None)
        if date_col:
            df["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.date
        # Look for S&P 500 row as broad market positioning indicator
        name_col = next((c for c in df.columns if "market" in c.lower() or "name" in c.lower() or "commodity" in c.lower()), None)
        net_col = next((c for c in df.columns if "net" in c.lower() and "noncomm" in c.lower()), None)
        if name_col and net_col and date_col:
            sp_mask = df[name_col].str.contains("S&P 500|S&P500|NASDAQ|E-MINI", case=False, na=False)
            sp_df = df[sp_mask][["date", net_col]].copy()
            sp_df = sp_df.rename(columns={net_col: "cftc_sp500_net_noncomm"})
            sp_df["cftc_sp500_net_noncomm"] = pd.to_numeric(sp_df["cftc_sp500_net_noncomm"], errors="coerce").fillna(0)
            return sp_df.dropna(subset=["date"]).set_index("date").sort_index()
    except Exception as exc:
        log.warning("CFTC CoT fetch failed: %s", exc)
    return pd.DataFrame()


def load_coingecko_crypto(start: datetime.date, end: datetime.date) -> pd.DataFrame:
    """
    Fetch BTC and ETH prices from CoinGecko as risk-on/risk-off macro indicator.
    Returns DataFrame indexed by date with columns: btc_price, eth_price, btc_7d_change_pct.
    """
    records = []
    for coin_id, col in [("bitcoin", "btc_price"), ("ethereum", "eth_price")]:
        try:
            days = max(1, (end - start).days + 1)
            r = httpx.get(
                f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart",
                params={"vs_currency": "usd", "days": str(days), "interval": "daily"},
                timeout=20,
                headers={"Accept": "application/json"},
            )
            if r.status_code != 200:
                continue
            prices = r.json().get("prices", [])
            for ts_ms, price in prices:
                dt = datetime.date.fromtimestamp(ts_ms / 1000)
                if start <= dt <= end:
                    records.append({"date": dt, col: price})
        except Exception as exc:
            log.warning("CoinGecko %s failed: %s", coin_id, exc)
        time.sleep(1.5)  # CoinGecko free tier rate limit

    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records).groupby("date").first()
    # Compute BTC 7-day change
    if "btc_price" in df.columns:
        df["btc_7d_change_pct"] = df["btc_price"].pct_change(7) * 100
    return df.sort_index()


def load_sec_xbrl_financials(tickers: list[str], settings) -> pd.DataFrame:
    """
    Fetch structured financial data from SEC EDGAR XBRL API.
    Returns DataFrame: ticker, revenue_ttm, net_income_ttm, eps_ttm, gross_margin.
    """
    user_agent = getattr(settings, "sec_edgar_user_agent", "AI-Trader research bot")
    headers = {"User-Agent": user_agent, "Accept": "application/json"}

    # First, get CIK mapping from SEC
    cik_map: dict[str, str] = {}
    try:
        r = httpx.get(SEC_COMPANY_TICKERS_URL, headers=headers, timeout=20)
        if r.status_code == 200:
            data = r.json()
            for _, entry in data.items():
                t = entry.get("ticker", "")
                cik = str(entry.get("cik_str", "")).zfill(10)
                if t:
                    cik_map[t.upper()] = cik
    except Exception as exc:
        log.warning("SEC CIK fetch failed: %s", exc)

    records = []
    for ticker in tickers:
        cik = cik_map.get(ticker)
        if not cik:
            continue
        try:
            url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
            r = httpx.get(url, headers=headers, timeout=30)
            if r.status_code != 200:
                continue
            facts = r.json().get("facts", {})
            gaap = facts.get("us-gaap", {})

            def _latest_annual(concept: str) -> float:
                entries = gaap.get(concept, {}).get("units", {}).get("USD", [])
                annual = [e for e in entries if e.get("form") == "10-K" and "val" in e]
                if not annual:
                    return 0.0
                return float(sorted(annual, key=lambda e: e.get("end", ""))[-1]["val"])

            def _latest_pure(concept: str) -> float:
                entries = gaap.get(concept, {}).get("units", {}).get("pure", [])
                annual = [e for e in entries if e.get("form") == "10-K" and "val" in e]
                if not annual:
                    return 0.0
                return float(sorted(annual, key=lambda e: e.get("end", ""))[-1]["val"])

            revenue = _latest_annual("Revenues") or _latest_annual("RevenueFromContractWithCustomerExcludingAssessedTax")
            net_income = _latest_annual("NetIncomeLoss")
            gross_profit = _latest_annual("GrossProfit")
            gross_margin = (gross_profit / revenue * 100) if revenue > 0 else 0.0

            records.append({
                "ticker": ticker,
                "xbrl_revenue": revenue,
                "xbrl_net_income": net_income,
                "xbrl_gross_margin_pct": round(gross_margin, 2),
            })
            log.info("  SEC XBRL: fetched financials for %s", ticker)
        except Exception as exc:
            log.warning("  SEC XBRL %s failed: %s", ticker, exc)
        time.sleep(0.15)

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def load_bls_macro(start: datetime.date, end: datetime.date, api_key: str | None = None) -> pd.DataFrame:
    """
    Fetch macro economic data from BLS (Bureau of Labor Statistics).
    Series: CES0000000001 (total nonfarm payrolls), PCU (PPI), LNS14000000 (unemployment rate).
    Returns DataFrame indexed by date with BLS series values.
    Free API key gives higher limits (500 series/day vs 25/day).
    """
    series = {
        "CES0000000001": "bls_nonfarm_payrolls",  # total nonfarm payrolls (thousands)
        "LNS14000000": "bls_unemployment_rate",   # unemployment rate %
        "WPUFD49104": "bls_ppi_finished_goods",   # PPI finished goods
    }
    start_year = start.year
    end_year = end.year
    records: dict[datetime.date, dict] = {}

    for series_id, col_name in series.items():
        try:
            payload: dict = {
                "seriesid": [series_id],
                "startyear": str(start_year),
                "endyear": str(end_year),
            }
            if api_key:
                payload["registrationkey"] = api_key

            r = httpx.post(
                "https://api.bls.gov/publicAPI/v2/timeseries/data/",
                json=payload, timeout=20,
                headers={"Content-Type": "application/json"},
            )
            if r.status_code != 200:
                continue
            data = r.json()
            series_data = (data.get("Results", {}).get("series") or [{}])[0]
            for obs in series_data.get("data", []):
                try:
                    period = obs.get("period", "")  # e.g., "M01"
                    year = int(obs.get("year", 0))
                    month = int(period.replace("M", "")) if period.startswith("M") else 1
                    dt = datetime.date(year, month, 1)
                    val = float(obs.get("value", 0))
                    records.setdefault(dt, {})[col_name] = val
                except Exception:
                    pass
        except Exception as exc:
            log.warning("BLS %s failed: %s", series_id, exc)

    if not records:
        return pd.DataFrame()
    df = pd.DataFrame.from_dict(records, orient="index").sort_index()
    df.index.name = "date"
    return df


def load_eia_energy(start: datetime.date, end: datetime.date, api_key: str | None = None) -> pd.DataFrame:
    """
    Fetch crude oil and natural gas prices from EIA.
    Returns DataFrame indexed by date: eia_crude_wti, eia_natural_gas.
    """
    if not api_key:
        log.info("EIA API key not set — skipping EIA energy data")
        return pd.DataFrame()

    records: dict[datetime.date, dict] = {}
    series = {
        "PET.RWTC.W": "eia_crude_wti",         # WTI crude weekly
        "NG.RNGWHHD.W": "eia_natural_gas",      # Henry Hub weekly
    }
    for series_id, col_name in series.items():
        try:
            r = httpx.get(
                "https://api.eia.gov/v2/seriesid/" + series_id,
                params={"api_key": api_key, "start": start.isoformat(), "end": end.isoformat(), "frequency": "weekly"},
                timeout=20,
            )
            if r.status_code != 200:
                continue
            for obs in r.json().get("response", {}).get("data", []):
                try:
                    dt = datetime.date.fromisoformat(obs["period"])
                    val = float(obs.get("value") or 0)
                    records.setdefault(dt, {})[col_name] = val
                except Exception:
                    pass
        except Exception as exc:
            log.warning("EIA %s failed: %s", series_id, exc)

    if not records:
        return pd.DataFrame()
    df = pd.DataFrame.from_dict(records, orient="index").sort_index()
    df.index.name = "date"
    return df


def load_alpha_vantage_technicals(
    tickers: list[str],
    start: datetime.date,
    end: datetime.date,
    api_key: str | None = None,
) -> pd.DataFrame:
    """
    Fetch RSI and MACD from Alpha Vantage.
    Free tier: 25 requests/day. Limit tickers to avoid exhausting quota.
    Returns DataFrame: ticker, date, av_rsi_14, av_macd_signal.
    """
    if not api_key:
        log.info("Alpha Vantage API key not set — skipping AV technicals")
        return pd.DataFrame()

    records = []
    MAX_TICKERS_AV = 10  # conserve free quota
    for ticker in tickers[:MAX_TICKERS_AV]:
        for function, col in [("RSI", "av_rsi_14"), ("MACD", "av_macd_signal")]:
            try:
                params: dict = {
                    "function": function,
                    "symbol": ticker,
                    "interval": "daily",
                    "apikey": api_key,
                }
                if function == "RSI":
                    params["time_period"] = "14"
                    params["series_type"] = "close"
                r = httpx.get("https://www.alphavantage.co/query", params=params, timeout=20)
                if r.status_code != 200:
                    continue
                data = r.json()
                # RSI returns "Technical Analysis: RSI", MACD returns "Technical Analysis: MACD"
                key = f"Technical Analysis: {function}"
                for date_str, vals in (data.get(key) or {}).items():
                    try:
                        dt = datetime.date.fromisoformat(date_str)
                        if start <= dt <= end:
                            val_key = "RSI" if function == "RSI" else "MACD_Signal"
                            val = float(vals.get(val_key, 0))
                            records.append({"ticker": ticker, "date": dt, col: val})
                    except Exception:
                        pass
                time.sleep(12)  # 25 req/day free = ~1 req per 3456 s; 12s is generous for batch
            except Exception as exc:
                log.warning("  Alpha Vantage %s %s failed: %s", function, ticker, exc)

    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df = df.groupby(["ticker", "date"]).first().reset_index()
    return df


def load_google_news_sentiment(tickers: list[str]) -> pd.DataFrame:
    """
    Fetch headlines from Google News RSS per ticker and compute naive sentiment score.
    Returns DataFrame: ticker, date, gnews_mentions, gnews_sentiment.
    """
    records = []
    today = datetime.date.today()
    for ticker in tickers:
        try:
            url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
            r = httpx.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0 AI-Trader research"})
            if r.status_code != 200:
                continue
            root = ET.fromstring(r.text)
            items = root.findall(".//item")
            pos_words = {"buy", "surge", "rally", "beat", "upgrade", "bullish", "gain", "profit", "record", "strong"}
            neg_words = {"sell", "drop", "crash", "miss", "downgrade", "bearish", "loss", "lawsuit", "fraud", "decline"}
            pos = neg = 0
            for item in items[:40]:
                text = ((item.findtext("title") or "") + " " + (item.findtext("description") or "")).lower()
                pos += sum(1 for w in pos_words if w in text)
                neg += sum(1 for w in neg_words if w in text)
            sentiment = (pos - neg) / max(pos + neg, 1)
            records.append({
                "ticker": ticker,
                "date": today,
                "gnews_mentions": len(items),
                "gnews_sentiment": round(sentiment, 4),
            })
            log.info("  Google News: %d headlines for %s, sentiment=%.3f", len(items), ticker, sentiment)
        except Exception as exc:
            log.warning("  Google News %s failed: %s", ticker, exc)
        time.sleep(0.5)

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def load_stocktwits_sentiment(tickers: list[str]) -> pd.DataFrame:
    """
    Fetch trader sentiment from Stocktwits public API (no key required).
    Returns DataFrame: ticker, date, st_bullish, st_bearish, st_total, st_bull_score.
    """
    records = []
    today = datetime.date.today()
    for ticker in tickers:
        try:
            r = httpx.get(
                f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json",
                params={"limit": "30"},
                timeout=15,
                headers={"User-Agent": "Mozilla/5.0 AI-Trader research"},
            )
            if r.status_code == 429:
                log.warning("  Stocktwits rate limit hit — pausing 30s")
                time.sleep(30)
                continue
            if r.status_code != 200:
                continue
            data = r.json()
            messages = data.get("messages", [])
            bullish = sum(
                1 for m in messages
                if (m.get("entities", {}).get("sentiment") or {}).get("basic") == "Bullish"
            )
            bearish = sum(
                1 for m in messages
                if (m.get("entities", {}).get("sentiment") or {}).get("basic") == "Bearish"
            )
            total = len(messages)
            bull_score = (bullish - bearish) / max(bullish + bearish, 1)
            records.append({
                "ticker": ticker,
                "date": today,
                "st_bullish": bullish,
                "st_bearish": bearish,
                "st_total": total,
                "st_bull_score": round(bull_score, 4),
            })
            log.info("  Stocktwits %s: %d bull %d bear / %d msgs", ticker, bullish, bearish, total)
        except Exception as exc:
            log.warning("  Stocktwits %s failed: %s", ticker, exc)
        time.sleep(1.0)  # respect rate limits

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def load_google_trends(tickers: list[str], timeframe: str = "today 12-m") -> pd.DataFrame:
    """
    Fetch Google Trends search interest for each ticker using pytrends.
    Returns DataFrame: ticker, date, gtrends_interest (0-100).
    Higher interest = more public attention = often precedes volatility.
    """
    try:
        from pytrends.request import TrendReq
    except ImportError:
        log.warning("pytrends not installed — skipping Google Trends. Install with: pip install pytrends")
        return pd.DataFrame()

    records = []
    BATCH = 5  # pytrends allows up to 5 terms per request
    for i in range(0, len(tickers), BATCH):
        batch = tickers[i: i + BATCH]
        try:
            pt = TrendReq(hl="en-US", tz=360, timeout=(10, 30))
            pt.build_payload(batch, cat=0, timeframe=timeframe, geo="US")
            df = pt.interest_over_time()
            if df.empty:
                continue
            df = df.drop(columns=["isPartial"], errors="ignore")
            df.index = pd.to_datetime(df.index).date
            for t in batch:
                if t not in df.columns:
                    continue
                for dt, val in df[t].items():
                    records.append({"ticker": t, "date": dt, "gtrends_interest": int(val)})
            time.sleep(2)  # avoid rate limit
        except Exception as exc:
            log.warning("  Google Trends batch %s failed: %s", batch, exc)
            time.sleep(5)

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def load_cboe_vix_history(start: datetime.date, end: datetime.date) -> pd.DataFrame:
    """
    Download VIX daily history from CBOE free CSV.
    Returns DataFrame indexed by date with columns: vix_close, vix_1m_avg.
    """
    url = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
    try:
        r = httpx.get(url, timeout=30, headers={"User-Agent": "AI-Trader research bot"})
        r.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(r.text))
        df.columns = [c.strip().lower() for c in df.columns]
        date_col = next((c for c in df.columns if "date" in c), None)
        close_col = next((c for c in df.columns if c in {"close", "vix close"}), None)
        if not date_col or not close_col:
            log.warning("CBOE VIX CSV unexpected format: %s", list(df.columns))
            return pd.DataFrame()
        df["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.date
        df["vix_close"] = pd.to_numeric(df[close_col], errors="coerce")
        df = df.dropna(subset=["date", "vix_close"]).set_index("date").sort_index()
        df = df[["vix_close"]]
        df["vix_1m_avg"] = df["vix_close"].rolling(21).mean()
        mask = (df.index >= start) & (df.index <= end)
        return df[mask]
    except Exception as exc:
        log.warning("CBOE VIX history failed: %s", exc)
        return pd.DataFrame()


def load_fred_extended(start: datetime.date, end: datetime.date, api_key: str | None = None) -> pd.DataFrame:
    """
    Fetch additional FRED series beyond the core macro set:
      - VIXCLS: VIX closing level (cross-check vs CBOE)
      - BAMLH0A0HYM2EY: ICE BofA US High Yield effective yield (credit stress)
      - M2SL: M2 money stock (liquidity)
      - MORTGAGE30US: 30-year fixed mortgage rate
      - DGS10: 10-year treasury yield
      - DGS2: 2-year treasury yield
    Returns DataFrame indexed by date.
    """
    if not api_key:
        log.info("FRED_API_KEY not set — skipping extended FRED series")
        return pd.DataFrame()

    series = {
        "VIXCLS": "fred_vix",
        "BAMLH0A0HYM2EY": "fred_hy_yield",
        "M2SL": "fred_m2_billions",
        "MORTGAGE30US": "fred_mortgage30",
        "DGS10": "fred_treasury_10y",
        "DGS2": "fred_treasury_2y",
    }
    records: dict[datetime.date, dict] = {}
    for series_id, col_name in series.items():
        try:
            r = httpx.get(
                FRED_OBSERVATIONS_URL,
                params={
                    "series_id": series_id,
                    "observation_start": start.isoformat(),
                    "observation_end": end.isoformat(),
                    "api_key": api_key,
                    "file_type": "json",
                    "frequency": "d",
                },
                timeout=20,
            )
            if r.status_code != 200:
                continue
            for obs in r.json().get("observations", []):
                try:
                    dt = datetime.date.fromisoformat(obs["date"])
                    val_str = obs.get("value", ".")
                    if val_str == ".":
                        continue
                    records.setdefault(dt, {})[col_name] = float(val_str)
                except Exception:
                    pass
        except Exception as exc:
            log.warning("FRED extended %s failed: %s", series_id, exc)

    if not records:
        return pd.DataFrame()
    df = pd.DataFrame.from_dict(records, orient="index").sort_index()
    df.index.name = "date"
    # Derived: HY spread is a credit stress indicator
    if "fred_hy_yield" in df.columns and "fred_treasury_10y" in df.columns:
        df["fred_hy_spread"] = df["fred_hy_yield"] - df["fred_treasury_10y"]
    return df


def load_sec_8k_events(
    tickers: list[str],
    start: datetime.date,
    end: datetime.date,
    settings,
) -> pd.DataFrame:
    """
    Count recent SEC 8-K filings per ticker as a material events signal.
    Uses SEC EDGAR submissions JSON (free, no key needed).
    Returns DataFrame: ticker, date, sec_8k_count.
    """
    user_agent = getattr(settings, "sec_edgar_user_agent", "AI-Trader research bot")
    headers = {"User-Agent": user_agent, "Accept": "application/json"}

    # Reuse CIK mapping from XBRL loader
    cik_map: dict[str, str] = {}
    try:
        r = httpx.get(SEC_COMPANY_TICKERS_URL, headers=headers, timeout=20)
        if r.status_code == 200:
            for _, entry in r.json().items():
                t = entry.get("ticker", "")
                cik = str(entry.get("cik_str", "")).zfill(10)
                if t:
                    cik_map[t.upper()] = cik
    except Exception as exc:
        log.warning("SEC CIK fetch for 8-K failed: %s", exc)

    records = []
    for ticker in tickers:
        cik = cik_map.get(ticker)
        if not cik:
            continue
        try:
            url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            r = httpx.get(url, headers=headers, timeout=20)
            if r.status_code != 200:
                continue
            filings = r.json().get("filings", {}).get("recent", {})
            forms = filings.get("form", [])
            dates_filed = filings.get("filedAt", filings.get("filingDate", []))
            for form, filed in zip(forms, dates_filed):
                if form != "8-K":
                    continue
                try:
                    dt = datetime.date.fromisoformat(str(filed)[:10])
                    if start <= dt <= end:
                        records.append({"ticker": ticker, "date": dt, "sec_8k_count": 1})
                except Exception:
                    pass
        except Exception as exc:
            log.warning("  SEC 8-K %s failed: %s", ticker, exc)
        time.sleep(0.15)

    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df = df.groupby(["ticker", "date"])["sec_8k_count"].sum().reset_index()
    return df


def load_usd_strength(start: datetime.date, end: datetime.date) -> pd.DataFrame:
    """
    Fetch USD exchange rates from Open Exchange Rates (free, no key needed for latest).
    Uses a basket of currencies to compute a simple USD strength index.
    Returns DataFrame indexed by date: usd_strength_index.
    """
    # Open Exchange Rates free endpoint returns latest only; we'll use FRED DXY-equivalent
    # via daily rates from an open API
    try:
        r = httpx.get(
            "https://open.er-api.com/v6/latest/USD",
            timeout=15,
        )
        if r.status_code != 200:
            return pd.DataFrame()
        data = r.json()
        rates = data.get("rates", {})
        # Simple DXY-like index: weighted vs EUR, GBP, JPY, CAD, CHF, SEK
        weights = {"EUR": 0.576, "GBP": 0.119, "JPY": 0.136, "CAD": 0.091, "CHF": 0.036, "SEK": 0.042}
        # rates are USD per unit, so EUR rate=1.09 means 1 USD = 1/1.09 EUR
        # Higher USD = lower EUR rate number (more USD buys fewer foreign units)
        # We want DXY style: higher = stronger USD
        index_val = sum(
            w * (1.0 / rates.get(ccy, 1.0))
            for ccy, w in weights.items()
        ) * 100  # arbitrary scaling
        today = datetime.date.today()
        if start <= today <= end:
            return pd.DataFrame({"date": [today], "usd_strength_index": [round(index_val, 4)]}).set_index("date")
    except Exception as exc:
        log.warning("USD strength fetch failed: %s", exc)
    return pd.DataFrame()


def load_hacker_news_tech_sentiment(tickers: list[str]) -> pd.DataFrame:
    """
    Query Hacker News Algolia API for ticker mentions as tech-community buzz signal.
    Free, no API key required. Best for tech tickers.
    Returns DataFrame: ticker, date, hn_hits, hn_sentiment.
    """
    TECH_TICKERS = {
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "NFLX",
        "AMD", "INTC", "CRM", "ORCL", "ADBE", "SNOW", "PLTR", "UBER",
        "LYFT", "ABNB", "SHOP", "SQ", "PYPL", "ROKU",
    }
    records = []
    today = datetime.date.today()
    for ticker in tickers:
        if ticker not in TECH_TICKERS:
            continue
        try:
            r = httpx.get(
                "https://hn.algolia.com/api/v1/search",
                params={"query": ticker, "tags": "story", "hitsPerPage": "50"},
                timeout=15,
            )
            if r.status_code != 200:
                continue
            hits = r.json().get("hits", [])
            pos_words = {"great", "amazing", "impressive", "bull", "growth", "launch", "beat", "record"}
            neg_words = {"terrible", "scam", "crash", "layoff", "fail", "miss", "sued", "fraud", "drop"}
            pos = neg = 0
            for h in hits:
                text = ((h.get("title") or "") + " " + (h.get("story_text") or "")).lower()
                pos += sum(1 for w in pos_words if w in text)
                neg += sum(1 for w in neg_words if w in text)
            sentiment = (pos - neg) / max(pos + neg, 1)
            records.append({
                "ticker": ticker,
                "date": today,
                "hn_hits": len(hits),
                "hn_sentiment": round(sentiment, 4),
            })
            log.info("  HN %s: %d hits, sentiment=%.3f", ticker, len(hits), sentiment)
        except Exception as exc:
            log.warning("  HN %s failed: %s", ticker, exc)
        time.sleep(0.3)

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def load_patentsview(tickers: list[str], start: datetime.date, end: datetime.date) -> pd.DataFrame:
    """
    Fetch patent grant data from USPTO PatentsView API (free, no key required).
    More comprehensive than Quiver patent data. Returns: ticker, date, pv_patent_count.
    """
    TICKER_TO_ASSIGNEE = {
        "AAPL": "Apple Inc", "MSFT": "Microsoft Technology Licensing",
        "NVDA": "NVIDIA", "AMZN": "Amazon Technologies",
        "GOOGL": "Google LLC", "META": "Meta Platforms",
        "TSLA": "Tesla", "IBM": "International Business Machines",
        "INTC": "Intel Corporation", "AMD": "Advanced Micro Devices",
        "QCOM": "QUALCOMM", "TXN": "Texas Instruments",
        "BA": "Boeing", "GE": "General Electric",
        "HON": "Honeywell", "MMM": "3M Innovative Properties",
        "RTX": "RTX Corporation", "CAT": "Caterpillar",
        "DE": "Deere", "EMR": "Emerson Electric",
    }
    records = []
    for ticker in tickers:
        assignee = TICKER_TO_ASSIGNEE.get(ticker)
        if not assignee:
            continue
        try:
            r = httpx.get(
                "https://search.patentsview.org/api/v1/patent/",
                params={
                    "q": f'{{"_and":[{{"_gte":{{"patent_date":"{start.isoformat()}"}}}},{{"_lte":{{"patent_date":"{end.isoformat()}"}}}},{{"_text_phrase":{{"assignee_organization":"{assignee}"}}}}]}}',
                    "f": '["patent_id","patent_date","assignee_organization"]',
                    "o": '{"per_page":100}',
                },
                timeout=30,
                headers={"User-Agent": "AI-Trader research bot"},
            )
            if r.status_code != 200:
                continue
            patents = r.json().get("patents") or []
            # Bin by month
            from collections import Counter
            monthly: Counter = Counter()
            for p in patents:
                try:
                    dt = datetime.date.fromisoformat(p["patent_date"][:10])
                    monthly[datetime.date(dt.year, dt.month, 1)] += 1
                except Exception:
                    pass
            for dt, count in monthly.items():
                records.append({"ticker": ticker, "date": dt, "pv_patent_count": count})
            log.info("  PatentsView %s: %d patents", ticker, len(patents))
        except Exception as exc:
            log.warning("  PatentsView %s failed: %s", ticker, exc)
        time.sleep(0.5)

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def load_world_bank_macro(start: datetime.date, end: datetime.date) -> pd.DataFrame:
    """
    Fetch US GDP growth rate and global macro indicators from World Bank Open Data API.
    Returns DataFrame indexed by year-start date: wb_us_gdp_growth, wb_us_inflation.
    """
    indicators = {
        "NY.GDP.MKTP.KD.ZG": "wb_us_gdp_growth",   # GDP growth annual %
        "FP.CPI.TOTL.ZG": "wb_us_inflation",         # CPI inflation %
        "SL.UEM.TOTL.ZS": "wb_unemployment_pct",     # Unemployment % of labor force
    }
    records: dict[datetime.date, dict] = {}
    for indicator, col_name in indicators.items():
        try:
            r = httpx.get(
                f"https://api.worldbank.org/v2/country/US/indicator/{indicator}",
                params={
                    "format": "json",
                    "per_page": "20",
                    "mrv": "10",
                    "date": f"{start.year}:{end.year}",
                },
                timeout=20,
            )
            if r.status_code != 200:
                continue
            data = r.json()
            if len(data) < 2:
                continue
            for obs in data[1] or []:
                try:
                    year = int(obs.get("date", "0"))
                    val = obs.get("value")
                    if val is None:
                        continue
                    dt = datetime.date(year, 1, 1)
                    records.setdefault(dt, {})[col_name] = float(val)
                except Exception:
                    pass
        except Exception as exc:
            log.warning("World Bank %s failed: %s", indicator, exc)

    if not records:
        return pd.DataFrame()
    df = pd.DataFrame.from_dict(records, orient="index").sort_index()
    df.index.name = "date"
    return df


def load_open_insider(tickers: list[str], start: datetime.date, end: datetime.date) -> pd.DataFrame:
    """
    Scrape OpenInsider.com for Form 4 insider transactions.
    Supplements SEC EDGAR Form 4 data with cluster buy/sell signals.
    Returns DataFrame: ticker, date, oi_buy_count, oi_sell_count, oi_net_value.
    """
    records = []
    for ticker in tickers:
        try:
            url = (
                f"http://openinsider.com/screener?s={ticker}&o=&pl=&ph=&ll=&lh="
                f"&fd=365&fdr=&td=0&tdr=&fdlyl=&fdlyh=&daysago=&xs=1"
                f"&vl=&vh=&ocl=&och=&sic1=-1&sicl=100&sich=9999"
                f"&grp=0&nfl=&nfh=&nil=&nih=&nol=&noh=&v2l=&v2h="
                f"&ov=&rc=10&d=t&download=1"
            )
            r = httpx.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0 AI-Trader research"})
            if r.status_code != 200:
                continue
            from io import StringIO
            df = pd.read_csv(StringIO(r.text))
            if df.empty:
                continue
            df.columns = [c.strip().strip('"') for c in df.columns]
            # Find date and trade type columns
            date_col = next((c for c in df.columns if "date" in c.lower() and "filing" not in c.lower()), None)
            type_col = next((c for c in df.columns if "trade" in c.lower() or "type" in c.lower()), None)
            val_col = next((c for c in df.columns if "value" in c.lower()), None)
            if not date_col:
                continue
            df["_date"] = pd.to_datetime(df[date_col], errors="coerce").dt.date
            df = df.dropna(subset=["_date"])
            mask = (df["_date"] >= start) & (df["_date"] <= end)
            df = df[mask]
            for dt, group in df.groupby("_date"):
                buys = sells = 0
                net_val = 0.0
                for _, row in group.iterrows():
                    t_type = str(row.get(type_col, "")).upper() if type_col else ""
                    if "P" in t_type or "BUY" in t_type:
                        buys += 1
                    elif "S" in t_type or "SELL" in t_type:
                        sells += 1
                    if val_col:
                        try:
                            v = str(row.get(val_col, "0")).replace("$", "").replace(",", "")
                            sign = -1 if sells > buys else 1
                            net_val += sign * float(v)
                        except Exception:
                            pass
                records.append({
                    "ticker": ticker, "date": dt,
                    "oi_buy_count": buys, "oi_sell_count": sells, "oi_net_value": net_val,
                })
        except Exception as exc:
            log.warning("  OpenInsider %s failed: %s", ticker, exc)
        time.sleep(1.0)

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def load_gdelt_news_tone(tickers: list[str], start: datetime.date, end: datetime.date) -> pd.DataFrame:
    """
    Query GDELT GKG (Global Knowledge Graph) summary API for news tone per company.
    Free, no API key. Returns DataFrame: ticker, date, gdelt_avg_tone, gdelt_article_count.
    """
    TICKER_TO_NAME = {
        "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA", "AMZN": "Amazon",
        "GOOGL": "Google", "META": "Meta", "TSLA": "Tesla", "JPM": "JPMorgan",
        "BAC": "Bank of America", "GS": "Goldman Sachs", "BA": "Boeing",
        "CAT": "Caterpillar", "GE": "General Electric", "F": "Ford", "GM": "General Motors",
        "DIS": "Disney", "NFLX": "Netflix", "HD": "Home Depot", "WMT": "Walmart",
        "PG": "Procter Gamble",
    }
    records = []
    for ticker in tickers:
        name = TICKER_TO_NAME.get(ticker)
        if not name:
            continue
        try:
            r = httpx.get(
                "https://api.gdeltproject.org/api/v2/summary/summary",
                params={
                    "d": "web", "t": "summary", "a": "domain",
                    "ni": "10", "k": f"{name} stock",
                    "ts": "custom",
                    "sd": start.strftime("%Y%m%d%H%M%S"),
                    "ed": end.strftime("%Y%m%d%H%M%S"),
                    "format": "json",
                },
                timeout=20,
                headers={"User-Agent": "AI-Trader research bot"},
            )
            if r.status_code != 200:
                continue
            data = r.json()
            # GDELT returns various tone metrics
            tone = float(data.get("avgtone", 0) or 0)
            articles = int(data.get("numarticles", 0) or 0)
            mid = (start + (end - start) / 2) if isinstance((end - start), datetime.timedelta) else start
            mid_date = start + (end - start) // 2
            records.append({
                "ticker": ticker,
                "date": mid_date,
                "gdelt_avg_tone": round(tone, 4),
                "gdelt_article_count": articles,
            })
            log.info("  GDELT %s: tone=%.2f, articles=%d", ticker, tone, articles)
        except Exception as exc:
            log.warning("  GDELT %s failed: %s", ticker, exc)
        time.sleep(0.5)

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# News fetch via RSS
# ---------------------------------------------------------------------------

def fetch_news_corpus_notes() -> list[str]:
    """Pull headlines from RSS feeds and return as plain-text notes."""
    notes: list[str] = []
    all_feeds = RSS_FEEDS + EXTRA_RSS_FEEDS
    for url in all_feeds:
        try:
            r = httpx.get(url, timeout=15, headers={"User-Agent": "AI-Trader research edenjcokeer@gmail.com"})
            r.raise_for_status()
            root = ET.fromstring(r.text)
            items = root.findall(".//item")
            for item in items[:30]:
                title = (item.findtext("title") or "").strip()
                desc = (item.findtext("description") or "").strip()
                if title:
                    notes.append(f"{title}. {desc}"[:300])
            log.info("Fetched %d headlines from %s", len(items), url)
        except Exception as exc:
            log.warning("RSS %s failed: %s", url, exc)
    return notes
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest all training data to JSONL")
    parser.add_argument("--out", default="logs/training_examples.jsonl", help="Output JSONL path")
    parser.add_argument("--tickers", nargs="*", help="Limit to these tickers (default: all found in data)")
    parser.add_argument("--min-date", default="2018-01-01", help="Earliest date to include")
    parser.add_argument("--max-date", default=datetime.date.today().isoformat(), help="Latest date to include")
    parser.add_argument("--news-corpus-out", default="examples/trader_corpus/live_news.txt", help="Write news headlines to RAG corpus file")
    parser.add_argument("--no-ibkr", action="store_true", help="Skip IBKR execution data even if TWS/IB Gateway is running")
    args = parser.parse_args()

    settings = get_settings()
    if settings.polygon_api_key is None:
        log.error("POLYGON_API_KEY required for price data. Aborting.")
        sys.exit(1)
    polygon_key = settings.polygon_api_key.get_secret_value()

    min_date = datetime.date.fromisoformat(args.min_date)
    max_date = datetime.date.fromisoformat(args.max_date)

    log.info("Loading static datasets …")
    fear_greed_df = load_fear_greed()
    wsb_df = load_wsb()
    congress_df = load_congress()
    lobby_df = load_lobbying()
    contract_df = load_contracts()
    patent_df = load_patents()

    log.info("Loading IBKR execution history …")
    ibkr_df = (
        pd.DataFrame(columns=["ticker", "date", "ibkr_buy", "ibkr_sell", "ibkr_qty", "ibkr_price"])
        if args.no_ibkr
        else load_ibkr_executions(settings)
    )

    # Determine tickers to process
    all_tickers: set[str] = set()
    for df, col in [(congress_df, "ticker"), (lobby_df, "ticker"), (contract_df, "ticker"),
                    (patent_df, "ticker"), (wsb_df, "ticker"), (ibkr_df, "ticker")]:
        if col in df.columns:
            all_tickers.update(df[col].dropna().unique().tolist())

    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    else:
        tickers = sorted(all_tickers)

    # Cap ticker list to avoid multi-hour per-ticker API loops (EDGAR, Earnings, etc.)
    _TICKER_CAP = 300
    if len(tickers) > _TICKER_CAP:
        log.warning(
            "Ticker list has %d tickers; capping to %d for per-ticker API calls",
            len(tickers), _TICKER_CAP,
        )
        tickers = tickers[:_TICKER_CAP]

    log.info("%d tickers to process: %s …", len(tickers), ", ".join(tickers[:20]))

    log.info("Loading expanded signal datasets …")
    insider_df = load_insider_trades(tickers, min_date, max_date) if tickers else pd.DataFrame()
    macro_df = load_fred_macro(min_date, max_date)
    earnings_df = load_earnings_surprises(tickers, min_date, max_date) if tickers else pd.DataFrame()
    options_df = load_options_put_call_ratios(tickers, max_date) if tickers else pd.DataFrame()
    inst_df = load_13f_changes(tickers, min_date, max_date) if tickers else pd.DataFrame()
    short_interest_df = load_short_interest(tickers, min_date, max_date) if tickers else pd.DataFrame()

    # --- New data source loaders ---
    log.info("Fetching Yahoo Finance fundamentals …")
    yf_fundamentals_df = load_yfinance_fundamentals(tickers)
    log.info("Fetching Yahoo Finance options sentiment …")
    yf_options_df = load_yfinance_options_sentiment(tickers, max_date)
    log.info("Fetching Reddit multi-subreddit sentiment …")
    reddit_df = load_reddit_multi_subreddit(tickers, settings)
    log.info("Fetching Wikipedia pageviews …")
    wiki_df = load_wikipedia_pageviews(tickers, min_date, max_date) if tickers else pd.DataFrame()
    log.info("Fetching USASpending.gov contracts …")
    usa_spending_df = load_usaspending_contracts(tickers, min_date, max_date) if tickers else pd.DataFrame()
    log.info("Fetching CFTC Commitments of Traders …")
    cftc_df = load_cftc_positioning()
    log.info("Fetching CoinGecko crypto prices …")
    crypto_df = load_coingecko_crypto(min_date, max_date)
    log.info("Fetching SEC XBRL financial data …")
    xbrl_df = load_sec_xbrl_financials(tickers, settings)
    log.info("Fetching BLS macro data …")
    bls_key = settings.bls_api_key.get_secret_value() if settings.bls_api_key else None
    bls_df = load_bls_macro(min_date, max_date, api_key=bls_key)
    log.info("Fetching EIA energy data …")
    eia_key = settings.eia_api_key.get_secret_value() if settings.eia_api_key else None
    eia_df = load_eia_energy(min_date, max_date, api_key=eia_key)
    log.info("Fetching Alpha Vantage technical indicators …")
    av_key = settings.alpha_vantage_api_key.get_secret_value() if settings.alpha_vantage_api_key else None
    av_df = load_alpha_vantage_technicals(tickers, min_date, max_date, api_key=av_key) if tickers else pd.DataFrame()
    log.info("Fetching Google News sentiment …")
    gnews_df = load_google_news_sentiment(tickers) if tickers else pd.DataFrame()
    log.info("Fetching Stocktwits trader sentiment …")
    st_df = load_stocktwits_sentiment(tickers) if tickers else pd.DataFrame()
    log.info("Fetching Google Trends search interest …")
    gtrends_df = load_google_trends(tickers) if tickers else pd.DataFrame()
    log.info("Fetching CBOE VIX history …")
    vix_df = load_cboe_vix_history(min_date, max_date)
    log.info("Fetching extended FRED series …")
    fred_key = settings.fred_api_key.get_secret_value() if settings.fred_api_key else None
    fred_ext_df = load_fred_extended(min_date, max_date, api_key=fred_key)
    log.info("Fetching SEC 8-K material events …")
    sec_8k_df = load_sec_8k_events(tickers, min_date, max_date, settings) if tickers else pd.DataFrame()
    log.info("Fetching USD strength …")
    usd_df = load_usd_strength(min_date, max_date)
    log.info("Fetching Hacker News tech sentiment …")
    hn_df = load_hacker_news_tech_sentiment(tickers) if tickers else pd.DataFrame()
    log.info("Fetching PatentsView USPTO data …")
    pv_df = load_patentsview(tickers, min_date, max_date) if tickers else pd.DataFrame()
    log.info("Fetching World Bank macro data …")
    wb_df = load_world_bank_macro(min_date, max_date)
    log.info("Fetching OpenInsider cluster trades …")
    oi_df = load_open_insider(tickers, min_date, max_date) if tickers else pd.DataFrame()
    log.info("Fetching GDELT news tone …")
    gdelt_df = load_gdelt_news_tone(tickers, min_date, max_date) if tickers else pd.DataFrame()
    log.info("Fetching live news from RSS feeds …")
    news_notes = fetch_news_corpus_notes()
    if news_notes:
        news_path = Path(args.news_corpus_out)
        news_path.parent.mkdir(parents=True, exist_ok=True)
        news_path.write_text(
            "Live news headlines fetched " + datetime.date.today().isoformat() + ":\n\n"
            + "\n".join(f"- {n}" for n in news_notes),
            encoding="utf-8",
        )
        log.info("Wrote %d news items to %s", len(news_notes), news_path)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_written = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for ticker in tickers:
            log.info("Processing %s …", ticker)

            # Get all dates from the various signal sources for this ticker
            dates: set[datetime.date] = set()
            for df, col in [(congress_df, "ticker"), (lobby_df, "ticker"),
                            (contract_df, "ticker"), (patent_df, "ticker"),
                            (wsb_df, "ticker"), (ibkr_df, "ticker"),
                            (insider_df, "ticker"), (earnings_df, "ticker"),
                            (options_df, "ticker"), (inst_df, "ticker"),
                            (short_interest_df, "ticker")]:
                if col in df.columns and "date" in df.columns:
                    sub = df[df[col] == ticker]
                    dates.update(sub["date"].dropna().tolist())

            dates = {d for d in dates if min_date <= d <= max_date}
            if not dates:
                log.info("  No events for %s, skipping", ticker)
                continue

            # Fetch Polygon prices with forward HOLDING_DAYS buffer
            price_end = min(max_date + datetime.timedelta(days=HOLDING_DAYS * 2), datetime.date.today())
            prices = fetch_polygon_prices(ticker, min_date, price_end, polygon_key)
            if prices.empty:
                log.info("  No price data for %s, skipping", ticker)
                continue

            for as_of in sorted(dates):
                # Gather signals for this date
                fg = float(fear_greed_df.loc[as_of, "fear_greed"]) if as_of in fear_greed_df.index else 50.0

                def _get(df, ticker_col, ticker_val, date_col, date_val, val_cols):
                    sub = df[(df[ticker_col] == ticker_val) & (df[date_col] == date_val)]
                    if sub.empty:
                        return {c: 0 for c in val_cols}
                    return {c: sub.iloc[0][c] for c in val_cols}

                cong = _get(congress_df, "ticker", ticker, "date", as_of, ["congress_buy", "congress_sell", "congress_amount"])
                lob = _get(lobby_df, "ticker", ticker, "date", as_of, ["lobby_amount"])
                con = _get(contract_df, "ticker", ticker, "date", as_of, ["contract_amount"])
                pat = _get(patent_df, "ticker", ticker, "date", as_of, ["patent_count"])
                wsb = _get(wsb_df, "ticker", ticker, "date", as_of, ["wsb_sentiment", "wsb_mentions"])
                ibkr = _get(ibkr_df, "ticker", ticker, "date", as_of, ["ibkr_buy", "ibkr_sell", "ibkr_qty", "ibkr_price"])
                insider = _get(
                    insider_df, "ticker", ticker, "date", as_of,
                    [
                        "insider_buy_qty", "insider_sell_qty", "insider_net_qty",
                        "insider_value_usd", "insider_officer_count",
                        "insider_director_count",
                    ],
                )
                earnings = _get(
                    earnings_df, "ticker", ticker, "date", as_of,
                    ["eps_actual", "eps_estimate", "eps_surprise_pct"],
                )
                options = _get(
                    options_df, "ticker", ticker, "date", as_of,
                    ["put_call_ratio", "put_open_interest", "call_open_interest"],
                )
                inst = _get(
                    inst_df, "ticker", ticker, "date", as_of,
                    [
                        "institutional_delta_shares", "institutional_delta_pct",
                        "institutional_manager", "institutional_market_value_usd",
                    ],
                )
                short_interest = _get(
                    short_interest_df, "ticker", ticker, "date", as_of,
                    ["short_interest_shares", "days_to_cover", "short_interest_change_pct"],
                )
                macro = _macro_for_date(macro_df, as_of)
                price_ctx = _price_context(prices, as_of)

                # --- New data lookups ---
                # yfinance fundamentals (ticker-level, not date-specific)
                def _get_ticker_only(df: pd.DataFrame, ticker_val: str, cols: list[str]) -> dict:
                    if df.empty or "ticker" not in df.columns:
                        return {c: 0 for c in cols}
                    sub = df[df["ticker"] == ticker_val]
                    if sub.empty:
                        return {c: 0 for c in cols}
                    return {c: sub.iloc[0].get(c, 0) for c in cols}

                yf_fund = _get_ticker_only(yf_fundamentals_df, ticker, [
                    "pe_ratio", "forward_pe", "revenue_growth", "earnings_growth", "beta",
                    "analyst_upside_pct", "short_ratio", "profit_margin", "debt_to_equity",
                    "roe", "institutional_pct_held", "fifty_two_week_high_pct", "recommendation",
                ])
                yf_opt = _get(yf_options_df, "ticker", ticker, "date", as_of,
                              ["yf_put_call_ratio", "yf_implied_volatility_avg"])
                reddit = _get(reddit_df, "ticker", ticker, "date", as_of,
                              ["reddit_mentions", "reddit_sentiment_score"]) if not reddit_df.empty else {"reddit_mentions": 0, "reddit_sentiment_score": 0.0}
                wiki = _get(wiki_df, "ticker", ticker, "date", as_of,
                            ["wiki_pageviews"]) if not wiki_df.empty else {"wiki_pageviews": 0}
                usa = _get(usa_spending_df, "ticker", ticker, "date", as_of,
                           ["usa_contract_amount"]) if not usa_spending_df.empty else {"usa_contract_amount": 0.0}
                gnews = _get(gnews_df, "ticker", ticker, "date", as_of,
                             ["gnews_mentions", "gnews_sentiment"]) if not gnews_df.empty else {"gnews_mentions": 0, "gnews_sentiment": 0.0}
                xbrl = _get_ticker_only(xbrl_df, ticker, ["xbrl_revenue", "xbrl_net_income", "xbrl_gross_margin_pct"])
                av = _get(av_df, "ticker", ticker, "date", as_of,
                          ["av_rsi_14", "av_macd_signal"]) if not av_df.empty else {"av_rsi_14": 50.0, "av_macd_signal": 0.0}

                # Date-indexed macro lookups (forward-fill to nearest available date)
                def _macro_indexed(df: pd.DataFrame, col: str, dt: datetime.date) -> float:
                    if df.empty or col not in df.columns:
                        return 0.0
                    available = df.index[df.index <= dt]
                    if available.empty:
                        return 0.0
                    return float(df.loc[available[-1], col])

                cftc_val = _macro_indexed(cftc_df, "cftc_sp500_net_noncomm", as_of)
                btc_price_val = _macro_indexed(crypto_df, "btc_price", as_of)
                btc_7d_val = _macro_indexed(crypto_df, "btc_7d_change_pct", as_of)
                eth_price_val = _macro_indexed(crypto_df, "eth_price", as_of)
                bls_payrolls = _macro_indexed(bls_df, "bls_nonfarm_payrolls", as_of)
                bls_ur = _macro_indexed(bls_df, "bls_unemployment_rate", as_of)
                bls_ppi = _macro_indexed(bls_df, "bls_ppi_finished_goods", as_of)
                eia_wti = _macro_indexed(eia_df, "eia_crude_wti", as_of)
                eia_ng = _macro_indexed(eia_df, "eia_natural_gas", as_of)

                # Round 2 new source lookups
                st = _get(st_df, "ticker", ticker, "date", as_of,
                          ["st_bull_score", "st_total"]) if not st_df.empty else {"st_bull_score": 0.0, "st_total": 0}
                gt = _get(gtrends_df, "ticker", ticker, "date", as_of,
                          ["gtrends_interest"]) if not gtrends_df.empty else {"gtrends_interest": 0}
                vix_close_val = _macro_indexed(vix_df, "vix_close", as_of)
                vix_avg_val = _macro_indexed(vix_df, "vix_1m_avg", as_of)
                fred_hy = _macro_indexed(fred_ext_df, "fred_hy_spread", as_of)
                fred_m2 = _macro_indexed(fred_ext_df, "fred_m2_billions", as_of)
                fred_mtg = _macro_indexed(fred_ext_df, "fred_mortgage30", as_of)
                fred_10y = _macro_indexed(fred_ext_df, "fred_treasury_10y", as_of)
                sec8k = _get(sec_8k_df, "ticker", ticker, "date", as_of,
                             ["sec_8k_count"]) if not sec_8k_df.empty else {"sec_8k_count": 0}
                usd_idx = _macro_indexed(usd_df, "usd_strength_index", as_of)
                hn = _get(hn_df, "ticker", ticker, "date", as_of,
                          ["hn_hits", "hn_sentiment"]) if not hn_df.empty else {"hn_hits": 0, "hn_sentiment": 0.0}
                pv = _get(pv_df, "ticker", ticker, "date", as_of,
                          ["pv_patent_count"]) if not pv_df.empty else {"pv_patent_count": 0}
                wb_gdp = _macro_indexed(wb_df, "wb_us_gdp_growth", as_of)
                wb_inf = _macro_indexed(wb_df, "wb_us_inflation", as_of)
                oi = _get(oi_df, "ticker", ticker, "date", as_of,
                          ["oi_buy_count", "oi_sell_count", "oi_net_value"]) if not oi_df.empty else {"oi_buy_count": 0, "oi_sell_count": 0, "oi_net_value": 0.0}
                gdelt = _get(gdelt_df, "ticker", ticker, "date", as_of,
                             ["gdelt_avg_tone", "gdelt_article_count"]) if not gdelt_df.empty else {"gdelt_avg_tone": 0.0, "gdelt_article_count": 0}

                bundle = build_signal_bundle(
                    ticker, as_of,
                    congress_buy=int(cong["congress_buy"]),
                    congress_sell=int(cong["congress_sell"]),
                    congress_amount=float(cong["congress_amount"]),
                    lobby_amount=float(lob["lobby_amount"]),
                    contract_amount=float(con["contract_amount"]),
                    patent_count=int(pat["patent_count"]),
                    wsb_sentiment=float(wsb["wsb_sentiment"]),
                    wsb_mentions=int(wsb["wsb_mentions"]),
                    fear_greed=fg,
                    ibkr_buy=int(ibkr["ibkr_buy"]),
                    ibkr_sell=int(ibkr["ibkr_sell"]),
                    ibkr_qty=float(ibkr["ibkr_qty"]),
                    ibkr_price=float(ibkr["ibkr_price"]),
                    insider_buy_qty=float(insider["insider_buy_qty"]),
                    insider_sell_qty=float(insider["insider_sell_qty"]),
                    insider_net_qty=float(insider["insider_net_qty"]),
                    insider_value_usd=float(insider["insider_value_usd"]),
                    insider_officer_count=int(insider["insider_officer_count"]),
                    insider_director_count=int(insider["insider_director_count"]),
                    yield_spread_2_10=macro.get("yield_spread_2_10"),
                    cpi_mom=macro.get("cpi_mom"),
                    ism_pmi=macro.get("ism_pmi"),
                    unemployment_claims=macro.get("unemployment_claims"),
                    fed_funds_rate=macro.get("fed_funds_rate"),
                    eps_actual=float(earnings["eps_actual"]),
                    eps_estimate=float(earnings["eps_estimate"]),
                    eps_surprise_pct=float(earnings["eps_surprise_pct"]),
                    put_call_ratio=float(options["put_call_ratio"]),
                    put_open_interest=float(options["put_open_interest"]),
                    call_open_interest=float(options["call_open_interest"]),
                    price_level=str(price_ctx["price_level"]),
                    price_momentum_20d=float(price_ctx["price_momentum_20d"]),
                    institutional_delta_shares=float(inst["institutional_delta_shares"]),
                    institutional_delta_pct=float(inst["institutional_delta_pct"]),
                    institutional_manager=str(inst["institutional_manager"]),
                    institutional_market_value_usd=float(inst["institutional_market_value_usd"]),
                    short_interest_shares=float(short_interest["short_interest_shares"]),
                    days_to_cover=float(short_interest["days_to_cover"]),
                    short_interest_change_pct=float(short_interest["short_interest_change_pct"]),
                    # Round 1 new signals
                    yf_pe_ratio=float(yf_fund.get("pe_ratio") or 0.0),
                    yf_forward_pe=float(yf_fund.get("forward_pe") or 0.0),
                    yf_revenue_growth=float(yf_fund.get("revenue_growth") or 0.0),
                    yf_earnings_growth=float(yf_fund.get("earnings_growth") or 0.0),
                    yf_beta=float(yf_fund.get("beta") or 1.0),
                    yf_analyst_upside_pct=float(yf_fund.get("analyst_upside_pct") or 0.0),
                    yf_short_ratio=float(yf_fund.get("short_ratio") or 0.0),
                    yf_profit_margin=float(yf_fund.get("profit_margin") or 0.0),
                    yf_debt_to_equity=float(yf_fund.get("debt_to_equity") or 0.0),
                    yf_roe=float(yf_fund.get("roe") or 0.0),
                    yf_institutional_pct_held=float(yf_fund.get("institutional_pct_held") or 0.0),
                    yf_fifty_two_week_high_pct=float(yf_fund.get("fifty_two_week_high_pct") or 0.0),
                    yf_recommendation=str(yf_fund.get("recommendation") or "none"),
                    yf_put_call_ratio=float(yf_opt.get("yf_put_call_ratio") or 0.0),
                    yf_implied_volatility_avg=float(yf_opt.get("yf_implied_volatility_avg") or 0.0),
                    wiki_pageviews=int(wiki.get("wiki_pageviews") or 0),
                    reddit_mentions=int(reddit.get("reddit_mentions") or 0),
                    reddit_sentiment_score=float(reddit.get("reddit_sentiment_score") or 0.0),
                    gnews_mentions=int(gnews.get("gnews_mentions") or 0),
                    gnews_sentiment=float(gnews.get("gnews_sentiment") or 0.0),
                    btc_price=btc_price_val,
                    btc_7d_change_pct=btc_7d_val,
                    eth_price=eth_price_val,
                    cftc_sp500_net_noncomm=cftc_val,
                    bls_nonfarm_payrolls=bls_payrolls,
                    bls_unemployment_rate=bls_ur,
                    bls_ppi_finished_goods=bls_ppi,
                    eia_crude_wti=eia_wti,
                    eia_natural_gas=eia_ng,
                    av_rsi_14=float(av.get("av_rsi_14") or 50.0),
                    av_macd_signal=float(av.get("av_macd_signal") or 0.0),
                    xbrl_revenue=float(xbrl.get("xbrl_revenue") or 0.0),
                    xbrl_net_income=float(xbrl.get("xbrl_net_income") or 0.0),
                    xbrl_gross_margin_pct=float(xbrl.get("xbrl_gross_margin_pct") or 0.0),
                    usa_contract_amount=float(usa.get("usa_contract_amount") or 0.0),
                    # Round 2 new signals
                    st_bull_score=float(st.get("st_bull_score") or 0.0),
                    st_total=int(st.get("st_total") or 0),
                    gtrends_interest=int(gt.get("gtrends_interest") or 0),
                    vix_close=vix_close_val,
                    vix_1m_avg=vix_avg_val,
                    fred_hy_spread=fred_hy,
                    fred_m2_billions=fred_m2,
                    fred_mortgage30=fred_mtg,
                    fred_treasury_10y=fred_10y,
                    sec_8k_count=int(sec8k.get("sec_8k_count") or 0),
                    usd_strength_index=usd_idx,
                    hn_hits=int(hn.get("hn_hits") or 0),
                    hn_sentiment=float(hn.get("hn_sentiment") or 0.0),
                    pv_patent_count=int(pv.get("pv_patent_count") or 0),
                    wb_us_gdp_growth=wb_gdp,
                    wb_us_inflation=wb_inf,
                    oi_buy_count=int(oi.get("oi_buy_count") or 0),
                    oi_sell_count=int(oi.get("oi_sell_count") or 0),
                    oi_net_value=float(oi.get("oi_net_value") or 0.0),
                    gdelt_avg_tone=float(gdelt.get("gdelt_avg_tone") or 0.0),
                    gdelt_article_count=int(gdelt.get("gdelt_article_count") or 0),
                )

                plan = build_trade_plan(bundle, as_of)
                pnl = compute_forward_return(prices, as_of, plan.holding_period_days)
                if pnl is None:
                    continue  # not enough forward price data

                example = LocalTrainingExample(
                    signal_bundle=bundle,
                    trade_plan=plan,
                    pnl_pct=round(pnl, 6),
                    metadata={
                        "source": "ingest_training_data",
                        "ticker": ticker,
                        "as_of": as_of.isoformat(),
                        "congress_buy": int(cong["congress_buy"]),
                        "congress_sell": int(cong["congress_sell"]),
                        "lobby_amount": float(lob["lobby_amount"]),
                        "contract_amount": float(con["contract_amount"]),
                        "patent_count": int(pat["patent_count"]),
                        "wsb_mentions": int(wsb["wsb_mentions"]),
                        "fear_greed": fg,
                        "forward_pnl_pct": round(pnl, 6),
                        "ibkr_buy": int(ibkr["ibkr_buy"]),
                        "ibkr_sell": int(ibkr["ibkr_sell"]),
                        "ibkr_qty": float(ibkr["ibkr_qty"]),
                        "ibkr_price": float(ibkr["ibkr_price"]),
                        "insider_buy_qty": float(insider["insider_buy_qty"]),
                        "insider_sell_qty": float(insider["insider_sell_qty"]),
                        "insider_net_qty": float(insider["insider_net_qty"]),
                        "insider_value_usd": float(insider["insider_value_usd"]),
                        "eps_surprise_pct": float(earnings["eps_surprise_pct"]),
                        "put_call_ratio": float(options["put_call_ratio"]),
                        "yield_spread_2_10": macro.get("yield_spread_2_10", 0.0),
                        "ism_pmi": macro.get("ism_pmi", 0.0),
                        "institutional_delta_shares": float(inst["institutional_delta_shares"]),
                        "institutional_delta_pct": float(inst["institutional_delta_pct"]),
                        "days_to_cover": float(short_interest["days_to_cover"]),
                        "short_interest_change_pct": float(short_interest["short_interest_change_pct"]),
                        "price_momentum_20d": float(price_ctx["price_momentum_20d"]),
                        # New metadata
                        "yf_revenue_growth": float(yf_fund.get("revenue_growth") or 0.0),
                        "yf_earnings_growth": float(yf_fund.get("earnings_growth") or 0.0),
                        "yf_analyst_upside_pct": float(yf_fund.get("analyst_upside_pct") or 0.0),
                        "yf_recommendation": str(yf_fund.get("recommendation") or "none"),
                        "yf_short_ratio": float(yf_fund.get("short_ratio") or 0.0),
                        "yf_put_call_ratio": float(yf_opt.get("yf_put_call_ratio") or 0.0),
                        "wiki_pageviews": int(wiki.get("wiki_pageviews") or 0),
                        "reddit_mentions": int(reddit.get("reddit_mentions") or 0),
                        "reddit_sentiment_score": float(reddit.get("reddit_sentiment_score") or 0.0),
                        "gnews_mentions": int(gnews.get("gnews_mentions") or 0),
                        "gnews_sentiment": float(gnews.get("gnews_sentiment") or 0.0),
                        "btc_7d_change_pct": btc_7d_val,
                        "cftc_sp500_net_noncomm": cftc_val,
                        "bls_nonfarm_payrolls": bls_payrolls,
                        "bls_unemployment_rate": bls_ur,
                        "eia_crude_wti": eia_wti,
                        "av_rsi_14": float(av.get("av_rsi_14") or 50.0),
                        "xbrl_gross_margin_pct": float(xbrl.get("xbrl_gross_margin_pct") or 0.0),
                        "usa_contract_amount": float(usa.get("usa_contract_amount") or 0.0),
                        # Round 2 metadata
                        "st_bull_score": float(st.get("st_bull_score") or 0.0),
                        "st_total": int(st.get("st_total") or 0),
                        "gtrends_interest": int(gt.get("gtrends_interest") or 0),
                        "vix_close": vix_close_val,
                        "fred_hy_spread": fred_hy,
                        "fred_mortgage30": fred_mtg,
                        "fred_treasury_10y": fred_10y,
                        "sec_8k_count": int(sec8k.get("sec_8k_count") or 0),
                        "usd_strength_index": usd_idx,
                        "hn_hits": int(hn.get("hn_hits") or 0),
                        "pv_patent_count": int(pv.get("pv_patent_count") or 0),
                        "wb_us_gdp_growth": wb_gdp,
                        "oi_buy_count": int(oi.get("oi_buy_count") or 0),
                        "oi_sell_count": int(oi.get("oi_sell_count") or 0),
                        "gdelt_avg_tone": float(gdelt.get("gdelt_avg_tone") or 0.0),
                    },
                )
                fh.write(example.model_dump_json() + "\n")
                total_written += 1

    log.info("Done. Wrote %d training examples to %s", total_written, out_path)


if __name__ == "__main__":
    main()
