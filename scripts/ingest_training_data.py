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

XL = {"engine": "calamine"}  # fast xlsx reader


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


def _quiver_paginate(url: str, headers: dict, params: dict | None = None, page_size: int = 500) -> list[dict]:
    """Paginate through a Quiver endpoint, returning all records."""
    all_rows: list[dict] = []
    page = 1
    while True:
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
        for filing in filings:
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
# News fetch via RSS
# ---------------------------------------------------------------------------

def fetch_news_corpus_notes() -> list[str]:
    """Pull headlines from RSS feeds and return as plain-text notes."""
    notes: list[str] = []
    for url in RSS_FEEDS:
        try:
            r = httpx.get(url, timeout=15, headers={"User-Agent": "AI-Trader research edenjcokeer@gmail.com"})
            r.raise_for_status()
            import xml.etree.ElementTree as ET
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
# Main ingestion pipeline
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

    log.info("%d tickers to process: %s …", len(tickers), ", ".join(tickers[:20]))

    log.info("Loading expanded signal datasets …")
    insider_df = load_insider_trades(tickers, min_date, max_date) if tickers else pd.DataFrame()
    macro_df = load_fred_macro(min_date, max_date)
    earnings_df = load_earnings_surprises(tickers, min_date, max_date) if tickers else pd.DataFrame()
    options_df = load_options_put_call_ratios(tickers, max_date) if tickers else pd.DataFrame()
    inst_df = load_13f_changes(tickers, min_date, max_date) if tickers else pd.DataFrame()
    short_interest_df = load_short_interest(tickers, min_date, max_date) if tickers else pd.DataFrame()

    # Fetch news and write to RAG corpus
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
                    },
                )
                fh.write(example.model_dump_json() + "\n")
                total_written += 1

    log.info("Done. Wrote %d training examples to %s", total_written, out_path)


if __name__ == "__main__":
    main()
