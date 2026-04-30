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
import json
import logging
import sys
import time
from pathlib import Path

import httpx
import pandas as pd

# Ensure src/ is on the path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ai_trader.config import get_settings
from ai_trader.domain.signals import Signal, SignalBundle, SignalDirection
from ai_trader.intelligence.trade_plan import TradePlan
from ai_trader.training.data import LocalTrainingExample

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("ingest")

TRAINING_DATA = Path("Training Data")
POLYGON_BASE = "https://api.polygon.io/v2/aggs/ticker"
RSS_FEEDS = [
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.bbci.co.uk/news/business/rss.xml",
    "https://rss.ft.com/rss/companies",
]
HOLDING_DAYS = 30  # forward return window for pnl_pct


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


def _fetch_congress_from_api() -> pd.DataFrame:
    """Fallback: pull congressional trades from Quiver API."""
    settings = get_settings()
    if settings.quiver_api_key is None:
        log.warning("No QUIVER_API_KEY — congressional trades unavailable")
        return pd.DataFrame(columns=["ticker", "date", "congress_buy", "congress_sell", "congress_amount"])
    try:
        r = httpx.get(
            "https://api.quiverquant.com/beta/bulk/congresstrading",
            headers={"Authorization": f"Bearer {settings.quiver_api_key.get_secret_value()}"},
            timeout=30,
        )
        r.raise_for_status()
        rows = r.json()
        log.info("Quiver API returned %d congressional trade rows", len(rows))
        records = []
        for item in rows:
            ticker = str(item.get("Ticker") or item.get("ticker") or "").upper()
            raw_date = item.get("FiledAfterDate") or item.get("ReportDate") or item.get("FiledDate")
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
        df = pd.DataFrame(records)
        if df.empty:
            return pd.DataFrame(columns=["ticker", "date", "congress_buy", "congress_sell", "congress_amount"])
        return df.groupby(["ticker", "date"]).agg(
            congress_buy=("congress_buy", "sum"),
            congress_sell=("congress_sell", "sum"),
            congress_amount=("congress_amount", "sum"),
        ).reset_index()
    except Exception as exc:
        log.warning("Quiver API failed: %s", exc)
        return pd.DataFrame(columns=["ticker", "date", "congress_buy", "congress_sell", "congress_amount"])


def load_lobbying() -> pd.DataFrame:
    """Return per-ticker/date lobbying amount aggregated."""
    log.info("  loading lobbying-recent.xlsx …")
    df = pd.read_excel(TRAINING_DATA / "lobbying-recent.xlsx", parse_dates=["Date"], **XL)
    df = df.rename(columns={"Ticker": "ticker", "Date": "date", "Amount": "lobby_amount"})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["ticker"] = df["ticker"].astype(str).str.upper()
    out = df.groupby(["ticker", "date"]).agg(lobby_amount=("lobby_amount", "sum")).reset_index()
    return out


def load_contracts() -> pd.DataFrame:
    """Return per-ticker/date government contract amount aggregated."""
    log.info("  loading contracts-recent.xlsx …")
    df = pd.read_excel(TRAINING_DATA / "contracts-recent.xlsx", parse_dates=["Date"], **XL)
    df = df.rename(columns={"Ticker": "ticker", "Date": "date", "Amount": "contract_amount"})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["ticker"] = df["ticker"].astype(str).str.upper()
    out = df.groupby(["ticker", "date"]).agg(contract_amount=("contract_amount", "sum")).reset_index()
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

    return SignalBundle(ticker=ticker, as_of=as_of, signals=tuple(signals))


def build_trade_plan(bundle: SignalBundle, as_of: datetime.date) -> TradePlan:
    direction = bundle.direction
    conviction = round(bundle.conviction, 4)
    size_mult = round(min(1.0 + conviction, 2.0), 2)
    if direction == SignalDirection.NEUTRAL:
        size_mult = 0.0
    exit_trigger = "Exit if price falls more than 8% below entry or holding period expires."
    return TradePlan(
        ticker=bundle.ticker,
        as_of=as_of,
        direction=direction,
        conviction=conviction,
        size_multiplier=size_mult,
        holding_period_days=HOLDING_DAYS,
        exit_trigger=exit_trigger,
    )


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

    # Determine tickers to process
    all_tickers: set[str] = set()
    for df, col in [(congress_df, "ticker"), (lobby_df, "ticker"), (contract_df, "ticker"),
                    (patent_df, "ticker"), (wsb_df, "ticker")]:
        if col in df.columns:
            all_tickers.update(df[col].dropna().unique().tolist())

    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    else:
        tickers = sorted(all_tickers)

    log.info("%d tickers to process: %s …", len(tickers), ", ".join(tickers[:20]))

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
                            (contract_df, "ticker"), (patent_df, "ticker"), (wsb_df, "ticker")]:
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
                pnl = compute_forward_return(prices, as_of, HOLDING_DAYS)
                if pnl is None:
                    continue  # not enough forward price data

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
                )

                plan = build_trade_plan(bundle, as_of)

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
                    },
                )
                fh.write(example.model_dump_json() + "\n")
                total_written += 1

    log.info("Done. Wrote %d training examples to %s", total_written, out_path)


if __name__ == "__main__":
    main()
