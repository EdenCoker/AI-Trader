"""Ticker extraction from headlines.

Two signals, both high-precision by construction:

1. Cashtags — ``$`` + 1-5 uppercase letters, word-bounded ($AAPL). Any
   well-formed cashtag is accepted: it is explicit author intent.
   Lowercase ($aapl) is dropped, not normalized — mixed case is far more
   often a typo or a price tag.
2. Company names — whole-word, case-insensitive matches against a curated
   name→symbol dictionary, longest name first.

Bare symbols WITHOUT ``$`` are deliberately never matched: GM, ALL, IT, V
and friends are ordinary English words, and a false ticker routes a story
into the wrong SignalBundle — the worst failure mode this module can have.
Names that are also common words (Apple, Visa, Meta...) require a cashtag
or a disambiguating context word to tag.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

MAX_TICKERS = 8

_CASHTAG_RE = re.compile(r"(?<![A-Za-z0-9$])\$([A-Z]{1,5})(?![A-Za-z0-9])")

# name (lowercase) -> symbol. Curated for the large-cap universe this repo
# trades; extend via the `extra_names` argument (e.g. from a watchlist).
_COMPANY_NAMES: dict[str, str] = {
    "microsoft": "MSFT",
    "nvidia": "NVDA",
    "tesla": "TSLA",
    "netflix": "NFLX",
    "broadcom": "AVGO",
    "berkshire hathaway": "BRK.B",
    "eli lilly": "LLY",
    "jpmorgan": "JPM",
    "jp morgan": "JPM",
    "goldman sachs": "GS",
    "morgan stanley": "MS",
    "bank of america": "BAC",
    "wells fargo": "WFC",
    "exxon": "XOM",
    "exxonmobil": "XOM",
    "chevron": "CVX",
    "johnson & johnson": "JNJ",
    "pfizer": "PFE",
    "moderna": "MRNA",
    "merck": "MRK",
    "abbvie": "ABBV",
    "unitedhealth": "UNH",
    "walmart": "WMT",
    "costco": "COST",
    "home depot": "HD",
    "mcdonald's": "MCD",
    "mcdonalds": "MCD",
    "starbucks": "SBUX",
    "nike": "NKE",
    "disney": "DIS",
    "comcast": "CMCSA",
    "verizon": "VZ",
    "at&t": "T",
    "intel": "INTC",
    "advanced micro devices": "AMD",
    "qualcomm": "QCOM",
    "texas instruments": "TXN",
    "micron": "MU",
    "salesforce": "CRM",
    "adobe": "ADBE",
    "palantir": "PLTR",
    "snowflake": "SNOW",
    "coinbase": "COIN",
    "paypal": "PYPL",
    "mastercard": "MA",
    "american express": "AXP",
    "boeing": "BA",
    "airbus": "EADSY",
    "lockheed martin": "LMT",
    "raytheon": "RTX",
    "northrop grumman": "NOC",
    "general electric": "GE",
    "general motors": "GM",
    "ford": "F",
    "caterpillar": "CAT",
    "deere": "DE",
    "john deere": "DE",
    "3m": "MMM",
    "honeywell": "HON",
    "united airlines": "UAL",
    "delta air lines": "DAL",
    "american airlines": "AAL",
    "southwest airlines": "LUV",
    "fedex": "FDX",
    "united parcel service": "UPS",
    "activision blizzard": "ATVI",
    "uber": "UBER",
    "lyft": "LYFT",
    "airbnb": "ABNB",
    "shopify": "SHOP",
    "spotify": "SPOT",
    "zoom": "ZM",
    "robinhood": "HOOD",
    "gamestop": "GME",
    "amc entertainment": "AMC",
    "occidental petroleum": "OXY",
    "conocophillips": "COP",
    "schlumberger": "SLB",
    "freeport-mcmoran": "FCX",
    "newmont": "NEM",
    "blackrock": "BLK",
    "charles schwab": "SCHW",
    "citigroup": "C",
    "citi": "C",
}

# Names that are ordinary English words or too generic — a bare-name match
# would tag unrelated stories ("Amazon rainforest" → AMZN, "Visa
# restrictions" → V, "meta-analysis" → META). These need a cashtag OR the
# name plus a market-context word within the headline.
_AMBIGUOUS_NAMES: dict[str, str] = {
    "apple": "AAPL",
    "amazon": "AMZN",
    "alphabet": "GOOGL",
    "google": "GOOGL",
    "meta": "META",
    "visa": "V",
    "oracle": "ORCL",
    "target": "TGT",
    "gap": "GAP",
    "alcoa": "AA",
}

_CONTEXT_WORDS_RE = re.compile(
    r"\b(shares?|stock|earnings|revenue|profit|quarter|guidance|forecast|ceo|cfo|"
    r"ipo|dividend|market cap|investors?|nasdaq|nyse|wall street|price target|"
    r"upgrade[ds]?|downgrade[ds]?|beats?|misses?|acquisition|merger|buyback)\b",
    re.IGNORECASE,
)


def _name_pattern(names: Mapping[str, str]) -> re.Pattern[str] | None:
    if not names:
        return None
    parts = sorted((re.escape(name) for name in names), key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(parts) + r")\b", re.IGNORECASE)


_UNAMBIGUOUS_RE = _name_pattern(_COMPANY_NAMES)
_AMBIGUOUS_RE = _name_pattern(_AMBIGUOUS_NAMES)


def extract_tickers(
    text: str,
    extra_names: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Extract tickers: uppercase, deduped, first-occurrence order,
    capped at MAX_TICKERS."""

    if not text:
        return ()

    found: list[str] = []
    seen: set[str] = set()

    def add(symbol: str) -> None:
        symbol = symbol.upper()
        if symbol not in seen and len(found) < MAX_TICKERS:
            seen.add(symbol)
            found.append(symbol)

    for match in _CASHTAG_RE.finditer(text):
        add(match.group(1))

    if _UNAMBIGUOUS_RE is not None:
        for match in _UNAMBIGUOUS_RE.finditer(text):
            add(_COMPANY_NAMES[match.group(1).lower()])

    if _AMBIGUOUS_RE is not None and _CONTEXT_WORDS_RE.search(text):
        for match in _AMBIGUOUS_RE.finditer(text):
            add(_AMBIGUOUS_NAMES[match.group(1).lower()])

    if extra_names:
        extra_pattern = _name_pattern(extra_names)
        if extra_pattern is not None:
            lowered = {name.lower(): symbol for name, symbol in extra_names.items()}
            for match in extra_pattern.finditer(text):
                add(lowered[match.group(1).lower()])

    return tuple(found)
