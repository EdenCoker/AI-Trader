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
    # "3M" is deliberately absent: lowercase "3m" is almost always an
    # abbreviation for 3 million/meters/months; use the $MMM cashtag.
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
    "zoom": "ZM",
}

# Adjacency gate for ambiguous names. Mere co-occurrence of a context word
# anywhere in the headline is self-defeating ("Walmart price target raised"
# contains "target"; "Gap between rich and poor widens" plus any market
# noun would tag GAP) — the ambiguous name itself must sit directly
# against company-shaped context: "Target shares", "Target beats
# estimates", "shares of Target", "Target's CEO", "Target Corp".
_ADJACENT_AFTER = (
    r"(?:'s)?\s+(?:shares?|stock|earnings|revenue|profits?|guidance|forecast|"
    r"outlook|ceo|cfo|dividend|buyback|q[1-4]|quarterly|"
    r"beats?|misses?|tops?|posts?|reports?|raises?|cuts?|lowers?|jumps?|"
    r"falls?|surges?|slides?|rallies|corp|inc)\b"
)
_ADJACENT_BEFORE = r"(?:shares?\s+of|stake\s+in|owner\s+of|retailer|chipmaker)\s+"



def _name_pattern(names: Mapping[str, str]) -> re.Pattern[str] | None:
    if not names:
        return None
    parts = sorted((re.escape(name) for name in names), key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(parts) + r")\b", re.IGNORECASE)


_UNAMBIGUOUS_RE = _name_pattern(_COMPANY_NAMES)
_AMBIGUOUS_ADJACENT_RES: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (
        re.compile(
            r"(?:\b" + re.escape(name) + _ADJACENT_AFTER
            + r"|" + _ADJACENT_BEFORE + re.escape(name) + r"\b)",
            re.IGNORECASE,
        ),
        symbol,
    )
    for name, symbol in _AMBIGUOUS_NAMES.items()
)


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

    for pattern, symbol in _AMBIGUOUS_ADJACENT_RES:
        if pattern.search(text):
            add(symbol)

    if extra_names:
        extra_pattern = _name_pattern(extra_names)
        if extra_pattern is not None:
            lowered = {name.lower(): symbol for name, symbol in extra_names.items()}
            for match in extra_pattern.finditer(text):
                add(lowered[match.group(1).lower()])

    return tuple(found)
