"""Publisher provenance: families, source tiers, and propaganda risk.

Corroboration must count NEWSROOMS, not feed labels. One publisher often
ships under several feed labels ("Reuters World", "Reuters Business",
"Reuters via Google News") — counting labels lets a wire corroborate
itself, which inflates every corroboration-weighted score downstream.

Tier vocabulary (1 best):
  1 wire services / official government sources — fastest, most authoritative
  2 major established outlets
  3 specialty / regional outlets
  4 aggregators, blogs, unknown sources (FAIL-CLOSED default)

Propaganda risk is fail-closed too: an unreviewed source scores as
"unknown" (0.35 weight), never as trustworthy.
"""

from __future__ import annotations

import re

_FAMILY_SUFFIXES = re.compile(
    r"\s+(world|us|u\.s\.|uk|business|markets?|finance|financial|money|news|"
    r"top\s+stories|breaking|live|latest|wire|via\s+google\s+news)$",
    re.IGNORECASE,
)

# Explicit family assignments for sources whose label does not reduce to
# the family via suffix-stripping alone.
_EXPLICIT_FAMILIES: dict[str, str] = {
    "wall street journal": "wsj",
    "wsj": "wsj",
    "dow jones": "dowjones",
    "dow jones newswires": "dowjones",
    "marketwatch": "marketwatch",
    "the associated press": "ap",
    "associated press": "ap",
    "ap": "ap",
    "ap news": "ap",
    "financial times": "ft",
    "ft": "ft",
    "the new york times": "nyt",
    "new york times": "nyt",
    "nyt": "nyt",
    "the guardian": "guardian",
    "guardian": "guardian",
    "yahoo finance": "yahoo",
    "yahoo": "yahoo",
    "cnbc": "cnbc",
    "bbc": "bbc",
    "bbc news": "bbc",
    "reuters": "reuters",
    "thomson reuters": "reuters",
    "bloomberg": "bloomberg",
    "seeking alpha": "seekingalpha",
    "pr newswire": "prnewswire",
    "globenewswire": "globenewswire",
    "business wire": "businesswire",
    "federal reserve": "federalreserve",
    "u.s. federal reserve": "federalreserve",
    "sec": "sec",
    "u.s. securities and exchange commission": "sec",
    "securities and exchange commission": "sec",
}

_SOURCE_TIERS: dict[str, int] = {
    # Tier 1 — wires and official/primary sources
    "reuters": 1,
    "ap": 1,
    "bloomberg": 1,
    "dowjones": 1,
    "federalreserve": 1,
    "sec": 1,
    "prnewswire": 1,  # primary-source corporate disclosure wire
    "globenewswire": 1,
    "businesswire": 1,
    # Tier 2 — major outlets
    "wsj": 2,
    "ft": 2,
    "nyt": 2,
    "cnbc": 2,
    "bbc": 2,
    "guardian": 2,
    "marketwatch": 2,
    # Tier 3 — specialty
    "yahoo": 3,
    "seekingalpha": 3,
    "barrons": 3,
    "investors": 3,
    "fortune": 3,
    "forbes": 3,
    "businessinsider": 3,
}

# Publisher families with a known state-affiliation / propaganda concern.
# Everything not listed is "unknown" — fail closed, never fail open.
_HIGH_RISK_FAMILIES: frozenset[str] = frozenset(
    {"rt", "sputnik", "xinhua", "globaltimes", "presstv", "tass", "cgtn"}
)
_LOW_RISK_FAMILIES: frozenset[str] = frozenset(_SOURCE_TIERS)


def publisher_family(source_name: str) -> str:
    """Collapse a feed label to its publisher family key."""

    name = source_name.strip().lower()
    if not name:
        return "unknown"
    if name in _EXPLICIT_FAMILIES:
        return _EXPLICIT_FAMILIES[name]
    previous = None
    while previous != name:
        previous = name
        name = _FAMILY_SUFFIXES.sub("", name).strip()
        if name in _EXPLICIT_FAMILIES:
            return _EXPLICIT_FAMILIES[name]
    return re.sub(r"[^a-z0-9]+", "", name) or "unknown"


def count_publisher_families(source_names: list[str]) -> int:
    return len({publisher_family(name) for name in source_names if name.strip()}) or 1


def source_tier(source_name: str) -> int:
    """Tier for a source label. Unknown sources are tier 4 — fail closed."""

    return _SOURCE_TIERS.get(publisher_family(source_name), 4)


def propaganda_risk(source_name: str) -> str:
    """'low' | 'high' | 'unknown'. Only reviewed families get 'low'."""

    family = publisher_family(source_name)
    if family in _HIGH_RISK_FAMILIES:
        return "high"
    if family in _LOW_RISK_FAMILIES:
        return "low"
    return "unknown"
