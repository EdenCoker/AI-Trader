"""Deterministic finance-event classification for headlines.

Maps a headline (plus optional snippet) to an event category with a
severity (0-100, how market-moving) and a polarity (-1/0/+1, expected
price direction for the SUBJECT company). Keyword-driven, word-bounded,
case-insensitive — no LLM in the loop, so scoring stays reproducible and
auditable. An optional LLM pass can refine categories later, but it is
capped (see ``cap_llm_upgrade``) so a hallucinated "critical" cannot leak
into position sizing.

Polarity conventions worth stating out loud:
- M&A is scored for the TARGET (targets pop; acquirers drift). The signal
  layer records this asymmetry in metadata so a reasoner can invert it.
- Layoffs are polarity 0: markets often read cost cuts positively while
  the fundamental read is negative; a neutral polarity with real severity
  surfaces the story without pretending we know the sign.
"""

from __future__ import annotations

import re

from ai_trader.news.models import EventClassification

# (category, severity, polarity, keywords)
# First match wins within a severity band; bands are evaluated from most
# severe down so "bankruptcy" beats "files" noise.
_RULES: tuple[tuple[str, int, int, tuple[str, ...]], ...] = (
    # --- critical (100) ---
    ("bankruptcy", 100, -1, (
        "bankruptcy", "chapter 11", "chapter 7", "insolvency", "insolvent",
        "goes bust", "liquidation",
    )),
    ("default", 100, -1, ("defaults on", "debt default", "misses payment", "missed bond payment")),
    ("fraud", 100, -1, (
        "fraud", "accounting scandal", "embezzlement", "ponzi",
        "falsified", "criminal charges",
    )),
    ("delisting", 100, -1, ("delisted", "delisting", "trading halted", "trading halt")),
    ("going_concern", 100, -1, ("going concern",)),
    # --- high (75) ---
    ("investigation", 75, -1, (
        "sec investigation", "sec probe", "doj investigation", "doj probe",
        "under investigation", "subpoena", "antitrust probe", "antitrust investigation",
        "criminal investigation", "regulators probe", "ftc probe", "ftc investigation",
        "opens probe", "opens investigation", "faces probe", "faces investigation",
    )),
    ("lawsuit", 75, -1, ("class action", "sues", "sued", "lawsuit")),
    ("guidance_cut", 75, -1, (
        r"re:(?:cuts?|lowers?|slashe?s?|trims?|withdraws?|reduces?)\s+(?:[\w-]+\s+){0,3}"
        r"(?:guidance|forecast|outlook)",
        "guidance cut", "profit warning", "warns on profit",
    )),
    ("short_report", 75, -1, ("short seller", "short-seller", "short report")),
    ("recall", 75, -1, ("recalls", "recall of", "safety recall")),
    ("data_breach", 75, -1, ("data breach", "cyberattack", "cyber attack", "ransomware", "hacked")),
    ("ceo_exit", 75, -1, (
        "ceo resigns", "ceo steps down", "ceo departs", "ceo fired",
        "ceo ousted", "cfo resigns", "cfo steps down",
    )),
    ("credit_downgrade", 75, -1, (
        "downgraded to junk", "credit rating cut", "credit downgrade", "rating downgraded",
    )),
    ("fda_rejection", 75, -1, (
        "fda rejects", "complete response letter", "crl", "trial fails",
        "trial failure", "fails phase",
    )),
    ("mna", 75, 1, (
        "acquires", "to acquire", "acquisition of", "merger", "merges with", "to merge",
        "takeover bid", "buyout offer", "to buy", "agrees to buy", "acquisition talks",
    )),
    ("guidance_raise", 75, 1, (
        r"re:(?:raises?|lifts?|boosts?|hikes?|increases?)\s+(?:[\w-]+\s+){0,3}"
        r"(?:guidance|forecast|outlook)",
        "guidance raised",
    )),
    ("fda_approval", 75, 1, ("fda approves", "fda approval", "wins approval", "approval granted")),
    ("activist_stake", 75, 1, ("activist investor", "activist stake", "builds stake")),
    # --- medium (50) ---
    ("earnings_miss", 50, -1, (
        "misses estimates", "misses expectations", "falls short of estimates",
        "earnings miss", "revenue miss", "disappointing earnings", "misses on revenue",
    )),
    ("earnings_beat", 50, 1, (
        "beats estimates", "beats expectations", "tops estimates", "tops expectations",
        "earnings beat", "record revenue", "record profit", "record quarter",
        "beats on revenue",
    )),
    ("analyst_downgrade", 50, -1, ("downgrades", "downgraded", "cuts price target", "cut to sell")),
    ("analyst_upgrade", 50, 1, (
        "upgrades", "upgraded", "raises price target", "raised to buy",
    )),
    ("dividend_cut", 50, -1, ("cuts dividend", "suspends dividend", "dividend cut")),
    ("buyback", 50, 1, ("buyback", "share repurchase", "dividend increase", "raises dividend")),
    ("layoffs", 50, 0, ("layoffs", "job cuts", "cuts jobs", "workforce reduction")),
    ("contract_win", 50, 1, ("wins contract", "awarded contract", "wins order", "lands deal")),
    ("macro_rates", 50, 0, (
        "fed raises", "fed cuts", "rate hike", "rate cut", "interest rate decision",
        "fomc", "federal reserve", "basis points",
    )),
    ("macro_inflation", 50, 0, ("inflation", "cpi", "ppi", "jobs report", "payrolls", "gdp")),
    # --- low (25) ---
    ("product_launch", 25, 1, ("launches", "unveils", "announces new", "introduces")),
    ("partnership", 25, 1, ("partnership", "partners with", "collaboration with", "teams up")),
    ("executive_hire", 25, 0, ("appoints", "names new", "hires")),
    ("stake_change", 25, 0, ("raises stake", "trims stake", "sells stake", "reduces stake")),
)

# Historical-retrospective markers. A "ten years since the collapse of X"
# look-back matches the same keywords as the live event; downgrade instead
# of surfacing it as breaking news.
_HISTORICAL_RE = re.compile(
    r"\b(anniversary|years? (ago|since|later|on)|a decade (ago|since)|looking back|"
    r"look back|history of|timeline|explained|explainer|retrospective|throwback)\b",
    re.IGNORECASE,
)

_LEVEL_ORDER = ("info", "low", "medium", "high", "critical")
_SEVERITY_TO_LEVEL = {0: "info", 25: "low", 50: "medium", 75: "high", 100: "critical"}
_LEVEL_TO_SEVERITY = {level: severity for severity, level in _SEVERITY_TO_LEVEL.items()}


def _compile_rule(keywords: tuple[str, ...]) -> re.Pattern[str]:
    """Literal keywords are escaped; a ``re:`` prefix marks a raw regex
    fragment (used for verb-object patterns that tolerate interleaving
    words: "cuts FULL-YEAR guidance", "lowers ITS SALES outlook")."""

    parts = sorted(
        (
            keyword[3:] if keyword.startswith("re:") else re.escape(keyword)
            for keyword in keywords
        ),
        key=len,
        reverse=True,
    )
    return re.compile(r"\b(" + "|".join(parts) + r")\b", re.IGNORECASE)


_COMPILED: tuple[tuple[str, int, int, re.Pattern[str]], ...] = tuple(
    (category, severity, polarity, _compile_rule(keywords))
    for category, severity, polarity, keywords in _RULES
)


def classify_headline(title: str, snippet: str = "") -> EventClassification:
    """Classify a headline into a finance event. Deterministic, pure."""

    text = f"{title} {snippet}".strip()
    if not text:
        return EventClassification(
            category="general", severity=0, polarity=0, confidence=0.3
        )

    historical = bool(_HISTORICAL_RE.search(text))
    for category, severity, polarity, pattern in _COMPILED:
        match = pattern.search(text)
        if match is None:
            continue
        effective_severity = severity
        confidence = 0.7
        if historical:
            # A retrospective is not a live event: demote two bands.
            effective_severity = max(0, severity - 50)
            confidence = 0.4
        return EventClassification(
            category=category,
            severity=effective_severity,
            polarity=polarity,
            confidence=confidence,
            matched_keywords=(match.group(0).lower(),),
            historical=historical,
        )

    return EventClassification(category="general", severity=0, polarity=0, confidence=0.3)


def cap_llm_upgrade(keyword_severity: int, llm_severity: int) -> int:
    """Cap an LLM-provided severity to at most two bands above the
    deterministic keyword severity.

    This blocks the contamination path where a low-information headline
    (keyword severity 0) gets hallucinated into a 100-severity crisis and
    flows straight into conviction. keyword 0 → cap 50; keyword 25 → cap
    75; keyword 50+ → uncapped.
    """

    keyword_level = _SEVERITY_TO_LEVEL.get(keyword_severity)
    llm_level = _SEVERITY_TO_LEVEL.get(llm_severity)
    if keyword_level is None or llm_level is None:
        return keyword_severity
    keyword_rank = _LEVEL_ORDER.index(keyword_level)
    llm_rank = _LEVEL_ORDER.index(llm_level)
    capped_rank = min(llm_rank, keyword_rank + 2)
    return _LEVEL_TO_SEVERITY[_LEVEL_ORDER[capped_rank]]
