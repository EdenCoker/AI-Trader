"""Ticker extraction: cashtags, unambiguous names, ambiguity adjacency gate."""

from __future__ import annotations

from ai_trader.news.tickers import MAX_TICKERS, extract_tickers


def test_cashtags_and_names():
    assert extract_tickers("$AAPL rallies as Microsoft slides") == ("AAPL", "MSFT")
    assert extract_tickers("lowercase $aapl is dropped") == ()
    assert extract_tickers("US$100 is not a cashtag") == ()


def test_ambiguous_names_need_adjacent_company_context():
    cases_tagged = [
        ("Target shares jump after earnings beat", "TGT"),
        ("Gap shares rally on turnaround plan", "GAP"),
        ("Zoom shares fall after weak guidance", "ZM"),
        ("Amazon beats estimates with record revenue", "AMZN"),
        ("shares of Visa climb on payment volume", "V"),
        ("Meta reports record quarterly revenue", "META"),
    ]
    for text, symbol in cases_tagged:
        assert extract_tickers(text) == (symbol,), text

    cases_untagged = [
        "Walmart price target raised to $80",  # 'target' is analyst jargon
        "Gap between rich and poor widens, shares data shows",
        "Zoom in on the market rally this quarter",
        "Amazon rainforest deforestation accelerates",
        "Visa restrictions tighten for foreign students",
        "Meta-analysis of drug trials finds no link",
        "Company posts 3m units sold",
    ]
    for text in cases_untagged:
        tickers = extract_tickers(text)
        assert not set(tickers) & {"TGT", "GAP", "ZM", "AMZN", "V", "META", "MMM"}, (
            text,
            tickers,
        )


def test_walmart_price_target_tags_walmart_not_target():
    assert extract_tickers("Walmart price target raised to $80") == ("WMT",)


def test_cap_and_dedup():
    symbols = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH", "III"]
    text = " ".join(f"${symbol}" for symbol in symbols)
    result = extract_tickers(text + " $AAA")
    assert len(result) == MAX_TICKERS
    assert result[0] == "AAA"


def test_extra_names_from_watchlist():
    assert extract_tickers(
        "Initech wins defense contract", extra_names={"Initech": "INI"}
    ) == ("INI",)
