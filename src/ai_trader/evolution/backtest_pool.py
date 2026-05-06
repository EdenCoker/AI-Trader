from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class BacktestPoolEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    tickers: tuple[str, ...]
    start: date
    end: date
    config: dict = Field(default_factory=dict)


class BacktestPool(BaseModel):
    model_config = ConfigDict(frozen=True)

    entries: tuple[BacktestPoolEntry, ...] = ()

    @classmethod
    def load(cls, path: Path) -> BacktestPool:
        if not path.exists():
            return cls(entries=default_backtest_entries())
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return cls(entries=tuple(BacktestPoolEntry.model_validate(item) for item in payload))
        return cls.model_validate(payload)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True),
            encoding="utf-8",
        )


def ensure_backtest_pool(path: Path) -> BacktestPool:
    pool = BacktestPool.load(path)
    if not path.exists():
        pool.save(path)
    return pool


def default_backtest_entries() -> tuple[BacktestPoolEntry, ...]:
    raw_entries = [
        ("bt_001", ("MSFT", "AAPL", "NVDA"), "2022-01-03", "2022-12-30", "bear_tech"),
        ("bt_002", ("XOM", "CVX", "COP"), "2022-01-03", "2023-03-31", "energy_inflation"),
        ("bt_003", ("JPM", "BAC", "GS"), "2022-03-01", "2023-06-30", "financials_rates"),
        ("bt_004", ("UNH", "LLY", "JNJ"), "2022-07-01", "2023-12-29", "healthcare_defensive"),
        ("bt_005", ("MSFT", "META", "GOOGL"), "2023-01-03", "2023-12-29", "bull_mega_cap"),
        ("bt_006", ("TSLA", "F", "GM"), "2023-01-03", "2024-03-29", "autos"),
        ("bt_007", ("BA", "LMT", "RTX"), "2023-04-03", "2024-06-28", "aerospace_defense"),
        ("bt_008", ("WMT", "COST", "TGT"), "2023-04-03", "2024-12-31", "retail"),
        ("bt_009", ("AMD", "NVDA", "AVGO"), "2023-07-03", "2024-12-31", "semiconductors"),
        ("bt_010", ("AMZN", "SHOP", "EBAY"), "2023-07-03", "2025-03-31", "commerce"),
        ("bt_011", ("KO", "PEP", "MDLZ"), "2022-01-03", "2024-12-31", "consumer_staples"),
        ("bt_012", ("CAT", "DE", "HON"), "2022-06-01", "2024-12-31", "industrials"),
        ("bt_013", ("CRM", "ADBE", "NOW"), "2022-09-01", "2025-03-31", "software"),
        ("bt_014", ("NOC", "GD", "LHX"), "2022-01-03", "2023-12-29", "defense_contracts"),
        ("bt_015", ("PFE", "MRK", "ABBV"), "2022-04-01", "2024-06-28", "pharma"),
        ("bt_016", ("V", "MA", "PYPL"), "2023-01-03", "2025-03-31", "payments"),
        ("bt_017", ("NFLX", "DIS", "CMCSA"), "2022-01-03", "2024-12-31", "media"),
        ("bt_018", ("GE", "ETN", "EMR"), "2023-01-03", "2025-03-31", "industrial_recovery"),
        ("bt_019", ("PLTR", "SNOW", "MDB"), "2023-04-03", "2025-03-31", "high_growth_software"),
        ("bt_020", ("SPY", "QQQ", "IWM"), "2022-01-03", "2025-03-31", "broad_market"),
    ]
    return tuple(
        BacktestPoolEntry(
            id=entry_id,
            tickers=tickers,
            start=date.fromisoformat(start),
            end=date.fromisoformat(end),
            config={"walk_forward": True, "monte_carlo": True, "regime": regime},
        )
        for entry_id, tickers, start, end, regime in raw_entries
    )
