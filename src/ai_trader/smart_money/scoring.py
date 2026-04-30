from __future__ import annotations

import math
from datetime import date
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from ai_trader.domain.events import (
    Chamber,
    CommitteeAssignment,
    CongressionalTrade,
    ThirteenFPositionChange,
    TransactionType,
)
from ai_trader.domain.signals import Signal, SignalBundle, SignalDirection


class LookAheadBiasError(ValueError):
    """Raised when a score tries to use information before it was public."""


class SmartMoneyWeights(BaseModel):
    model_config = ConfigDict(frozen=True)

    base_congressional_trade: float = Field(default=0.20, ge=0, le=1)
    committee_sector_match: float = Field(default=0.35, ge=0, le=1)
    congressional_amount: float = Field(default=0.15, ge=0, le=1)
    congressional_recency: float = Field(default=0.15, ge=0, le=1)
    senate_bonus: float = Field(default=0.05, ge=0, le=1)
    sale_discount: float = Field(default=0.20, ge=0, le=1)

    manager_profile: float = Field(default=0.35, ge=0, le=1)
    position_change: float = Field(default=0.35, ge=0, le=1)
    filing_recency: float = Field(default=0.15, ge=0, le=1)
    position_size: float = Field(default=0.15, ge=0, le=1)


class ManagerProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    aliases: tuple[str, ...] = ()
    cik: str | None = None
    style: str
    base_weight: float = Field(ge=0, le=1)

    def matches(self, manager_name: str, cik: str | None = None) -> bool:
        normalized = _normalize_name(manager_name)
        names = {_normalize_name(self.name), *{_normalize_name(alias) for alias in self.aliases}}
        if normalized in names:
            return True
        return cik is not None and self.cik is not None and cik.lstrip("0") == self.cik.lstrip("0")


DEFAULT_MANAGER_PROFILES: tuple[ManagerProfile, ...] = (
    ManagerProfile(
        name="Berkshire Hathaway",
        aliases=("Warren Buffett", "Buffett"),
        cik="1067983",
        style="quality_value",
        base_weight=0.90,
    ),
    ManagerProfile(
        name="Duquesne Family Office",
        aliases=("Stanley Druckenmiller", "Druckenmiller"),
        cik="1536411",
        style="macro_growth",
        base_weight=0.88,
    ),
    ManagerProfile(
        name="Soros Fund Management",
        aliases=("George Soros", "Soros"),
        cik="1029160",
        style="reflexive_macro",
        base_weight=0.82,
    ),
    ManagerProfile(
        name="Renaissance Technologies",
        aliases=("Renaissance", "RenTech"),
        cik="1037389",
        style="quant_stat_arb",
        base_weight=0.70,
    ),
    ManagerProfile(
        name="Citadel Advisors",
        aliases=("Citadel",),
        cik="1423053",
        style="multi_strategy",
        base_weight=0.62,
    ),
)


class SmartMoneyScorer:
    def __init__(
        self,
        committee_assignments: tuple[CommitteeAssignment, ...] = (),
        manager_profiles: tuple[ManagerProfile, ...] = DEFAULT_MANAGER_PROFILES,
        weights: SmartMoneyWeights | None = None,
    ) -> None:
        self.committee_assignments = committee_assignments
        self.manager_profiles = manager_profiles
        self.weights = weights or SmartMoneyWeights()

    def score_congressional_trade(self, trade: CongressionalTrade, as_of: date) -> Signal:
        if trade.disclosure_date > as_of:
            raise LookAheadBiasError(
                "Congressional trade is not public yet: "
                f"transaction_date={trade.transaction_date}, "
                f"disclosure_date={trade.disclosure_date}, as_of={as_of}. "
                "Backtests must use disclosure dates, never transaction dates."
            )

        direction = self._congressional_direction(trade.transaction_type)
        committee_matches = self._matching_committees(trade)
        recency = _linear_decay((as_of - trade.disclosure_date).days, half_life_days=45)
        amount_score = _amount_score(trade.amount_midpoint, scale=7.0)

        strength = (
            self.weights.base_congressional_trade
            + (self.weights.committee_sector_match if committee_matches else 0.0)
            + self.weights.congressional_amount * amount_score
            + self.weights.congressional_recency * recency
            + (self.weights.senate_bonus if trade.chamber is Chamber.SENATE else 0.0)
        )
        if trade.transaction_type is TransactionType.SALE:
            strength *= 1.0 - self.weights.sale_discount

        strength = _clamp01(strength)
        confidence = _clamp01(
            0.45
            + (0.25 if committee_matches else 0.0)
            + (0.10 if trade.amount_high >= Decimal("50000") else 0.0)
            + 0.10 * recency
        )

        reasons = [
            f"STOCK Act disclosure became public on {trade.disclosure_date}",
            f"{trade.transaction_type.value} in {trade.ticker} by {trade.member_name}",
        ]
        if committee_matches:
            committee_names = ", ".join(match.committee_name for match in committee_matches)
            reasons.append(f"sector-policy overlap with {committee_names}")
        if trade.chamber is Chamber.SENATE:
            reasons.append("senate disclosure receives chamber impact bonus")

        return Signal(
            name="smart_money.congressional_trade",
            ticker=trade.ticker.upper(),
            direction=direction,
            strength=strength,
            confidence=confidence,
            effective_date=trade.disclosure_date,
            horizon_days=60,
            invalidation=(
                "Later amendment reverses the disclosure, or the committee-sector link no longer "
                "applies at the disclosure date."
            ),
            reasons=tuple(reasons),
            metadata={
                "member_name": trade.member_name,
                "chamber": trade.chamber.value,
                "transaction_date": trade.transaction_date.isoformat(),
                "disclosure_date": trade.disclosure_date.isoformat(),
                "amount_midpoint": str(trade.amount_midpoint),
                "committee_matches": [match.committee_name for match in committee_matches],
                "source": trade.source.value,
                "source_url": trade.source_url,
            },
        )

    def score_13f_change(self, change: ThirteenFPositionChange, as_of: date) -> Signal:
        holding = change.current
        if holding.filing_date > as_of:
            raise LookAheadBiasError(
                "13F holding is not public yet: "
                f"report_period={holding.report_period}, filing_date={holding.filing_date}, "
                f"as_of={as_of}. Backtests must use filing dates, never report-period dates."
            )

        direction = _position_change_direction(change)
        profile = self._manager_profile(holding.manager_name, holding.cik)
        recency = _linear_decay((as_of - holding.filing_date).days, half_life_days=90)
        change_magnitude = _position_change_magnitude(change)
        size_score = _amount_score(holding.market_value_usd, scale=11.0)
        profile_weight = profile.base_weight if profile is not None else 0.35

        strength = _clamp01(
            self.weights.manager_profile * profile_weight
            + self.weights.position_change * change_magnitude
            + self.weights.filing_recency * recency
            + self.weights.position_size * size_score
        )
        confidence = _clamp01(0.40 + 0.25 * profile_weight + 0.20 * change_magnitude)

        if change.previous_shares is None:
            change_reason = "new or first-observed 13F position"
        elif change.current.shares > change.previous_shares:
            change_reason = "13F share count increased"
        elif change.current.shares < change.previous_shares:
            change_reason = "13F share count decreased"
        else:
            change_reason = "13F share count unchanged"

        return Signal(
            name="smart_money.13f_change",
            ticker=holding.ticker.upper(),
            direction=direction,
            strength=strength,
            confidence=confidence,
            effective_date=holding.filing_date,
            horizon_days=90,
            invalidation="Next 13F filing shows the manager exited or materially reversed the position.",
            reasons=(
                f"13F filing became public on {holding.filing_date}",
                f"{holding.manager_name}: {change_reason}",
            ),
            metadata={
                "manager_name": holding.manager_name,
                "cik": holding.cik,
                "report_period": holding.report_period.isoformat(),
                "filing_date": holding.filing_date.isoformat(),
                "market_value_usd": str(holding.market_value_usd),
                "shares": str(holding.shares),
                "previous_shares": str(change.previous_shares)
                if change.previous_shares is not None
                else None,
                "manager_profile": profile.name if profile is not None else None,
                "manager_style": profile.style if profile is not None else None,
                "source": holding.source.value,
                "source_url": holding.source_url,
            },
        )

    def build_bundle(
        self,
        ticker: str,
        as_of: date,
        congressional_trades: tuple[CongressionalTrade, ...] = (),
        institutional_changes: tuple[ThirteenFPositionChange, ...] = (),
    ) -> SignalBundle:
        signals: list[Signal] = []
        for trade in congressional_trades:
            if trade.ticker.upper() == ticker.upper():
                signals.append(self.score_congressional_trade(trade, as_of))
        for change in institutional_changes:
            if change.current.ticker.upper() == ticker.upper():
                signals.append(self.score_13f_change(change, as_of))
        return SignalBundle(ticker=ticker.upper(), as_of=as_of, signals=tuple(signals))

    def _matching_committees(self, trade: CongressionalTrade) -> tuple[CommitteeAssignment, ...]:
        member = _normalize_name(trade.member_name)
        trade_sector = _normalize_sector(trade.sector)

        matches = []
        for assignment in self.committee_assignments:
            if assignment.chamber is not trade.chamber:
                continue
            if _normalize_name(assignment.member_name) != member:
                continue
            if not assignment.active_on(trade.disclosure_date):
                continue
            sectors = {_normalize_sector(sector) for sector in assignment.sectors}
            if trade_sector in sectors:
                matches.append(assignment)
        return tuple(matches)

    def _manager_profile(self, manager_name: str, cik: str | None) -> ManagerProfile | None:
        for profile in self.manager_profiles:
            if profile.matches(manager_name, cik):
                return profile
        return None

    @staticmethod
    def _congressional_direction(transaction_type: TransactionType) -> SignalDirection:
        if transaction_type is TransactionType.PURCHASE:
            return SignalDirection.LONG
        if transaction_type is TransactionType.SALE:
            return SignalDirection.SHORT
        return SignalDirection.NEUTRAL


def _normalize_name(value: str) -> str:
    return " ".join(value.casefold().replace(".", "").split())


def _normalize_sector(value: str) -> str:
    normalized = " ".join(value.casefold().replace("&", "and").replace("-", " ").split())
    aliases = {
        "aerospace": "defense",
        "aerospace and defense": "defense",
        "defence": "defense",
        "health care": "healthcare",
        "information technology": "technology",
        "it": "technology",
    }
    return aliases.get(normalized, normalized)


def _amount_score(value: Decimal, scale: float) -> float:
    if value <= 0:
        return 0.0
    return _clamp01(math.log10(float(value) + 1.0) / scale)


def _linear_decay(age_days: int, half_life_days: int) -> float:
    if age_days <= 0:
        return 1.0
    return _clamp01(1.0 - (age_days / (half_life_days * 2)))


def _position_change_direction(change: ThirteenFPositionChange) -> SignalDirection:
    if change.previous_shares is None:
        return SignalDirection.LONG
    if change.current.shares > change.previous_shares:
        return SignalDirection.LONG
    if change.current.shares < change.previous_shares:
        return SignalDirection.SHORT
    return SignalDirection.NEUTRAL


def _position_change_magnitude(change: ThirteenFPositionChange) -> float:
    if change.previous_shares is None:
        return 0.70
    if change.previous_shares == 0:
        return 1.0
    return _clamp01(abs(float((change.current.shares - change.previous_shares) / change.previous_shares)))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))

