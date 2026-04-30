from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


class SourceName(str, Enum):
    QUIVER = "quiver"
    SEC_EDGAR = "sec_edgar"
    POLYGON = "polygon"
    X = "x"
    REDDIT = "reddit"
    FRED = "fred"
    INTERNAL = "internal"


class Chamber(str, Enum):
    HOUSE = "house"
    SENATE = "senate"


class TransactionType(str, Enum):
    PURCHASE = "purchase"
    SALE = "sale"
    EXCHANGE = "exchange"
    OPTION_EXERCISE = "option_exercise"
    OTHER = "other"


class CommitteeAssignment(BaseModel):
    model_config = ConfigDict(frozen=True)

    member_name: str
    chamber: Chamber
    committee_name: str
    sectors: tuple[str, ...] = ()
    start_date: date | None = None
    end_date: date | None = None

    def active_on(self, day: date) -> bool:
        if self.start_date is not None and day < self.start_date:
            return False
        if self.end_date is not None and day > self.end_date:
            return False
        return True


class CongressionalTrade(BaseModel):
    model_config = ConfigDict(frozen=True)

    event_id: str | None = None
    member_name: str
    chamber: Chamber
    ticker: str
    issuer: str
    sector: str
    transaction_date: date
    disclosure_date: date
    transaction_type: TransactionType
    amount_low: Decimal = Field(ge=0)
    amount_high: Decimal = Field(ge=0)
    owner: str | None = None
    source: SourceName = SourceName.QUIVER
    source_url: str | None = None

    @model_validator(mode="after")
    def _validate_amount_range(self) -> CongressionalTrade:
        if self.amount_high < self.amount_low:
            raise ValueError("amount_high must be greater than or equal to amount_low")
        return self

    @computed_field
    @property
    def amount_midpoint(self) -> Decimal:
        return (self.amount_low + self.amount_high) / Decimal("2")

    @property
    def effective_date(self) -> date:
        return self.disclosure_date


class ThirteenFHolding(BaseModel):
    model_config = ConfigDict(frozen=True)

    manager_name: str
    cik: str
    accession_number: str | None = None
    filing_date: date
    report_period: date
    ticker: str
    issuer: str
    cusip: str | None = None
    market_value_usd: Decimal = Field(ge=0)
    shares: Decimal = Field(ge=0)
    source: SourceName = SourceName.SEC_EDGAR
    source_url: str | None = None

    @model_validator(mode="after")
    def _validate_filing_timing(self) -> ThirteenFHolding:
        if self.filing_date < self.report_period:
            raise ValueError("filing_date must be on or after report_period")
        return self

    @property
    def effective_date(self) -> date:
        return self.filing_date


class ThirteenFPositionChange(BaseModel):
    model_config = ConfigDict(frozen=True)

    current: ThirteenFHolding
    previous_shares: Decimal | None = None
    previous_market_value_usd: Decimal | None = None

    @computed_field
    @property
    def share_delta(self) -> Decimal | None:
        if self.previous_shares is None:
            return None
        return self.current.shares - self.previous_shares

    @computed_field
    @property
    def share_delta_pct(self) -> float | None:
        if self.previous_shares is None or self.previous_shares == 0:
            return None
        return float((self.current.shares - self.previous_shares) / self.previous_shares)


class MarketTick(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    timestamp: datetime
    price: Decimal = Field(gt=0)
    size: Decimal = Field(ge=0)
    exchange: str | None = None
    source: SourceName = SourceName.POLYGON


class NewsArticle(BaseModel):
    model_config = ConfigDict(frozen=True)

    article_id: str
    published_at: datetime
    title: str
    source_name: str
    tickers: tuple[str, ...] = ()
    summary: str | None = None
    url: str | None = None
    source: SourceName = SourceName.POLYGON


class SocialMention(BaseModel):
    model_config = ConfigDict(frozen=True)

    mention_id: str
    platform: SourceName
    published_at: datetime
    author: str
    text: str
    tickers: tuple[str, ...] = ()
    engagement_count: int = Field(default=0, ge=0)


class MacroObservation(BaseModel):
    model_config = ConfigDict(frozen=True)

    series_id: str
    observed_on: date
    release_date: date
    value: Decimal
    source: SourceName = SourceName.FRED

    @property
    def effective_date(self) -> date:
        return self.release_date

