# Architecture

## Core Boundary

The project separates source ingestion from signal production:

- `providers`: contracts for data sources and external APIs
- `domain`: immutable-ish market, disclosure, filing, and signal models
- `smart_money`: scoring logic for congressional and institutional disclosures
- future packages: `news`, `psychology`, `rag`, `fusion`, `memory`, `bridge`, and `backtesting`

Provider implementations should return domain models and keep authentication, pagination, retries, and raw payload quirks behind the contract. Scorers should be deterministic and testable with fixture data.

## Signal Lifecycle

Every event has an economic date and an availability date. Backtests and live scoring must only act on availability dates:

- STOCK Act trades: `transaction_date` is economic, `disclosure_date` is usable
- 13F filings: `report_period` is economic, `filing_date` is usable
- macro data: `observed_on` is economic, `release_date` is usable
- news/social: `published_at` is usable once ingested

`SignalBundle` rejects future-effective signals for its `as_of` date.

## Smart-Money Formula

The Phase 1 congressional score is intentionally transparent:

```text
strength =
  base_event
  + committee_sector_match
  + amount_component
  + disclosure_recency_component
  + chamber_bonus
```

Sales receive a discount because public sale disclosures are often more ambiguous than purchases. The scorer records the matched committees and date fields in metadata so later explainability and audits can see why the signal exists.

13F scoring combines manager profile strength, position-change magnitude, filing recency, and position size. Because 13F filings are delayed and long-only, they are slower regime/context signals rather than direct tick-level triggers.

## Safety Rails

- Source dates are first-class model fields.
- Scoring raises `LookAheadBiasError` if `as_of` predates disclosure or filing availability.
- Provider keys are loaded from environment variables and redacted for diagnostics.
- Future self-improvement should propose changes in auditable records before any production application.

