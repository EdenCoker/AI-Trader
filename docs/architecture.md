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

Congressional trades are now split into three signals:
- `congressional_insider` — combined buy/sell net (legacy, kept for compatibility)
- `house_trade` — House member trades only (confidence capped at 0.55; historically weaker alpha)
- `senate_trade` — Senate member trades only (confidence capped at 0.65; historically stronger alpha)

Chamber is sourced from Quiver's dedicated `/live/housetrading` and `/live/senatetrading` endpoints, with a fallback `chamber` column from the bulk Excel file when available.

13F scoring combines manager profile strength, position-change magnitude, filing recency, and position size. Because 13F filings are delayed and long-only, they are slower regime/context signals rather than direct tick-level triggers.

Two 13F signals are now produced per filing event:
- `institutional_accumulation` — incremental add (delta_pct ≥ 20%)
- `institutional_initiation` — brand-new position (`previous_shares is None`); confidence 0.60, horizon 63 days

## SEC Form 4 Cluster Signal

The `sec_form4_cluster` signal fires when ≥3 distinct insiders (by `rptOwnerCik`) buy in the same disclosure window. This is a stronger signal than a single insider purchase. Strength scales with the number of unique filers up to 10:

```text
strength = min(unique_filers / 10, 1.0) * 0.70
confidence = 0.72
horizon = 20 days
```

The existing `insider_buy` / `insider_sell` signals continue to fire independently for any non-zero net transaction.

## Live Fear/Greed Composite

`LiveFearGreedProvider` builds a live market psychology snapshot from component-level
signals rather than relying on a single scraped index. Each snapshot records
`observed_at`, `available_at`, component freshness, source, confidence, and raw
component metadata. Snapshots can be appended to `data/live/fear_greed.jsonl` via:

```text
ai-trader fear-greed
```

The current composite includes market momentum, broad ETF breadth proxy,
volatility, credit risk appetite, safe-haven demand, growth speculation,
Cboe total put/call ratio when available, and optional Alpha Vantage news
sentiment when `ALPHA_VANTAGE_API_KEY` is configured.

Historical backtests continue to consume dated historical rows. Live ingestion
only appends the current snapshot to today's `fear_greed` row so older training
examples do not receive future sentiment values.

## Source Evolution Pipeline

The `evolution/` package implements autonomous data-source discovery and implementation:

```
DiscoveryAgent → SourceImplementationAgent → IngestionOrchestrator → TrainingAgent → PromotionGate → ModelPromoter
```

Source scoring formula:
$$\text{score} = 0.35 \cdot \text{coverage} + 0.25 \cdot \text{freshness} + 0.30 \cdot \text{lift} - 0.10 \cdot \text{complexity}$$

Implementation confidence formula:
$$\text{confidence} = 0.65 \cdot \max(\text{base\_score}, 0) + 0.35 \cdot \text{profitability\_proxy}$$

## Ingestion Acceleration

The training ingestion path keeps the original source semantics but runs
independent source loaders concurrently, builds the ticker/date event index once,
and prefetches Polygon prices in parallel before example construction. Polygon
daily price frames are cached under `AI_TRADER_INGESTION_CACHE_DIR` so repeated
or resumed runs can reuse covering ticker/date ranges.

Every run writes a timing profile to `AI_TRADER_INGESTION_PROFILE_PATH` with
per-source durations, row counts, price-cache hits/misses, and slowest stages.
Concurrency is controlled separately for source loaders and price fetches:

```text
AI_TRADER_INGESTION_SOURCE_WORKERS
AI_TRADER_INGESTION_PRICE_WORKERS
```

Provider throttling still wins over local machine capacity, so worker counts
should be tuned from the profile rather than only from CPU/network headroom.

## Local Frontend

The `ai-trader gui` command starts a local `ThreadingHTTPServer` that serves the
single-page frontend from `ai_trader.gui.frontend`. The server intentionally
keeps a small API surface and reuses the same whitelisted command definitions in
`ai_trader.gui.actions` that the older launcher used.

Frontend views:

- **Dashboard** calls `/api/overview` for provider readiness, training/review
  counts, hardware-tuned ingestion defaults, paths, and recent artifacts.
- **Workbench** renders action forms from `/api/actions`, executes short commands
  through `/api/run`, and streams long-running commands through `/api/stream`.
- **Label Review** reads `/api/review` and writes decisions through
  `/api/review/decide`, producing human-labeled examples under `logs/`.
- **Artifacts** previews repository-local JSON, JSONL, log, and text files via
  `/api/artifact?path=...`.

Artifact reads are constrained to the repository root before any file content is
returned. Trade actions still route through the normal CLI safety gates and IBKR
configuration; the GUI does not bypass live-trading checks.

## Safety Rails

- Source dates are first-class model fields.
- Scoring raises `LookAheadBiasError` if `as_of` predates disclosure or filing availability.
- Provider keys are loaded from environment variables and redacted for diagnostics.
- Future self-improvement should propose changes in auditable records before any production application.
