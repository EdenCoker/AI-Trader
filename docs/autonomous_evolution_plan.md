# Autonomous Evolution Plan
## AI-Trader: Continuous Self-Improvement Architecture

> **Goal**: The system should discover new data sources, expand its ticker universe,
> ingest richer data each cycle, train a candidate model, and automatically promote
> that candidate to production if it beats the current model on 3 randomly selected
> out-of-sample backtests — all without human intervention.

---

## 1. Philosophy & Principles

| Principle | Implementation |
|---|---|
| **Never break production** | Candidate model runs in shadow mode; only promoted after passing the gate |
| **Fail loudly** | Every agent writes a structured JSON report; failures trigger alerts via `alerts.py` |
| **Reversibility** | Each promoted model is versioned and archived; one-command rollback |
| **Auditability** | Full provenance chain: data source → example → training run → backtest → promotion decision |
| **Minimal human surface** | Humans approve data-source additions once; everything else is automated |

---

## 2. Agent Topology

```
┌──────────────────────────────────────────────────────────────────────┐
│                     WEEKLY EVOLUTION CYCLE                           │
│                                                                      │
│  ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐  │
│  │ 1. Discovery    │───▶│ 2. Source        │───▶│ 3. Ingestion    │  │
│  │    Agent        │    │    Implement.    │    │    Orchestrator │  │
│  └─────────────────┘    │    Agent         │    └─────────────────┘  │
│           │             └──────────────────┘           │            │
│           ▼                                             ▼            │
│  ┌─────────────────┐                         ┌────────────────────┐  │
│  │ 4. Ticker        │                        │ 5. Training        │  │
│  │    Expansion     │                        │    Agent           │  │
│  │    Agent         │                        └────────────────────┘  │
│  └─────────────────┘                                  │              │
│                                                        ▼             │
│                                            ┌────────────────────┐    │
│                                            │ 6. Promotion       │    │
│                                            │    Gate (3-BT)     │    │
│                                            └────────────────────┘    │
│                                               Pass    │    Fail      │
│                                               ┌───────┘    └──────┐  │
│                                               ▼                   ▼  │ 
│                                      ┌──────────────┐   ┌──────────┐ │
│                                      │ 7. Model     │   │ Archive  │ │
│                                      │    Promoter  │   │ & Alert  │ │
│                                      └──────────────┘   └──────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

Each agent is a standalone Python class with a `.run() -> AgentReport` method.
The weekly scheduler (`scripts/weekly_evolution.py`) orchestrates them in order.

---

## 3. Weekly Cycle — Step-by-Step

### Day 1 (Monday night, ~00:00 UTC): Discovery + Expansion
1. **DiscoveryAgent** scans the data-source registry for any sources that have
   not been validated in the last 30 days. For each candidate source it:
   - Fetches a small sample and validates schema
   - Scores the source on coverage (% of current tickers with data), freshness,
     and historical backtest lift (estimated via a quick 10-example regression probe)
   - Writes a `DataSourceProposal` to `data/source_proposals/`
   - Auto-registers new probes with `status: candidate`; promotes sources with
     `profitability_proxy >= 0.7` directly to `pending_approval` without requiring
     the numeric score threshold to be met
2. **SourceImplementationAgent** reads the registry and builds a ranked task queue
   at `data/source_proposals/implementation_tasks.json`. Each task contains:
   - Priority (1–5), adapter ID, auth env var, confidence score, and bootstrap steps
   - Confidence formula: `0.65 * base_score + 0.35 * profitability_proxy` (uses a
     0.75 freshness prior for candidates without a validation timestamp)
   - Tasks with priority 1–2 are auto-wired into the ingest pipeline on the same cycle
     (pending implementation); `--skip-implementation` CLI flag disables this step
3. **TickerExpansionAgent** queries multiple discovery feeds (see §5) to find
   tickers that are not in the current watchlist but appear frequently in
   smart-money signals. It scores each candidate and appends approved ones to
   `data/watchlist.txt` up to a configured weekly cap (default: +20 tickers).

### Day 2–3 (Tue–Wed): Ingestion
4. **IngestionOrchestrator** runs the full pipeline for the expanded ticker
   universe across all active data sources:
   ```
   scripts/ingest_training_data.py --out logs/candidate_examples.jsonl \
       --tickers $(cat data/watchlist.txt)
   ```
   - Each source is fetched with retry/backoff; failures are logged but do not
     halt the run.
   - New `LocalTrainingExample` records are appended to a rolling 52-week window
     (`logs/rolling_examples.jsonl`); records older than 52 weeks are evicted.
   - A data-quality report is written: schema violations, ticker coverage %, date
     gaps, and outlier counts.

### Day 3–4 (Wed–Thu): Training
5. **TrainingAgent** trains a candidate `LocalCalibratorModel` on the rolling
   window:
   ```
   python -m ai_trader.cli train local \
       --examples-file logs/rolling_examples.jsonl \
       --horizon all \
       --out data/models/candidate.json
   ```
   - Model metadata (training count, feature means, metrics) is written alongside.
   - The candidate is NOT yet used in live trading.

### Day 5–6 (Thu–Fri): Promotion Gate
6. **PromotionGate** runs the 3-backtest challenge (see §6 for full detail).
   - On **pass**: promotes candidate → production, archives previous model.
   - On **fail**: archives candidate with reason, sends alert, keeps current model.

### Day 7 (Weekend): Housekeeping
6. Rotate old proposal files, prune Polygon cache entries older than 2 years,
   compact the rolling JSONL, run `pytest` to verify no regressions were
   introduced by promoted changes.

---

## 4. Data Source Discovery Agent

### 4a. Source Registry (`data/source_registry.json`)
```json
{
  "sources": [
    {
      "id": "quiver_congress",
      "type": "api",
      "url": "https://api.quiverquant.com/beta/bulk/congresstrading",
      "auth": "QUIVER_API_KEY",
      "schema_version": 1,
      "last_validated": "2026-05-06",
      "status": "active",
      "lift_score": 0.14,
      "free_tier": true,
      "category": "general",
      "profitability_proxy": 0.0,
      "ingestion_adapter": null
    },
    ...
  ]
}
```

The `DataSourceRecord` schema (pydantic model in `evolution/source_registry.py`) includes
four additional fields added in May 2026:

| Field | Type | Purpose |
|---|---|---|
| `free_tier` | `bool` | Whether the source requires a paid API subscription |
| `category` | `str` | Signal category: `insider`, `congress`, `institutional`, `earnings`, `news`, `general` |
| `profitability_proxy` | `float 0–1` | Expert-estimated signal alpha; used in implementation confidence formula |
| `ingestion_adapter` | `str \| None` | Key matching the loader in `ingest_training_data.py` |

**Current registry state (May 2026):** 14 total sources — 7 active (original), 3 active
(newly implemented: `sec_form4_cluster`, `quiver_live_house_senate`,
`sec_13f_position_initiations`), 1 pending approval (`openinsider_cluster_buys`),
3 candidate (`fmp_earnings_surprises`, `koyfin_insider_news_rss`, `gdelt_finance_feed`).

### 4b. Implementation Task Queue

The `SourceImplementationAgent` (`evolution/source_implementation.py`) converts
registry candidates into an actionable task list:

```
data/source_proposals/implementation_tasks.json
```

Each `ImplementationTask` contains:
- `source_id`, `priority` (1–5), `category`, `ingestion_adapter`
- `auth_env` (env var required), `free_tier`, `profitability_proxy`
- `confidence` — implementation confidence score
- `bootstrap_steps` — ordered checklist for implementing the adapter

Confidence formula:
$$\text{confidence} = 0.65 \times \max(\text{base\_score}, 0) + 0.35 \times \text{profitability\_proxy}$$

Priority mapping: `profitability_proxy ≥ 0.80 → 1`, `≥ 0.65 → 2`, `≥ 0.50 → 3`,
`≥ 0.35 → 4`, otherwise `5`.

### 4c. Candidate Source Scoring

For each candidate source the `DiscoveryAgent` computes a **Source Score**:

$$\text{score} = w_1 \cdot \text{coverage} + w_2 \cdot \text{freshness} + w_3 \cdot \text{lift} - w_4 \cdot \text{complexity}$$

Where:
- **coverage** = fraction of watchlist tickers with ≥1 signal in the last 30 days
- **freshness** = 1 if data updated within 24 h, decays exponentially
- **lift** = correlation of the source's signal with 30-day forward returns on a held-out 500-example probe
- **complexity** = normalized schema complexity (number of parse steps required)

Sources scoring above 0.5 are added as `pending_approval` status. Sources with
`profitability_proxy >= 0.7` are also promoted to `pending_approval` regardless of
the numeric score. A daily Slack/email digest lists pending sources for human
one-time approval; once approved they become `active` without further human action.

### 4d. Auto-Discovery Probes

The agent runs probes against the following feeds each week looking for new signal:

Seven probes are pre-registered in `DEFAULT_DISCOVERY_PROBES` (all `free_tier=True`):

| Source ID | Category | Adapter | `profitability_proxy` | Status |
|---|---|---|---|---|
| `sec_form4_cluster` | insider | `sec_form4` | 0.88 | **active** |
| `quiver_live_house_senate` | congress | `quiver_congress` | 0.82 | **active** |
| `sec_13f_position_initiations` | institutional | `sec_13f` | 0.79 | **active** |
| `openinsider_cluster_buys` | insider | `openinsider_csv` | 0.74 | pending_approval |
| `fmp_earnings_surprises` | earnings | `fmp_earnings` | 0.67 | candidate |
| `koyfin_insider_news_rss` | news | `rss_events` | 0.52 | candidate |
| `gdelt_finance_feed` | news | `gdelt` | 0.44 | candidate |

Additional probe targets for future cycles:

| Category | Probe Target |
|---|---|
| Options flow | Unusual Whales public RSS, Tradytics public endpoints |
| Macro | FRED series catalog — scan for new series tagged "finance" or "business" |
| News/NLP | RSS autodiscovery on major financial publishers |
| Short interest | Ortex public summaries (supplement existing FINRA feed) |
| Insider filings | Form 144 (pre-planned sales; SEC EDGAR) |
| Patent filings | USPTO bulk data RSS for assignee matches against watchlist tickers |
| Supply chain | Bloomberg Industry Classification hierarchy — upstream/downstream peers |

---

## 5. Ticker Expansion Agent

### 5a. Candidate Ticker Feeds

Every week the agent collects candidate tickers from:

1. **Congressional trade filings** — any ticker appearing in a new Quiver record not
   currently in watchlist
2. **SEC 13F new positions** — tickers that appear as new positions in ≥3 large
   institutional 13F filings this quarter
3. **Options flow anomalies** — tickers with unusually large call/put sweeps from the
   Unusual Whales RSS in the past 7 days
4. **S&P 500 constituent changes** — additions to any major index (S&P 500, Russell 1000,
   Nasdaq 100) via a free index-change feed
5. **Peer expansion** — for each current watchlist ticker, fetch its sector peers from
   Polygon's Ticker Details v3 (`related_companies`) and add any peer appearing ≥3 times
6. **Earnings surprise movers** — tickers with >5% absolute post-earnings move in the
   past 2 weeks that are not already covered

### 5b. Candidate Filtering

Before adding a ticker the agent checks:
- Market cap ≥ $1B (via Polygon Ticker Details)
- Average daily volume ≥ 500K shares over the past 30 days
- At least 2 years of OHLCV history available in Polygon
- Not an ADR, ETF, or SPAC (type == "CS" in Polygon)
- Not already on the watchlist or on a manual exclusion list

### 5c. Watchlist Growth Cap

- Maximum **+20 tickers** added per week (configurable via `MAX_WEEKLY_TICKER_ADDS`)
- Tickers are ranked by composite score (frequency in discovery feeds + signal strength
  from any already-ingested data) and the top-N are added
- A ticker that has not generated a single actionable signal in 6 consecutive weeks
  is **demoted** to a `cold_storage` list and excluded from future ingestion (reducing
  compute cost); it can be reactivated automatically if it re-appears in a discovery feed

---

## 6. Promotion Gate: The 3-Backtest Challenge

This is the core safety mechanism. Before any candidate model goes live:

### 6a. Backtest Pool Construction

A **backtest pool** of ~50 distinct configurations is maintained in
`data/backtest_pool.json`. Each entry specifies:
```json
{
  "id": "bt_042",
  "tickers": ["NVDA", "MSFT", "META"],
  "start": "2022-01-03",
  "end": "2024-12-31",
  "config": { "walk_forward": true, "monte_carlo": true }
}
```

Pool entries span different:
- Market regimes (bull 2023, bear 2022, sideways 2024)
- Sector clusters (tech, financials, energy, healthcare)
- Holding horizons (short 7d, medium 30d, long 90d)

New entries are added automatically each month based on the most recent completed
quarter, ensuring the pool stays current.

### 6b. Random Selection & Scoring

```python
import random

selected = random.sample(backtest_pool, k=3)
results = []
for bt in selected:
    baseline_score = run_backtest(current_model, bt)
    candidate_score = run_backtest(candidate_model, bt)
    results.append({
        "id": bt["id"],
        "baseline_sharpe": baseline_score.sharpe,
        "candidate_sharpe": candidate_score.sharpe,
        "passed": candidate_score.sharpe >= baseline_score.sharpe - TOLERANCE
    })
```

`TOLERANCE = 0.05` (candidate is allowed to be 0.05 Sharpe points below baseline on
any individual test — what matters is the aggregate).

### 6c. Promotion Decision Matrix

| Backtests Won | Outcome |
|---|---|
| 3 / 3 | **Promote** — full confidence |
| 2 / 3 | **Conditional promote** — promote only if max drawdown also improved; log warning |
| 1 / 3 or 0 / 3 | **Reject** — archive candidate, keep current model, schedule diagnostic review |

The aggregate win condition used in code:

```python
wins = sum(1 for r in results if r["passed"])
drawdown_ok = candidate_max_dd <= current_max_dd * 1.05  # allow 5% slack

if wins == 3 or (wins == 2 and drawdown_ok):
    promote(candidate)
else:
    reject(candidate, reason=f"{wins}/3 backtests passed")
```

### 6d. Model Versioning

```
data/models/
    production.json          ← symlink/copy of active model
    archive/
        v0001_2026-04-28.json
        v0002_2026-05-05.json   ← previous
        v0003_candidate_REJECTED_2026-05-12.json
        v0004_2026-05-19.json   ← current production
```

Version metadata is stored in `data/models/version_registry.json`:
```json
{
  "current": "v0004",
  "history": [
    { "version": "v0004", "promoted_at": "2026-05-19T02:14:00Z",
      "backtest_wins": 3, "sharpe_delta": 0.12, "training_count": 14822 },
    ...
  ]
}
```

One-command rollback: `python -m ai_trader.cli model rollback --to v0003`

---

## 7. New File / Module Map

```
scripts/
    weekly_evolution.py          ← top-level weekly orchestrator
                                    (--skip-implementation flag available)

src/ai_trader/
    evolution/                   (package)
        __init__.py
        discovery.py             ← DiscoveryAgent (source scoring, probes, auto-registration)
        source_implementation.py ← SourceImplementationAgent (task queue builder)
        source_registry.py       ← DataSourceRecord + SourceRegistry pydantic models
        ticker_expansion.py      ← TickerExpansionAgent
        ingestion_orchestrator.py← parallel ingestion with retry
        training_agent.py        ← wraps LocalCalibratorTrainer, writes versioned model
        promotion_gate.py        ← 3-backtest challenge, version registry
        promoter.py              ← atomic file-swap + archive + alert
        backtest_pool.py         ← pool management (add/evict entries)
        watchlist_manager.py     ← add/demote tickers, enforce cap
        reports.py               ← AgentReport base model, JSON serialization

data/
    source_registry.json         (managed by DiscoveryAgent — 14 sources as of May 2026)
    backtest_pool.json           (managed by backtest_pool.py)
    watchlist.txt                (written by watchlist_manager.py)
    source_proposals/
        implementation_tasks.json← ranked task queue from SourceImplementationAgent
    models/
        production.json          (promoted atomically)
        version_registry.json
        archive/
```

---

## 8. `weekly_evolution.py` Skeleton

```python
"""Weekly autonomous evolution cycle."""
from __future__ import annotations

import datetime as dt
import json
import random
from pathlib import Path

from ai_trader.evolution.discovery import DiscoveryAgent
from ai_trader.evolution.ticker_expansion import TickerExpansionAgent
from ai_trader.evolution.ingestion_orchestrator import IngestionOrchestrator
from ai_trader.evolution.training_agent import TrainingAgent
from ai_trader.evolution.promotion_gate import PromotionGate
from ai_trader.evolution.promoter import ModelPromoter
from ai_trader.alerts import send_alert


def main() -> None:
    run_id = dt.datetime.now(dt.UTC).strftime("weekly_%Y%m%d")
    report: dict = {"run_id": run_id, "steps": []}

    # 1. Discovery
    disc = DiscoveryAgent()
    report["steps"].append(disc.run())          # writes DataSourceProposals

    # 2. Ticker expansion
    exp = TickerExpansionAgent(max_adds=20)
    report["steps"].append(exp.run())           # updates data/watchlist.txt

    # 3. Ingestion
    ingest = IngestionOrchestrator(
        watchlist=Path("data/watchlist.txt"),
        out=Path("logs/rolling_examples.jsonl"),
        rolling_weeks=52,
    )
    report["steps"].append(ingest.run())

    # 4. Training
    trainer = TrainingAgent(
        examples=Path("logs/rolling_examples.jsonl"),
        out=Path("data/models/candidate.json"),
    )
    report["steps"].append(trainer.run())

    # 5. Promotion gate (3 random backtests)
    gate = PromotionGate(
        current=Path("data/models/production.json"),
        candidate=Path("data/models/candidate.json"),
        pool=Path("data/backtest_pool.json"),
        k=3,
        tolerance=0.05,
    )
    gate_result = gate.run()
    report["steps"].append(gate_result)

    # 6. Promote or reject
    if gate_result["promoted"]:
        promoter = ModelPromoter()
        promoter.promote(
            candidate=Path("data/models/candidate.json"),
            reason=f"Won {gate_result['wins']}/3 backtests",
        )
        send_alert("model_promoted", gate_result)
    else:
        send_alert("model_rejected", gate_result)

    Path(f"logs/weekly_{run_id}.json").write_text(
        json.dumps(report, indent=2, default=str)
    )


if __name__ == "__main__":
    main()
```

---

## 9. Scheduling

### Windows (Task Scheduler)
```xml
<!-- weekly_evolution_task.xml -->
<Task>
  <Triggers>
    <CalendarTrigger>
      <StartBoundary>2026-05-11T01:00:00</StartBoundary>
      <ScheduleByWeek>
        <WeeksInterval>1</WeeksInterval>
        <DaysOfWeek><Monday /></DaysOfWeek>
      </ScheduleByWeek>
    </CalendarTrigger>
  </Triggers>
  <Actions>
    <Exec>
      <Command>python</Command>
      <Arguments>scripts\weekly_evolution.py</Arguments>
      <WorkingDirectory>C:\...\AI-Trader</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
```

### Linux / macOS (cron)
```cron
# Every Monday at 01:00 UTC
0 1 * * 1 cd /path/to/AI-Trader && python scripts/weekly_evolution.py >> logs/weekly_cron.log 2>&1
```

The nightly pipeline (`nightly_pipeline.py`) continues to run daily for
incremental data pulls; the weekly cycle handles the full retrain + promotion.

---

## 10. Alert & Observability Contract

Every agent report written to `logs/` follows this schema:
```json
{
  "agent": "DiscoveryAgent",
  "run_id": "weekly_20260519",
  "started_at": "2026-05-19T01:02:11Z",
  "finished_at": "2026-05-19T01:04:55Z",
  "status": "ok | partial | failed",
  "summary": { ... agent-specific KPIs ... },
  "errors": []
}
```

A lightweight dashboard (`scripts/show_evolution_status.py`) reads all weekly logs
and prints a table:

```
Week         Model     Training Ex  Tickers  BT Wins  Promoted  Sharpe Δ
2026-04-28   v0003     12 441        187      2/3      No        –
2026-05-05   v0004     13 809        193      3/3      Yes       +0.12
2026-05-12   v0005     14 822        201      3/3      Yes       +0.07
```

---

## 11. Safety Rails

| Risk | Mitigation |
|---|---|
| Candidate wipes out production model file | `promote()` uses atomic rename; previous model is archived before overwrite |
| Bad data source poisons training | Source scores are re-evaluated each week; sources with negative lift are automatically suspended |
| Ticker universe grows unbounded | Hard cap of 500 tickers enforced by `WatchlistManager`; cold-storage demotion after 6 idle weeks |
| Runaway API costs from new tickers | Ingestion budget check: abort if estimated API calls > `MAX_WEEKLY_POLYGON_CALLS` (default 50 000) |
| Backtest overfitting to pool | Pool entries are regenerated from live data each month; held-out dates are never in any training window |
| Model promotes on fluke 3-0 | Promotion also requires: `training_count ≥ 10 000`, `max_drawdown ≤ 0.30`, `win_rate ≥ 0.48` |
| Infinite promotion churn | Minimum 5-day hold before a newly promoted model is eligible to be replaced |

---

## 12. Implementation Roadmap

### Phase 1 — Foundation (1–2 weeks)
- [x] Create `src/ai_trader/evolution/` package with `reports.py`, `source_registry.py`, `watchlist_manager.py`
- [x] Create `data/source_registry.json` seeded from existing ingest sources
- [x] Create `data/backtest_pool.json` seeded with 20 entries spanning 2022–2025
- [x] Implement `PromotionGate` and `ModelPromoter` (atomic versioned swap)
- [x] Implement `scripts/weekly_evolution.py` (stub agents, real gate)
- [x] Add `model rollback` sub-command to `ai_trader.cli`

### Phase 2 — Ticker Expansion (1 week)
- [ ] Implement `TickerExpansionAgent` with Polygon peer lookup and index-change feed
- [x] Wire cold-storage demotion into `WatchlistManager`
- [x] Add `scripts/show_evolution_status.py` dashboard

### Phase 3 — Source Discovery (2 weeks)
- [x] Implement `DiscoveryAgent` with source scoring formula
- [ ] Add probes for 10 discovery targets listed in §4c
- [ ] Human-approval workflow for new sources (email digest + approval flag in registry)

### Phase 4 — Full Autonomy (1 week)
- [ ] Wire all agents into `weekly_evolution.py` end-to-end
- [ ] Schedule via Task Scheduler / cron
- [ ] Load-test with full 200-ticker universe
- [ ] Verify rollback path works correctly end-to-end
