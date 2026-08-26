# AI Trader

AI Trader is a staged research and execution system for fusing public smart-money data, market/news context, social velocity, trader-philosophy retrieval, and quant signals.

This repository currently implements the core research spine:

- typed configuration for provider keys and runtime settings
- domain models for congressional trades, committee assignments, 13F filings, news, social mentions, macro observations, and signal bundles
- provider contracts for Quiver, SEC EDGAR, Polygon, Reddit/X, and FRED integrations
- a smart-money scorer with explicit no-lookahead checks
- tests for disclosure-date and 13F filing-date guardrails
- a LangChain-core 3-stage narrative/news analyzer (structured JSON outputs)
- an IBKR (TWS/IB Gateway) broker adapter for balances, quotes, positions, and orders
- a reflexivity psychology state machine and social velocity signal
- a trader RAG memory feeding the Final Reasoner
- local training from closed trade outcomes to calibrate conviction and sizing
- self-improvement proposals gated by tests, backtests, safety filters, and human PR review
- a POSIX shared-memory bridge protocol for C++ AlphaEngine integration
- walk-forward backtesting, metrics, and stress Monte Carlo

The system is designed for lawful use of public disclosures. It is a research framework, not investment advice.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,broker,llm,rag,data]"
# Create a repo-root .env file with the settings in docs/HOW_TO_RUN.md
python -m pytest
```

For the full operator setup, software list, API-key locations, IBKR config, local/API LLM switching, RAG indexing, and bridge notes, see [docs/HOW_TO_RUN.md](docs/HOW_TO_RUN.md).

## CLI

```powershell
ai-trader status
ai-trader gui
ai-trader analyze-news --ticker AAPL --headline "..." --body-file .\\news.txt
ai-trader ibkr-positions
ai-trader reason --bundle-file .\\examples\\sample_signal_bundle.json
ai-trader train local --examples-file .\\examples\\sample_training_examples.jsonl --model-out data\\models\\local_calibrator.json
ai-trader train backtest --examples-file logs\\training_examples.jsonl --start-date 2025-01-01 --output logs\\training_backtest_recent.json
ai-trader rag-index
ai-trader backtest run --start 2022-01-01 --end 2024-12-31 --events-file .\\examples\\sample_events.jsonl --starting-balance 10000 --cash-fraction 0.02 --out result.json
ai-trader backtest monte-carlo --result-file result.json --n-sims 10000
ai-trader review-nightly --outcomes-file outcomes.jsonl
```

## Local GUI

Run the local frontend:

```powershell
ai-trader gui
```

It opens at `http://127.0.0.1:8787`. The GUI is a complete local control
surface with:

- a dashboard for provider readiness, training/review counts, hardware-tuned
  ingestion settings, and recent artifacts
- a workbench for whitelisted CLI workflows with live command output
- a label-review queue with accept, skip, and override controls
- an artifact browser for JSON, JSONL, log, and text previews

Trade and automation actions still use the same `.env`, IBKR, and live-trading
safety gates as the CLI.

## Phase Map

1. **Phase 0 - Foundations:** configuration, domain contracts, source boundaries, signal schema.
2. **Phase 1 - Smart Money Monitor:** STOCK Act disclosures and 13F filings scored without look-ahead bias.
3. **Phase 2 - News Intelligence:** staged LLM analysis for expectation calibration, surprise, and behavioral reaction.
4. **Phase 3 - Mass Psychology:** reflexivity state machine and social velocity indicators.
5. **Phase 4 - Trader Imitation:** RAG over public trader writing and interviews.
6. **Phase 5 - Signal Fusion:** unified `SignalBundle` into a final reasoner output.
7. **Phase 6 - Self-Improvement:** nightly post-trade review and controlled prompt/weight proposals.
8. **Phase 7 - C++ Integration:** POSIX shared memory bridge to AlphaEngine.
9. **Phase 8 - Backtesting:** walk-forward validation and stress-period Monte Carlo.
10. **Phase 9 - Build Loop:** Generate -> Self-Critique -> Test -> Backtest Gate -> Human PR.

## News Intelligence Feed

Phase 2 no longer requires hand-fed headlines. The `ai_trader.news` package
pulls a consistent, scored news feed from two independent paths and turns it
into `Signal` objects for the Final Reasoner:

- **World Monitor digest** — the public finance digest of the open-source
  [worldmonitor](https://github.com/koala73/worldmonitor) project (HTTP API
  only; no AGPL code is vendored). Items arrive clustered and scored, with
  per-item tickers and story lifecycle metadata.
- **Direct RSS** — a curated registry of working finance feeds with a
  per-feed circuit breaker (2 failures → 5-minute cooldown), TTL caching,
  and honest per-feed health reporting. Publishers without public RSS
  (Reuters, FT, Bloomberg) are reached via Google News site-search feeds,
  and the originating outlet is lifted from the RSS `<source>` element.

Every sighting is appended to a local archive stamped with the fetch time.
`Signal.effective_date` derives from **first_seen_at** (when this system
observed the story), never the publisher's claimed pubDate — the same
no-lookahead rule as `disclosure_date` and `filing_date`, and it makes
backtest replay deterministic (`stories(as_of)` reads only what was visible
at `as_of`).

Stories are deduplicated with feature-hashed title vectors (edit variants
merge; entity-swapped headlines and disjoint-ticker stories stay separate),
corroboration counts publisher families (Reuters World + Reuters US = one
publisher), and each story carries two orthogonal scores: `importance`
(severity/tier/corroboration/recency → `Signal.strength`) and `credibility`
(tier/propaganda-risk/corroboration, state-media capped → `Signal.confidence`).

```powershell
ai-trader news pull                      # one acquisition pass, archives sightings
ai-trader news stories --limit 15        # ranked stories from the archive window
ai-trader news signal --ticker AAPL      # news Signals for a ticker (JSON)
```

Settings (`.env`): `AI_TRADER_NEWS_ENABLED`, `AI_TRADER_NEWS_WORLDMONITOR_ENABLED`,
`AI_TRADER_WORLDMONITOR_BASE_URL`, `AI_TRADER_NEWS_VARIANT` (default `finance`),
`AI_TRADER_NEWS_MAX_AGE_HOURS` (default 96), `AI_TRADER_NEWS_ARCHIVE_PATH`.

## Look-Ahead Rule

Congressional trades use `disclosure_date` as the signal effective date. 13F holdings use `filing_date`, not `report_period`. Tests should fail if a component tries to score unavailable information.

## LLM Backend (Local vs API)

The LLM layer is pluggable. Select at runtime with environment variables:

- OpenAI API:
  - `AI_TRADER_LLM_BACKEND=openai`
  - `OPENAI_API_KEY=...`
  - `AI_TRADER_LLM_MODEL=gpt-4.5`
- Ollama local:
  - `AI_TRADER_LLM_BACKEND=ollama`
  - `OLLAMA_HOST=http://localhost:11434`
  - `AI_TRADER_LLM_MODEL=llama3.1`
- OpenAI-compatible local server (vLLM/llama.cpp server):
  - `AI_TRADER_LLM_BACKEND=openai_compatible`
  - `OPENAI_BASE_URL=http://localhost:8000/v1`
  - `AI_TRADER_LLM_MODEL=...`

## IBKR Notes

IBKR requires Trader Workstation (TWS) or IB Gateway running with API access enabled. Use `AI_TRADER_TRADING_MODE=paper` by default and only set `AI_TRADER_ALLOW_LIVE_TRADING=true` when you intentionally want live execution.

Trade execution no longer requires you to choose a ticker size manually. Omit `--shares`
and the order is sized from available account funds, the plan conviction, and
`--cash-fraction`. Use `--starting-balance` as a risk-budget cap, or for dry-run sizing
with `--reference-price`.

```powershell
ai-trader trade --plan-file .\\logs\\trade_plan.json --dry-run --starting-balance 10000 --reference-price 400
```

## Trader RAG (Upgrade)

Index a local corpus of trader philosophy (letters/interviews/notes) and feed retrieved analogies into the Final Reasoner.

```powershell
# Optional: local embeddings need sentence-transformers
python -m pip install -e ".[rag]"

ai-trader rag-index
ai-trader rag-query --query "late-cycle euphoria breaks and liquidity tightens" --k 3

# Use RAG automatically during `ai-trader reason`
$env:AI_TRADER_RAG_ENABLED="true"
ai-trader reason --bundle-file .\\examples\\sample_signal_bundle.json
```

## Local Training

Train a local calibrator from your own closed trade outcomes:

```powershell
ai-trader train local --examples-file .\\examples\\sample_training_examples.jsonl --model-out data\\models\\local_calibrator.json
```

Enable it in `.env`:

```env
AI_TRADER_LOCAL_TRAINING_ENABLED=true
AI_TRADER_LOCAL_CALIBRATOR_PATH=data/models/local_calibrator.json
```

When enabled, the Final Reasoner applies the calibrator after deterministic guardrails. It can cap conviction and size when your local history says a similar setup has poor expected P&L.

## C++ Bridge

The Python bridge serializes `TradePlan` objects into a fixed 172-byte little-endian shared-memory message. The matching C++ reader lives in `bridge/include/alpha_bridge.hpp` with an example consumer in `bridge/examples/consumer.cpp`.

```bash
cmake -S bridge -B bridge/build
cmake --build bridge/build
```

## Self-Improvement Safety

AI-generated changes are proposals only. The code rejects attempts to touch live-trading gates, look-ahead guards, disclosure dates, filing dates, effective dates, or broker safety controls before any git branch or PR is created. The GitHub Actions build loop opens `ai-proposal` PRs and never auto-merges them.
