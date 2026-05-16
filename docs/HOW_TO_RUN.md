# How To Run AI Trader

This guide covers the software to install, where API keys go, and the safest order to bring the system online.

AI Trader is a research and paper-trading system. Keep `AI_TRADER_TRADING_MODE=paper` and `AI_TRADER_ALLOW_LIVE_TRADING=false` until you have validated data, backtests, logs, and broker behavior.

## Required Software

Install these first:

- Windows 11 with PowerShell
- Python 3.11 or 3.12
- Git
- Visual Studio Build Tools or Visual Studio Community with C++ build tools
- CMake and Ninja, needed for the C++ bridge example
- Trader Workstation or IB Gateway from Interactive Brokers
- GitHub CLI (`gh`), needed only for AI proposal PR creation

Optional but recommended:

- Docker Desktop with WSL2 backend
- WSL2 Ubuntu, especially for POSIX shared memory bridge testing
- Ollama, if you want local models
- PostgreSQL, if you later enable persistent memory beyond local files

## Python Setup

From the repo root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,broker,llm,rag,data]"
# Create a repo-root .env file and add the keys/settings shown below.
python -m pytest
```

Optional bridge dependency on Linux/WSL2:

```bash
python -m pip install -e ".[bridge]"
```

## API Keys

Put keys in the repo-root `.env` file. Do not commit `.env`.

### LLM API

For OpenAI API:

```env
AI_TRADER_LLM_BACKEND=openai
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
AI_TRADER_LLM_MODEL=gpt-4.5
AI_TRADER_FINAL_REASONER_MODEL=gpt-4.5
```

For Ollama local:

```env
AI_TRADER_LLM_BACKEND=ollama
OLLAMA_HOST=http://localhost:11434
AI_TRADER_LLM_MODEL=llama3.1
```

For a local OpenAI-compatible server, such as vLLM or llama.cpp server:

```env
AI_TRADER_LLM_BACKEND=openai_compatible
OPENAI_BASE_URL=http://localhost:8000/v1
OPENAI_API_KEY=
AI_TRADER_LLM_MODEL=your-local-model
```

### Market And Public Data

```env
QUIVER_API_KEY=...
POLYGON_API_KEY=...
FRED_API_KEY=...
SEC_EDGAR_USER_AGENT="AI-Trader research your-email@example.com"
```

Look-ahead rules:

- Quiver congressional trades must use disclosure or `FiledAfterDate`, not transaction date.
- `house_trade` and `senate_trade` signals use separate Quiver endpoints
  (`/live/housetrading`, `/live/senatetrading`) — the chamber field is set automatically.
- SEC 13F signals must use EDGAR `acceptanceDatetime` or filing date, not report period.
  `institutional_initiation` fires on the filing date when `previous_shares is None`.
- SEC Form 4 cluster signal (`sec_form4_cluster`) fires when ≥3 distinct `rptOwnerCik`
  values buy in the same disclosure window.
- Polygon market data is usable at its market timestamp.

### Social Sentiment

```env
X_BEARER_TOKEN=...
REDDIT_CLIENT_ID=...
REDDIT_CLIENT_SECRET=...
REDDIT_USER_AGENT="ai-trader-sentiment/0.1 by your_reddit_username"
```

### Live Fear/Greed

The live fear/greed composite works without a key using market data and Cboe's
daily put/call page. Add Alpha Vantage if you want the optional news-sentiment
component:

```env
ALPHA_VANTAGE_API_KEY=...
AI_TRADER_FEAR_GREED_SNAPSHOT_PATH=data/live/fear_greed.jsonl
AI_TRADER_FEAR_GREED_COMPONENT_MAX_AGE_MINUTES=240
AI_TRADER_FEAR_GREED_MIN_COMPONENTS=4
```

Fetch and persist the latest snapshot:

```powershell
ai-trader fear-greed
```

### IBKR

Use paper trading first:

```env
AI_TRADER_TRADING_MODE=paper
AI_TRADER_ALLOW_LIVE_TRADING=false
IBKR_HOST=127.0.0.1
IBKR_PORT=7497
IBKR_CLIENT_ID=1
IBKR_ACCOUNT=
IBKR_READONLY=false
```

Typical IBKR ports:

- TWS paper: `7497`
- TWS live: `7496`
- IB Gateway paper: often `4002`
- IB Gateway live: often `4001`

Enable API access inside TWS/Gateway before running `ai-trader ibkr-positions`.

## RAG Memory

The default example corpus lives in `examples/trader_corpus`.

```powershell
ai-trader rag-index
ai-trader rag-query --query "late-cycle euphoria breaks and liquidity tightens" --k 3
```

To use RAG in the Final Reasoner:

```powershell
$env:AI_TRADER_RAG_ENABLED="true"
ai-trader reason --bundle-file .\examples\sample_signal_bundle.json
```

For your own trader corpus, place `.txt` files in a folder and set:

```env
AI_TRADER_RAG_CORPUS_DIR=path/to/trader_corpus
AI_TRADER_RAG_INDEX_DIR=data/rag/trader_memory
```

## Train On Local Data

Local training lets your own closed trades calibrate the Final Reasoner's conviction and sizing. It does not replace the LLM, smart-money scorer, or safety guardrails. The learned calibrator can cap risky plans when similar local setups historically performed poorly.

Train from JSONL examples:

```powershell
ai-trader train local `
  --examples-file .\examples\sample_training_examples.jsonl `
  --model-out data\models\local_calibrator.json
```

Each JSONL line is one closed outcome with:

- `signal_bundle`: the exact `SignalBundle` available at decision time
- `trade_plan`: the plan originally produced
- `pnl_pct`: realized return, such as `-0.07` for `-7%`
- `narrative`: optional `NarrativeIntelligence`
- `metadata`: optional audit details

Enable it in `.env`:

```env
AI_TRADER_LOCAL_TRAINING_ENABLED=true
AI_TRADER_LOCAL_CALIBRATOR_PATH=data/models/local_calibrator.json
```

Then run the reasoner normally:

```powershell
ai-trader reason --bundle-file .\examples\sample_signal_bundle.json
```

The output `guardrails` field will include the local calibrator's expected P&L and any conviction or size caps it applied.

## Common Commands

```powershell
ai-trader status
ai-trader gui
ai-trader analyze-news --ticker MSFT --headline "Earnings beat" --body-file .\examples\sample_news.txt
ai-trader reason --bundle-file .\examples\sample_signal_bundle.json
ai-trader train local --examples-file .\examples\sample_training_examples.jsonl --model-out data\models\local_calibrator.json
ai-trader train backtest --examples-file logs\training_examples.jsonl --start-date 2025-01-01 --output logs\training_backtest_recent.json
ai-trader ibkr-positions
ai-trader backtest run --start 2022-01-01 --end 2024-12-31 --events-file .\examples\sample_events.jsonl --starting-balance 10000 --cash-fraction 0.02 --out result.json
ai-trader backtest monte-carlo --result-file result.json --n-sims 10000
ai-trader review-nightly --outcomes-file outcomes.jsonl
```

Run the full weekly evolution cycle (discovery → source implementation → ingestion → training → promotion gate):

```powershell
python scripts\weekly_evolution.py
# Skip the source implementation step:
python scripts\weekly_evolution.py --skip-implementation
```

Run overnight ingest only:

```powershell
python scripts\ingest_training_data.py --out logs\training_examples.jsonl
```

Fast local ingest knobs for high-bandwidth machines:

```powershell
python scripts\ingest_training_data.py `
  --out logs\training_examples.jsonl `
  --source-workers 12 `
  --price-workers 16 `
  --cache-dir data\cache `
  --profile-out logs\ingestion_profile.json
```

Environment equivalents:

```env
AI_TRADER_INGESTION_SOURCE_WORKERS=12
AI_TRADER_INGESTION_PRICE_WORKERS=16
AI_TRADER_INGESTION_TICKER_WORKERS=8
AI_TRADER_INGESTION_HTTP_CONNECTIONS=64
AI_TRADER_INGESTION_WRITE_BUFFER=32768
AI_TRADER_INGESTION_CACHE_DIR=data/cache
AI_TRADER_INGESTION_PROFILE_PATH=logs/ingestion_profile.json
```

When these worker counts are not set, ingestion auto-detects CPU cores, available
RAM, and network bandwidth. Bandwidth is measured with a short download probe and
cached for 24 hours in `data/cache/hardware_profile.json`. To pin or disable the
network portion of the detector:

```env
AI_TRADER_INGESTION_NETWORK_MBPS=250
AI_TRADER_INGESTION_NETWORK_PROBE=0
```

The profile report lists per-source timings, cache hits/misses, and the slowest
stages so you can raise or lower workers based on actual provider throttling.

## Label Review

Each ingest run auto-labels every training example with an outcome tier and signal quality
tier. Examples where the labeler's confidence is low are queued for human review in
`data/review_queue.jsonl`. The end of every ingest run prints how many items are pending.

Check queue status:

```powershell
python scripts\review_labels.py --status
```

Step through pending items interactively:

```powershell
python scripts\review_labels.py
```

Shortcuts during review:

| Key | Action |
|---|---|
| `Enter` | Accept the auto-label as-is |
| `o sw\|w\|n\|l\|sl` | Override outcome label (strong_win / win / neutral / loss / strong_loss) |
| `q h\|m\|l` | Override signal quality (high / medium / low) |
| `s` | Skip — leave in queue for later |
| `x` | Exit and save progress |

Confirmed labels are appended to `logs/human_labeled_examples.jsonl`. To batch-confirm all
auto-labels without prompting (e.g. for a first-run baseline):

```powershell
python scripts\review_labels.py --auto-confirm
```

Label fields on every `LocalTrainingExample`:

| Field | Values | Notes |
|---|---|---|
| `outcome_label` | `strong_win` `win` `neutral` `loss` `strong_loss` | Based on realized `pnl_pct` |
| `signal_quality` | `high` `medium` `low` | Based on signal count, avg confidence, combined strength |
| `label_confidence` | 0–1 | Labeler's confidence in `signal_quality` assignment |
| `label_source` | `auto` `human` `none` | `none` = unlabeled (old examples); `human` = reviewer-confirmed |
| `needs_review` | bool | True when sent to review queue |

## Local GUI

Launch the local frontend:

```powershell
ai-trader gui
```

Open this URL if the browser does not open automatically:

```text
http://127.0.0.1:8787
```

Run on a different port or keep the browser closed:

```powershell
ai-trader gui --port 8790 --no-open-browser
```

The GUI is a full local control surface:

- **Dashboard:** provider readiness, training-example counts, pending review count,
  hardware-tuned ingestion workers, cache paths, and recent artifacts.
- **Workbench:** whitelisted CLI workflows with generated forms and command output.
  Long-running jobs such as ingestion stream logs live.
- **Label Review:** queue browser for low-confidence labels, with accept, skip,
  outcome override, and signal-quality override controls.
- **Artifacts:** preview recent JSON, JSONL, log, and text files from `logs/`,
  `data/models/`, and `data/cache/`.

The workbench can run status, news analysis, final reasoning, local training,
RAG indexing/querying, IBKR position checks, trade-plan dry runs/orders,
backtests, Monte Carlo, ingestion, nightly review, build loop, autopilot, and
background bridge startup.

The frontend talks to these local endpoints:

| Endpoint | Purpose |
|---|---|
| `/api/overview` | Dashboard metrics, provider status, hardware profile, recent artifacts |
| `/api/actions` | Whitelisted action metadata and form fields |
| `/api/run` | Blocking command execution |
| `/api/stream` | Live command output via server-sent events |
| `/api/review` | Pending label-review items |
| `/api/review/decide` | Accept, skip, or override a queued label |
| `/api/artifact?path=...` | Safe in-repository artifact preview |

Trade actions still use the same `.env`, IBKR, and live-trading safety gates as
the CLI. Leave Shares blank to let the bot size from balance. Use Starting
Balance as a simulated capital/risk budget for dry runs and backtests.

The GUI writes process output to the same `logs/` directory as the CLI. If you
start it from another shell and need diagnostics, check `logs/ai_trader.log` and
any `logs/gui_<action>_*.log` files created by background actions.

## Balance-Based Trading

Order size is automatic when `--shares` is omitted. The bot reads IBKR available funds,
gets a reference price, and sizes the order from:

```text
available balance * cash fraction * plan conviction * plan size multiplier
```

For dry runs without connecting to IBKR market data, state the starting balance and
reference price:

```powershell
ai-trader trade `
  --plan-file .\logs\trade_plan.json `
  --starting-balance 10000 `
  --reference-price 400 `
  --cash-fraction 0.02 `
  --dry-run
```

For paper/live execution, omit `--reference-price` so IBKR supplies the quote. If you pass
`--starting-balance`, it acts as a cap on the broker balance used for sizing.

## Historical Event Replay

Backtests can consume look-ahead-safe smart-money events from JSONL:

```powershell
ai-trader backtest run `
  --start 2022-01-01 `
  --end 2022-12-31 `
  --events-file .\examples\sample_events.jsonl `
  --starting-balance 10000 `
  --cash-fraction 0.02 `
  --out result.json
```

When `--tickers` is omitted, the backtest automatically derives the ticker universe from
the replay events available in the selected date range. Use `--tickers` only when you want
to explicitly override or limit that universe.

Each line in the event file is one `ReplayEvent`. Supported `event_type` values:

- `congressional_trade`: contains a `congressional_trade` payload matching the `CongressionalTrade` model. Its usable date is `disclosure_date`.
- `13f_change`: contains a `thirteen_f_change` payload matching the `ThirteenFPositionChange` model. Its usable date is `current.filing_date`.

The replay loader only exposes events with `effective_date <= current_replay_date`. This is the core no-look-ahead guard for backtests.

Backtest risk controls are configurable:

```powershell
ai-trader backtest run `
  --start 2022-01-01 `
  --end 2022-12-31 `
  --events-file .\examples\sample_events.jsonl `
  --train-window-days 0 `
  --test-window-days 364 `
  --step-days 364 `
  --max-holding-days 30 `
  --stop-loss-pct 0.08 `
  --take-profit-pct 0.20 `
  --starting-balance 25000 `
  --cash-fraction 0.05 `
  --out logs\event_backtest_result_risk.json
```

Backtest results include per-trade `quantity`, `notional`, `pnl_amount`,
`balance_before`, `balance_after`, and `account_return`, plus starting and ending
balances in metadata.

## Training Strategy Backtest

The local training backtest searches deterministic policy rules over your historical
`LocalTrainingExample` JSONL and ranks them by monthly equal-weight P&L, Sharpe, drawdown,
and sample coverage:

```powershell
ai-trader train backtest `
  --examples-file logs\training_examples.jsonl `
  --start-date 2025-01-01 `
  --min-trades 50 `
  --min-active-months 3 `
  --min-trades-per-month 5 `
  --output logs\training_backtest_recent.json
```

Use `--split-date YYYY-MM-DD` to include train/test metrics. Treat high P&L with sparse
months or no out-of-sample coverage as a research lead, not an execution rule.

## C++ Bridge

The bridge is intended for WSL2/Linux because it uses POSIX shared memory.

```bash
cmake -S bridge -B bridge/build
cmake --build bridge/build
./bridge/build/consumer
```

Python side:

```powershell
ai-trader bridge-serve
```

## Build Loop

GitHub PR generation requires:

- `gh auth login`
- a remote named `origin`
- GitHub token permissions for contents and pull requests
- branch protection requiring human review

Run locally:

```powershell
ai-trader build-loop run --start 2022-01-01 --end 2024-12-31 --events-file .\examples\sample_events.jsonl --max-proposals 2
```

The workflow file `.github/workflows/build_loop.yml` runs the same cycle nightly in GitHub Actions.
