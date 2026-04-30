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
- SEC 13F signals must use EDGAR `acceptanceDatetime` or filing date, not report period.
- Polygon market data is usable at its market timestamp.

### Social Sentiment

```env
X_BEARER_TOKEN=...
REDDIT_CLIENT_ID=...
REDDIT_CLIENT_SECRET=...
REDDIT_USER_AGENT="ai-trader-sentiment/0.1 by your_reddit_username"
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
ai-trader analyze-news --ticker MSFT --headline "Earnings beat" --body-file .\examples\sample_news.txt
ai-trader reason --bundle-file .\examples\sample_signal_bundle.json
ai-trader train local --examples-file .\examples\sample_training_examples.jsonl --model-out data\models\local_calibrator.json
ai-trader ibkr-positions
ai-trader backtest run --tickers AAPL --tickers MSFT --start 2022-01-01 --end 2024-12-31 --events-file .\examples\sample_events.jsonl --out result.json
ai-trader backtest monte-carlo --result-file result.json --n-sims 10000
ai-trader review-nightly --outcomes-file outcomes.jsonl
```

## Historical Event Replay

Backtests can consume look-ahead-safe smart-money events from JSONL:

```powershell
ai-trader backtest run `
  --tickers MSFT `
  --start 2022-01-01 `
  --end 2022-12-31 `
  --events-file .\examples\sample_events.jsonl `
  --out result.json
```

Each line in the event file is one `ReplayEvent`. Supported `event_type` values:

- `congressional_trade`: contains a `congressional_trade` payload matching the `CongressionalTrade` model. Its usable date is `disclosure_date`.
- `13f_change`: contains a `thirteen_f_change` payload matching the `ThirteenFPositionChange` model. Its usable date is `current.filing_date`.

The replay loader only exposes events with `effective_date <= current_replay_date`. This is the core no-look-ahead guard for backtests.

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
ai-trader build-loop run --tickers AAPL --tickers MSFT --start 2022-01-01 --end 2024-12-31 --max-proposals 2
```

The workflow file `.github/workflows/build_loop.yml` runs the same cycle nightly in GitHub Actions.
