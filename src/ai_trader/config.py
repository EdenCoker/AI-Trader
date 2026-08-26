from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, SecretStr

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency is declared, fallback keeps tests lightweight.
    load_dotenv = None

# Hardware-aware defaults for ingestion worker counts.
# The expensive bits are cached once per process, while explicit env overrides
# still win without forcing a probe.
@lru_cache(maxsize=1)
def _hw_profile():
    try:
        from ai_trader.ingestion.hardware import detect

        return detect()
    except Exception:
        return None


def _hw_source_workers() -> int:
    override = _env_int_or_none("AI_TRADER_INGESTION_SOURCE_WORKERS")
    if override is not None:
        return override
    profile = _hw_profile()
    return profile.source_workers if profile is not None else 8


def _hw_price_workers() -> int:
    override = _env_int_or_none("AI_TRADER_INGESTION_PRICE_WORKERS")
    if override is not None:
        return override
    profile = _hw_profile()
    return profile.price_workers if profile is not None else 8


def _hw_ticker_workers() -> int:
    override = _env_int_or_none("AI_TRADER_INGESTION_TICKER_WORKERS")
    if override is not None:
        return override
    profile = _hw_profile()
    return profile.ticker_workers if profile is not None else 8


def _hw_http_connections() -> int:
    override = _env_int_or_none("AI_TRADER_INGESTION_HTTP_CONNECTIONS")
    if override is not None:
        return override
    profile = _hw_profile()
    return profile.http_connections if profile is not None else 32


def _hw_write_buffer() -> int:
    override = _env_int_or_none("AI_TRADER_INGESTION_WRITE_BUFFER")
    if override is not None:
        return override
    profile = _hw_profile()
    return profile.write_buffer if profile is not None else 8192


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def _secret(name: str) -> SecretStr | None:
    value = _env(name)
    return SecretStr(value) if value else None


def _env_float(name: str, default: float) -> float:
    value = _env(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    value = _env(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_int_or_none(name: str) -> int | None:
    value = _env(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _env_bool(name: str, default: bool = False) -> bool:
    value = _env(name)
    if value is None:
        return default
    value = value.strip().casefold()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


class AppSettings(BaseModel):
    """Runtime settings loaded from environment variables."""

    env: str = Field(default_factory=lambda: _env("AI_TRADER_ENV", "dev") or "dev")
    log_level: str = Field(default_factory=lambda: _env("LOG_LEVEL", "INFO") or "INFO")
    trading_mode: str = Field(default_factory=lambda: _env("AI_TRADER_TRADING_MODE", "paper") or "")
    allow_live_trading: bool = Field(
        default_factory=lambda: _env_bool("AI_TRADER_ALLOW_LIVE_TRADING", default=False)
    )

    quiver_api_key: SecretStr | None = Field(default_factory=lambda: _secret("QUIVER_API_KEY"))
    polygon_api_key: SecretStr | None = Field(default_factory=lambda: _secret("POLYGON_API_KEY"))
    fred_api_key: SecretStr | None = Field(default_factory=lambda: _secret("FRED_API_KEY"))
    x_bearer_token: SecretStr | None = Field(default_factory=lambda: _secret("X_BEARER_TOKEN"))
    reddit_client_id: SecretStr | None = Field(default_factory=lambda: _secret("REDDIT_CLIENT_ID"))
    reddit_client_secret: SecretStr | None = Field(
        default_factory=lambda: _secret("REDDIT_CLIENT_SECRET")
    )
    reddit_user_agent: str = Field(
        default_factory=lambda: _env("REDDIT_USER_AGENT", "ai-trader-sentiment/0.1") or ""
    )
    sec_edgar_user_agent: str = Field(
        default_factory=lambda: _env(
            "SEC_EDGAR_USER_AGENT", "AI-Trader research (set SEC_EDGAR_USER_AGENT)"
        )
        or ""
    )
    alpha_vantage_api_key: SecretStr | None = Field(
        default_factory=lambda: _secret("ALPHA_VANTAGE_API_KEY")
    )
    bls_api_key: SecretStr | None = Field(default_factory=lambda: _secret("BLS_API_KEY"))
    eia_api_key: SecretStr | None = Field(default_factory=lambda: _secret("EIA_API_KEY"))
    newsapi_api_key: SecretStr | None = Field(default_factory=lambda: _secret("NEWSAPI_API_KEY"))
    fear_greed_snapshot_path: Path = Field(
        default_factory=lambda: Path(
            _env("AI_TRADER_FEAR_GREED_SNAPSHOT_PATH", "data/live/fear_greed.jsonl") or ""
        )
    )
    fear_greed_component_max_age_minutes: int = Field(
        default_factory=lambda: _env_int("AI_TRADER_FEAR_GREED_COMPONENT_MAX_AGE_MINUTES", 240)
    )
    fear_greed_min_components: int = Field(
        default_factory=lambda: _env_int("AI_TRADER_FEAR_GREED_MIN_COMPONENTS", 4)
    )
    news_enabled: bool = Field(
        default_factory=lambda: _env_bool("AI_TRADER_NEWS_ENABLED", default=True)
    )
    news_worldmonitor_enabled: bool = Field(
        default_factory=lambda: _env_bool("AI_TRADER_NEWS_WORLDMONITOR_ENABLED", default=True)
    )
    worldmonitor_base_url: str = Field(
        default_factory=lambda: _env(
            "AI_TRADER_WORLDMONITOR_BASE_URL", "https://api.worldmonitor.app"
        )
        or "https://api.worldmonitor.app"
    )
    news_variant: str = Field(
        default_factory=lambda: _env("AI_TRADER_NEWS_VARIANT", "finance") or "finance"
    )
    news_max_age_hours: int = Field(
        default_factory=lambda: _env_int("AI_TRADER_NEWS_MAX_AGE_HOURS", 96)
    )
    news_archive_path: Path = Field(
        default_factory=lambda: Path(
            _env("AI_TRADER_NEWS_ARCHIVE_PATH", "data/live/news_archive.jsonl") or ""
        )
    )
    openai_api_key: SecretStr | None = Field(default_factory=lambda: _secret("OPENAI_API_KEY"))
    openai_base_url: str = Field(
        default_factory=lambda: _env(
            "OPENAI_BASE_URL", _env("AI_TRADER_OPENAI_BASE_URL", "https://api.openai.com/v1")
        )
        or "https://api.openai.com/v1"
    )
    final_reasoner_model: str = Field(
        default_factory=lambda: _env("AI_TRADER_FINAL_REASONER_MODEL", "gpt-4.5") or "gpt-4.5"
    )
    llm_backend: str = Field(default_factory=lambda: _env("AI_TRADER_LLM_BACKEND", "openai") or "")
    llm_model: str = Field(
        default_factory=lambda: _env("AI_TRADER_LLM_MODEL")
        or _env("AI_TRADER_FINAL_REASONER_MODEL", "gpt-4.5")
        or ""
    )
    ollama_base_url: str = Field(
        default_factory=lambda: _env("OLLAMA_HOST", _env("AI_TRADER_OLLAMA_BASE_URL", "http://localhost:11434"))
        or "http://localhost:11434"
    )
    llm_timeout_s: float = Field(
        default_factory=lambda: _env_float("AI_TRADER_LLM_TIMEOUT_S", 60.0)
    )

    rag_enabled: bool = Field(
        default_factory=lambda: _env_bool("AI_TRADER_RAG_ENABLED", default=False)
    )
    rag_index_dir: Path = Field(
        default_factory=lambda: Path(
            _env("AI_TRADER_RAG_INDEX_DIR", "data/rag/trader_memory") or ""
        )
    )
    rag_corpus_dir: Path = Field(
        default_factory=lambda: Path(
            _env("AI_TRADER_RAG_CORPUS_DIR", "examples/trader_corpus") or ""
        )
    )
    embeddings_backend: str = Field(
        default_factory=lambda: _env("AI_TRADER_EMBEDDINGS_BACKEND", "local") or ""
    )
    local_embedding_model: str = Field(
        default_factory=lambda: _env("AI_TRADER_LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2") or ""
    )
    openai_embedding_model: str = Field(
        default_factory=lambda: _env(
            "AI_TRADER_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
        )
        or ""
    )
    local_training_enabled: bool = Field(
        default_factory=lambda: _env_bool("AI_TRADER_LOCAL_TRAINING_ENABLED", default=False)
    )
    local_calibrator_path: Path = Field(
        default_factory=lambda: Path(
            _env("AI_TRADER_LOCAL_CALIBRATOR_PATH", "data/models/local_calibrator.json") or ""
        )
    )
    local_calibrator_short_path: Path = Field(
        default_factory=lambda: Path(
            _env(
                "AI_TRADER_LOCAL_CALIBRATOR_SHORT_PATH",
                "data/models/local_calibrator_short.json",
            )
            or ""
        )
    )
    local_calibrator_medium_path: Path = Field(
        default_factory=lambda: Path(
            _env(
                "AI_TRADER_LOCAL_CALIBRATOR_MEDIUM_PATH",
                "data/models/local_calibrator_medium.json",
            )
            or ""
        )
    )
    local_calibrator_long_path: Path = Field(
        default_factory=lambda: Path(
            _env(
                "AI_TRADER_LOCAL_CALIBRATOR_LONG_PATH",
                "data/models/local_calibrator_long.json",
            )
            or ""
        )
    )
    fmp_base_url: str = Field(
        default_factory=lambda: _env(
            "FMP_BASE_URL", "https://financialmodelingprep.com/api/v3"
        )
        or "https://financialmodelingprep.com/api/v3"
    )

    ibkr_host: str = Field(default_factory=lambda: _env("IBKR_HOST", "127.0.0.1") or "127.0.0.1")
    ibkr_port: int = Field(default_factory=lambda: _env_int("IBKR_PORT", 7497))
    ibkr_client_id: int = Field(default_factory=lambda: _env_int("IBKR_CLIENT_ID", 1))
    ibkr_account: str | None = Field(default_factory=lambda: _env("IBKR_ACCOUNT"))
    ibkr_readonly: bool = Field(default_factory=lambda: _env_bool("IBKR_READONLY", default=False))
    polygon_cache_dir: Path = Field(
        default_factory=lambda: Path(_env("POLYGON_CACHE_DIR", ".polygon_cache") or "")
    )
    polygon_rate_limit_rpm: int = Field(
        default_factory=lambda: _env_int("POLYGON_RATE_LIMIT_RPM", 5)
    )
    ingestion_cache_dir: Path = Field(
        default_factory=lambda: Path(_env("AI_TRADER_INGESTION_CACHE_DIR", "data/cache") or "")
    )
    ingestion_profile_path: Path = Field(
        default_factory=lambda: Path(
            _env("AI_TRADER_INGESTION_PROFILE_PATH", "logs/ingestion_profile.json") or ""
        )
    )
    ingestion_source_workers: int = Field(default_factory=_hw_source_workers)
    ingestion_price_workers: int = Field(default_factory=_hw_price_workers)
    ingestion_ticker_workers: int = Field(default_factory=_hw_ticker_workers)
    ingestion_http_connections: int = Field(default_factory=_hw_http_connections)
    ingestion_write_buffer: int = Field(default_factory=_hw_write_buffer)

    database_url: str = Field(
        default_factory=lambda: _env(
            "DATABASE_URL",
            "postgresql+psycopg://ai_trader:ai_trader@localhost:5432/ai_trader",
        )
        or ""
    )
    faiss_index_path: Path = Field(
        default_factory=lambda: Path(
            _env("FAISS_INDEX_PATH", "data/faiss/trader_memory.index") or ""
        )
    )

    def provider_status(self) -> dict[str, bool]:
        """Return which external providers have enough configuration to be initialized."""

        return {
            "quiver": self.quiver_api_key is not None,
            "polygon": self.polygon_api_key is not None,
            "fred": self.fred_api_key is not None,
            "x": self.x_bearer_token is not None,
            "reddit": self.reddit_client_id is not None and self.reddit_client_secret is not None,
            "sec_edgar": "set SEC_EDGAR_USER_AGENT" not in self.sec_edgar_user_agent,
            "live_fear_greed": True,
            "openai": self.openai_api_key is not None,
            "ollama": bool(self.ollama_base_url),
            "ibkr": bool(self.ibkr_host) and bool(self.ibkr_port),
            "alpha_vantage": self.alpha_vantage_api_key is not None,
            "bls": self.bls_api_key is not None,
            "eia": self.eia_api_key is not None,
            "newsapi": self.newsapi_api_key is not None,
        }

    def redacted(self) -> dict[str, Any]:
        """Diagnostics-safe settings dump."""

        data = self.model_dump(mode="json")
        for key in (
            "quiver_api_key",
            "polygon_api_key",
            "fred_api_key",
            "x_bearer_token",
            "reddit_client_id",
            "reddit_client_secret",
            "alpha_vantage_api_key",
            "bls_api_key",
            "eia_api_key",
            "newsapi_api_key",
            "openai_api_key",
        ):
            if data.get(key):
                data[key] = "**********"
        return data


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    if load_dotenv is not None:
        load_dotenv()
    return AppSettings()
