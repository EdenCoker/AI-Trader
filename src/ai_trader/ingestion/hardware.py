"""
Hardware auto-tuning for the ingestion pipeline.

Detects CPU cores, RAM, and network bandwidth at runtime and derives
conservative but efficient worker counts for the I/O-heavy ingestion pipeline.

All values can be overridden via environment variables or CLI flags. This
module only supplies the defaults when nothing is configured.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter, time

log = logging.getLogger(__name__)

DEFAULT_NETWORK_PROBE_BYTES = 1_000_000
DEFAULT_NETWORK_PROBE_TIMEOUT_S = 2.5
DEFAULT_NETWORK_CACHE_TTL_S = 24 * 60 * 60
DEFAULT_NETWORK_PROBE_URLS = (
    "https://speed.cloudflare.com/__down?bytes=1000000",
)


@dataclass(frozen=True)
class HardwareProfile:
    physical_cores: int
    logical_cores: int
    ram_gb: float
    ram_available_gb: float
    network_mbps: float | None
    network_tier: str

    # Derived worker counts
    source_workers: int     # concurrent data-source loaders (I/O-bound)
    price_workers: int      # concurrent Polygon price fetches (I/O-bound)
    ticker_workers: int     # concurrent per-ticker API workers within each source
    http_connections: int   # shared async HTTP connection budget
    write_buffer: int       # examples to buffer before flushing to disk

    def log_summary(self) -> None:
        if self.network_mbps is None:
            network = f"network={self.network_tier}"
        else:
            network = f"network={self.network_mbps:.1f} Mbps ({self.network_tier})"
        log.info(
            "Hardware profile: %d physical / %d logical cores | "
            "%.1f GB RAM (%.1f GB free) | %s | "
            "source_workers=%d price_workers=%d ticker_workers=%d "
            "http_connections=%d write_buffer=%d",
            self.physical_cores,
            self.logical_cores,
            self.ram_gb,
            self.ram_available_gb,
            network,
            self.source_workers,
            self.price_workers,
            self.ticker_workers,
            self.http_connections,
            self.write_buffer,
        )


def detect(
    *,
    probe_network: bool | None = None,
    cache_path: Path | None = None,
    force_network_probe: bool = False,
) -> HardwareProfile:
    """
    Probe the current machine and return a tuned HardwareProfile.

    Strategy:
    - Source loaders are I/O-bound network tasks. We can run many more than
      physical cores, but still cap by RAM and bandwidth tier.
    - Per-ticker workers inside each source loader scale with cores, then cap
      harder on slower links because provider latency dominates.
    - Price workers: Polygon is the highest-throughput API. It gets a separate
      pool that can expand on fast links and shrink on slow links.
    - Network bandwidth is read from AI_TRADER_INGESTION_NETWORK_MBPS when set,
      otherwise from a 24-hour local cache, otherwise from a short download
      probe. Failed probes fall back to the previous CPU/RAM-only behavior.
    """
    physical_cores = _physical_cores()
    logical_cores = _logical_cores()
    ram_gb, ram_available_gb = _ram_gb()
    network_mbps = _network_mbps(
        probe_network=probe_network,
        cache_path=cache_path,
        force=force_network_probe,
    )
    network_tier = _network_tier(network_mbps)

    # I/O worker counts are network-bound, not CPU-bound. The RAM cap keeps
    # large machines from over-subscribing when available memory is tight.
    max_by_ram = max(4, int(ram_available_gb * 1024 / 64))  # 64 MB per worker
    multiplier = _network_multiplier(network_tier)
    caps = _network_caps(network_tier)

    source_workers = min(
        caps["source_workers"],
        max_by_ram,
        max(4, int(max(8, logical_cores * 4) * multiplier)),
    )
    price_workers = min(
        caps["price_workers"],
        max_by_ram,
        max(4, int(max(8, logical_cores * 3) * multiplier)),
    )
    ticker_workers = min(
        caps["ticker_workers"],
        max_by_ram,
        max(2, int(max(4, logical_cores * 2) * multiplier)),
    )
    http_connections = min(
        caps["http_connections"],
        max(8, max_by_ram * 4),
        max(source_workers + price_workers + ticker_workers, int(32 * multiplier)),
    )

    # Each example is about 3 KB. Reserve up to 256 MB, or less on small boxes.
    buffer_mb = min(256.0, max(16.0, ram_available_gb * 1024 * 0.05))
    write_buffer = min(65_536, max(512, int(buffer_mb * 1024 / 3)))

    return HardwareProfile(
        physical_cores=physical_cores,
        logical_cores=logical_cores,
        ram_gb=ram_gb,
        ram_available_gb=ram_available_gb,
        network_mbps=network_mbps,
        network_tier=network_tier,
        source_workers=source_workers,
        price_workers=price_workers,
        ticker_workers=ticker_workers,
        http_connections=http_connections,
        write_buffer=write_buffer,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _logical_cores() -> int:
    try:
        return os.cpu_count() or 1
    except Exception:
        return 1


def _physical_cores() -> int:
    # psutil gives physical core count; fall back to logical if unavailable.
    try:
        import psutil  # type: ignore

        count = psutil.cpu_count(logical=False)
        return count if count else _logical_cores()
    except ImportError:
        pass
    # Rough heuristic: assume hyper-threading means physical ~= logical / 2.
    return max(1, _logical_cores() // 2)


def _ram_gb() -> tuple[float, float]:
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        return vm.total / 1024**3, vm.available / 1024**3
    except ImportError:
        pass
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            lines = {
                line.split(":")[0].strip(): line.split(":")[1].strip()
                for line in handle
                if ":" in line
            }
        total_kb = int(lines.get("MemTotal", "0 kB").split()[0])
        avail_kb = int(lines.get("MemAvailable", "0 kB").split()[0])
        return total_kb / 1024**2, avail_kb / 1024**2
    except Exception:
        pass
    return 8.0, 4.0


def _network_mbps(
    *,
    probe_network: bool | None,
    cache_path: Path | None,
    force: bool,
) -> float | None:
    configured = _configured_network_mbps()
    if configured is not None:
        return configured

    cache_path = cache_path or _network_cache_path()
    ttl_s = _env_int("AI_TRADER_INGESTION_NETWORK_CACHE_TTL_SECONDS", DEFAULT_NETWORK_CACHE_TTL_S)
    cached = None if force else _read_cached_network_mbps(cache_path, ttl_s=ttl_s)
    if cached is not None:
        return cached

    if probe_network is None:
        probe_network = _env_bool("AI_TRADER_INGESTION_NETWORK_PROBE", default=True)
    if not probe_network:
        return None

    measured = _probe_network_mbps()
    if measured is not None:
        _write_cached_network_mbps(cache_path, measured)
        return measured

    return _read_cached_network_mbps(cache_path, ttl_s=None)


def _configured_network_mbps() -> float | None:
    for name in ("AI_TRADER_INGESTION_NETWORK_MBPS", "AI_TRADER_NETWORK_MBPS"):
        value = os.getenv(name)
        if value is None or value.strip() == "":
            continue
        try:
            mbps = float(value)
        except ValueError:
            log.warning("Ignoring invalid %s=%r", name, value)
            return None
        return max(0.0, mbps)
    return None


def _network_cache_path() -> Path:
    raw = os.getenv("AI_TRADER_INGESTION_HARDWARE_CACHE")
    return Path(raw) if raw else Path("data/cache/hardware_profile.json")


def _read_cached_network_mbps(path: Path, *, ttl_s: int | None) -> float | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        measured_at = float(payload.get("measured_at", 0.0))
        if ttl_s is not None and time() - measured_at > ttl_s:
            return None
        mbps = float(payload["network_mbps"])
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None
    return max(0.0, mbps)


def _write_cached_network_mbps(path: Path, mbps: float) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"network_mbps": round(mbps, 3), "measured_at": time()},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    except OSError as exc:
        log.debug("Could not write hardware probe cache %s: %s", path, exc)


def _probe_network_mbps() -> float | None:
    try:
        import httpx
    except ImportError:
        return None

    urls = _network_probe_urls()
    target_bytes = _env_int("AI_TRADER_INGESTION_NETWORK_PROBE_BYTES", DEFAULT_NETWORK_PROBE_BYTES)
    timeout_s = _env_float(
        "AI_TRADER_INGESTION_NETWORK_PROBE_TIMEOUT_S",
        DEFAULT_NETWORK_PROBE_TIMEOUT_S,
    )

    for url in urls:
        try:
            started = perf_counter()
            total = 0
            with httpx.Client(timeout=timeout_s, follow_redirects=True) as client:
                with client.stream("GET", url) as response:
                    response.raise_for_status()
                    for chunk in response.iter_bytes():
                        total += len(chunk)
                        if total >= target_bytes:
                            break
            elapsed = perf_counter() - started
            if total >= 64 * 1024 and elapsed > 0:
                return (total * 8) / elapsed / 1_000_000
        except Exception as exc:
            log.debug("Network probe failed for %s: %s", url, exc)
    return None


def _network_probe_urls() -> tuple[str, ...]:
    raw = os.getenv("AI_TRADER_INGESTION_NETWORK_PROBE_URLS")
    if raw:
        urls = tuple(url.strip() for url in raw.split(",") if url.strip())
        if urls:
            return urls
    return DEFAULT_NETWORK_PROBE_URLS


def _network_tier(mbps: float | None) -> str:
    if mbps is None:
        return "unknown"
    if mbps < 25:
        return "slow"
    if mbps < 100:
        return "normal"
    if mbps < 500:
        return "fast"
    return "very_fast"


def _network_multiplier(tier: str) -> float:
    return {
        "slow": 0.5,
        "normal": 1.0,
        "fast": 1.5,
        "very_fast": 2.0,
    }.get(tier, 1.0)


def _network_caps(tier: str) -> dict[str, int]:
    return {
        "slow": {
            "source_workers": 12,
            "price_workers": 8,
            "ticker_workers": 6,
            "http_connections": 24,
        },
        "normal": {
            "source_workers": 32,
            "price_workers": 24,
            "ticker_workers": 12,
            "http_connections": 64,
        },
        "fast": {
            "source_workers": 64,
            "price_workers": 64,
            "ticker_workers": 24,
            "http_connections": 128,
        },
        "very_fast": {
            "source_workers": 96,
            "price_workers": 96,
            "ticker_workers": 32,
            "http_connections": 192,
        },
    }.get(
        tier,
        {
            "source_workers": 48,
            "price_workers": 32,
            "ticker_workers": 16,
            "http_connections": 64,
        },
    )


def _env_bool(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip().casefold()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default
