from __future__ import annotations

import mmap
from typing import Any

from ai_trader.bridge.protocol import MSG_SIZE, serialize
from ai_trader.intelligence.trade_plan import TradePlan


class BridgeTimeoutError(RuntimeError):
    """Raised when AlphaEngine does not acknowledge a shared-memory write."""


class SharedMemWriter:
    SHM_NAME = "/ai_trader_bridge"
    SEM_READY = "/ai_trader_ready"
    SEM_ACK = "/ai_trader_ack"
    MSG_SIZE = MSG_SIZE

    def __init__(self) -> None:
        self._posix_ipc: Any = None
        self._shm: Any = None
        self._map: mmap.mmap | None = None
        self._sem_ready: Any = None
        self._sem_ack: Any = None

    def __enter__(self) -> SharedMemWriter:
        import posix_ipc

        self._posix_ipc = posix_ipc
        self._shm = posix_ipc.SharedMemory(
            self.SHM_NAME,
            flags=posix_ipc.O_CREAT,
            mode=0o600,
            size=self.MSG_SIZE,
        )
        self._map = mmap.mmap(self._shm.fd, self.MSG_SIZE)
        self._shm.close_fd()
        self._sem_ready = posix_ipc.Semaphore(self.SEM_READY, flags=posix_ipc.O_CREAT, initial_value=0)
        self._sem_ack = posix_ipc.Semaphore(self.SEM_ACK, flags=posix_ipc.O_CREAT, initial_value=0)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._map is not None:
            self._map.close()
        for semaphore in (self._sem_ready, self._sem_ack):
            if semaphore is not None:
                semaphore.close()

    def write(self, plan: TradePlan, ticker: str, timeout_ms: int = 5_000) -> None:
        if self._map is None or self._sem_ready is None or self._sem_ack is None:
            raise RuntimeError("SharedMemWriter must be used as a context manager")

        payload = serialize(plan, ticker)
        self._map.seek(0)
        self._map.write(payload)
        self._map.flush()
        self._sem_ready.release()
        try:
            self._sem_ack.acquire(timeout=timeout_ms / 1000)
        except Exception as exc:
            raise BridgeTimeoutError(f"AlphaEngine ack timed out after {timeout_ms}ms") from exc

