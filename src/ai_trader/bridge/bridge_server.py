from __future__ import annotations

import logging
from queue import Queue

from ai_trader.bridge.shm_writer import BridgeTimeoutError, SharedMemWriter
from ai_trader.intelligence.trade_plan import TradePlan

logger = logging.getLogger(__name__)


class BridgeServer:
    def __init__(self, queue: Queue[TradePlan] | None = None) -> None:
        self._queue = queue or Queue()

    @property
    def queue(self) -> Queue[TradePlan]:
        return self._queue

    def serve_forever(self) -> None:
        with SharedMemWriter() as writer:
            while True:
                plan = self._queue.get()
                try:
                    writer.write(plan, plan.ticker)
                    logger.info("wrote trade plan to shared memory: %s", plan.ticker)
                except BridgeTimeoutError:
                    logger.warning("shared-memory bridge timed out for %s", plan.ticker)

