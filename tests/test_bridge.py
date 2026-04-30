from datetime import date
import struct

import pytest

from ai_trader.bridge.protocol import MSG_SIZE, STRUCT_FORMAT, _checksum, deserialize, serialize
from ai_trader.bridge.shm_writer import BridgeTimeoutError, SharedMemWriter
from ai_trader.domain.signals import SignalDirection
from ai_trader.intelligence.trade_plan import TradePlan


def _plan() -> TradePlan:
    return TradePlan(
        ticker="MSFT",
        as_of=date(2026, 1, 1),
        direction=SignalDirection.LONG,
        conviction=0.7,
        size_multiplier=1.2,
        holding_period_days=30,
        exit_trigger="close below invalidation",
    )


def test_serialize_has_expected_layout():
    data = serialize(_plan())
    assert len(data) == MSG_SIZE == 172
    fields = struct.unpack(STRUCT_FORMAT, data)
    assert fields[1].split(b"\0", 1)[0] == b"MSFT"
    assert fields[2] == 1
    assert fields[3] == pytest.approx(0.7)
    assert fields[4] == pytest.approx(1.2)
    assert fields[5] == 30
    assert fields[7] == _checksum(data[:168])


def test_checksum_catches_corruption():
    data = bytearray(serialize(_plan()))
    assert deserialize(bytes(data))["ticker"] == "MSFT"
    data[10] ^= 1
    with pytest.raises(ValueError):
        deserialize(bytes(data))


def test_writer_timeout_with_mock_semaphore():
    class FakeMap:
        def seek(self, value):
            pass

        def write(self, value):
            pass

        def flush(self):
            pass

    class Ready:
        def release(self):
            pass

    class Ack:
        def acquire(self, timeout):
            raise TimeoutError

    writer = SharedMemWriter()
    writer._map = FakeMap()
    writer._sem_ready = Ready()
    writer._sem_ack = Ack()

    with pytest.raises(BridgeTimeoutError):
        writer.write(_plan(), "MSFT", timeout_ms=1)


def test_shared_memory_roundtrip_skipped_without_posix_ipc():
    pytest.importorskip("posix_ipc")

