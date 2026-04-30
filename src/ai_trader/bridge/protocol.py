from __future__ import annotations

import struct
from datetime import datetime, time, timezone
from typing import Any

from ai_trader.domain.signals import SignalDirection
from ai_trader.intelligence.trade_plan import TradePlan


STRUCT_FORMAT = "<Q16sb3xffi128sB3x"
MSG_SIZE = struct.calcsize(STRUCT_FORMAT)
if MSG_SIZE != 172:  # pragma: no cover
    raise RuntimeError(f"Bridge protocol size mismatch: {MSG_SIZE}")


def serialize(plan: TradePlan, ticker: str | None = None) -> bytes:
    ticker_value = _fixed_bytes((ticker or plan.ticker).upper(), 16)
    exit_trigger = _fixed_bytes(plan.exit_trigger, 128)
    timestamp_ns = _timestamp_ns(plan.as_of)
    direction = _direction_to_wire(plan.direction)

    without_checksum = struct.pack(
        "<Q16sb3xffi128s",
        timestamp_ns,
        ticker_value,
        direction,
        float(plan.conviction),
        float(plan.size_multiplier),
        int(plan.holding_period_days),
        exit_trigger,
    )
    checksum = _checksum(without_checksum)
    return struct.pack(
        STRUCT_FORMAT,
        timestamp_ns,
        ticker_value,
        direction,
        float(plan.conviction),
        float(plan.size_multiplier),
        int(plan.holding_period_days),
        exit_trigger,
        checksum,
    )


def deserialize(data: bytes) -> dict[str, Any]:
    if len(data) < MSG_SIZE:
        raise ValueError(f"Bridge message must be at least {MSG_SIZE} bytes")
    chunk = data[:MSG_SIZE]
    fields = struct.unpack(STRUCT_FORMAT, chunk)
    expected_checksum = _checksum(chunk[:168])
    if fields[7] != expected_checksum:
        raise ValueError("Bridge message checksum mismatch")
    return {
        "timestamp_ns": fields[0],
        "ticker": _read_c_string(fields[1]),
        "direction": _wire_to_direction(fields[2]),
        "conviction": fields[3],
        "size_multiplier": fields[4],
        "holding_period_days": fields[5],
        "exit_trigger": _read_c_string(fields[6]),
        "checksum": fields[7],
    }


def _checksum(data: bytes) -> int:
    value = 0
    for byte in data:
        value ^= byte
    return value


def _fixed_bytes(value: str, size: int) -> bytes:
    raw = value.encode("utf-8", errors="ignore")[: size - 1]
    return raw + b"\0" * (size - len(raw))


def _read_c_string(value: bytes) -> str:
    return value.split(b"\0", 1)[0].decode("utf-8", errors="ignore")


def _timestamp_ns(value) -> int:
    dt = datetime.combine(value, time.min, tzinfo=timezone.utc)
    return int(dt.timestamp() * 1_000_000_000)


def _direction_to_wire(direction: SignalDirection) -> int:
    if direction is SignalDirection.LONG:
        return 1
    if direction is SignalDirection.SHORT:
        return -1
    return 0


def _wire_to_direction(value: int) -> SignalDirection:
    if value > 0:
        return SignalDirection.LONG
    if value < 0:
        return SignalDirection.SHORT
    return SignalDirection.NEUTRAL

