"""Interactive human review for low-confidence auto-labeled training examples.

Usage:
    python scripts/review_labels.py
    python scripts/review_labels.py --queue data/review_queue.jsonl
    python scripts/review_labels.py --queue data/review_queue.jsonl --out logs/human_labeled_examples.jsonl
    python scripts/review_labels.py --status          # show queue stats only
    python scripts/review_labels.py --auto-confirm    # confirm all auto-labels without asking

Keyboard shortcuts during review:
    Enter        accept the auto-label as-is
    o <label>    override outcome label  (sw|w|n|l|sl)
    q <label>    override signal quality (h|m|l)
    s            skip this entry (leave in queue as unreviewed)
    x            exit and save progress
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo src is importable when running directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_trader.training.review_queue import (
    DEFAULT_HUMAN_OUT_PATH,
    DEFAULT_QUEUE_PATH,
    ReviewEntry,
    confirm_label,
    iter_queue,
    pending_count,
    rewrite_queue,
)

OUTCOME_SHORTCUTS = {
    "sw": "strong_win",
    "w": "win",
    "n": "neutral",
    "l": "loss",
    "sl": "strong_loss",
}
QUALITY_SHORTCUTS = {
    "h": "high",
    "m": "medium",
    "l": "low",
}


def _print_entry(idx: int, total: int, entry: ReviewEntry) -> None:
    ex = entry.example
    bundle = ex.signal_bundle
    plan = ex.trade_plan
    signals = bundle.signals

    print()
    print(f"{'─' * 70}")
    print(f"  [{idx}/{total}]  {bundle.ticker}  as_of={bundle.as_of}  pnl={ex.pnl_pct:+.2%}")
    print(f"{'─' * 70}")
    print(f"  Signals ({len(signals)}):", end="")
    if signals:
        for s in signals[:6]:
            print(f"\n    {s.name:<30} {s.direction.value:<7} str={s.strength:.2f} conf={s.confidence:.2f}", end="")
        if len(signals) > 6:
            print(f"\n    ... and {len(signals) - 6} more", end="")
    print()
    print(f"  Plan direction : {plan.direction.value}   conviction={plan.conviction:.2f}")
    print()
    print(f"  Auto outcome   : {entry.auto_outcome_label}")
    print(f"  Auto quality   : {entry.auto_signal_quality}   (labeler conf={entry.auto_label_confidence:.2f})")
    print(f"  Review reasons :")
    for r in entry.review_reasons:
        print(f"    • {r}")
    print()
    print("  [Enter]=accept  [o sw|w|n|l|sl]=override outcome  [q h|m|l]=override quality  [s]=skip  [x]=exit")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review low-confidence training labels")
    parser.add_argument(
        "--queue",
        type=Path,
        default=DEFAULT_QUEUE_PATH,
        help=f"Review queue JSONL (default: {DEFAULT_QUEUE_PATH})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_HUMAN_OUT_PATH,
        help=f"Output JSONL for confirmed examples (default: {DEFAULT_HUMAN_OUT_PATH})",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print queue statistics and exit",
    )
    parser.add_argument(
        "--auto-confirm",
        action="store_true",
        help="Accept all auto-labels without prompting (batch confirmation run)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    queue_path: Path = args.queue
    out_path: Path = args.out

    if args.status:
        total = sum(1 for _ in iter_queue(queue_path))
        pending = pending_count(queue_path)
        print(f"Queue : {queue_path}")
        print(f"Total : {total}")
        print(f"Pending review : {pending}")
        print(f"Reviewed       : {total - pending}")
        return

    entries = list(iter_queue(queue_path))
    pending = [e for e in entries if not e.reviewed]

    if not pending:
        print("No pending entries in the review queue.")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    confirmed_examples: list[str] = []

    print(f"\nStarting review of {len(pending)} pending item(s).")
    print("Output will be appended to:", out_path)

    for idx, entry in enumerate(pending, start=1):
        if not args.auto_confirm:
            _print_entry(idx, len(pending), entry)

        outcome = entry.auto_outcome_label
        quality = entry.auto_signal_quality
        notes = ""
        skipped = False

        if args.auto_confirm:
            pass  # keep auto values
        else:
            while True:
                raw = input("  > ").strip()
                if raw == "" or raw == "\n":
                    break  # accept
                if raw == "x":
                    print("  Exiting — saving progress so far.")
                    # Write confirmed examples collected so far
                    _flush(out_path, confirmed_examples)
                    rewrite_queue(entries, queue_path)
                    sys.exit(0)
                if raw == "s":
                    skipped = True
                    break
                parts = raw.split(None, 1)
                cmd = parts[0].lower()
                arg = parts[1].strip() if len(parts) > 1 else ""
                if cmd == "o":
                    mapped = OUTCOME_SHORTCUTS.get(arg) or arg
                    if mapped in ("strong_win", "win", "neutral", "loss", "strong_loss"):
                        outcome = mapped
                        print(f"  → outcome set to: {outcome}")
                    else:
                        print(f"  Unknown outcome shortcut '{arg}'. Use: sw w n l sl")
                elif cmd == "q":
                    mapped = QUALITY_SHORTCUTS.get(arg) or arg
                    if mapped in ("high", "medium", "low"):
                        quality = mapped
                        print(f"  → quality set to: {quality}")
                    else:
                        print(f"  Unknown quality shortcut '{arg}'. Use: h m l")
                elif cmd == "note":
                    notes = arg
                else:
                    print("  Unrecognised command. Use Enter, o, q, s, or x.")

        if skipped:
            continue

        corrected = confirm_label(entry, outcome_label=outcome, signal_quality=quality, notes=notes)
        confirmed_examples.append(corrected.model_dump_json())
        # entry is mutated in-place by confirm_label (reviewed=True)

    _flush(out_path, confirmed_examples)
    rewrite_queue(entries, queue_path)

    reviewed_now = len(confirmed_examples)
    remaining = sum(1 for e in entries if not e.reviewed)
    print(f"\nDone. Confirmed {reviewed_now} example(s) → {out_path}")
    print(f"Remaining unreviewed: {remaining}")


def _flush(out_path: Path, lines: list[str]) -> None:
    if not lines:
        return
    with out_path.open("a", encoding="utf-8") as fh:
        for line in lines:
            fh.write(line + "\n")
    lines.clear()


if __name__ == "__main__":
    main()
