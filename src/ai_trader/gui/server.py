from __future__ import annotations

# ruff: noqa: E501
import json
import os
import subprocess
import webbrowser
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from ai_trader.config import get_settings
from ai_trader.gui.actions import action_specs, build_command, run_action
from ai_trader.gui.frontend import FRONTEND_HTML
from ai_trader.ingestion import detect_hardware
from ai_trader.training.review_queue import (
    DEFAULT_HUMAN_OUT_PATH,
    DEFAULT_QUEUE_PATH,
    confirm_label,
    iter_queue,
    rewrite_queue,
)


def run_gui(
    *,
    host: str = "127.0.0.1",
    port: int = 8787,
    open_browser: bool = True,
    working_dir: Path | None = None,
) -> None:
    """Run the local AI Trader browser GUI."""

    working_dir = working_dir or Path.cwd()
    handler = _make_handler(working_dir)
    server = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{port}"
    print(f"AI Trader GUI running at {url}")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("AI Trader GUI stopped.")
    finally:
        server.server_close()


def _make_handler(working_dir: Path) -> type[BaseHTTPRequestHandler]:
    class GUIRequestHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/":
                self._send_html(FRONTEND_HTML)
                return
            if path == "/api/actions":
                self._send_json({"actions": action_specs()})
                return
            if path == "/api/overview":
                self._send_json(_overview_payload(working_dir))
                return
            if path == "/api/artifact":
                qs = parse_qs(urlparse(self.path).query)
                path_value = qs.get("path", [""])[0]
                try:
                    self._send_json(_read_artifact(working_dir, path_value))
                except Exception as exc:
                    self._send_json({"success": False, "error": str(exc)}, status=400)
                return
            if path == "/api/review":
                qs = parse_qs(urlparse(self.path).query)
                limit = int(qs.get("limit", ["20"])[0])
                offset = int(qs.get("offset", ["0"])[0])
                queue_path = working_dir / DEFAULT_QUEUE_PATH
                all_pending = [e for e in iter_queue(queue_path) if not e.reviewed]
                page = all_pending[offset : offset + limit]
                entries = json.loads("[" + ",".join(e.model_dump_json() for e in page) + "]") if page else []
                self._send_json(
                    {
                        "pending": len(all_pending),
                        "total": len(all_pending),
                        "offset": offset,
                        "limit": limit,
                        "entries": entries,
                    }
                )
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            path = urlparse(self.path).path

            if path == "/api/stream":
                try:
                    payload = self._read_json()
                    action_id = str(payload.get("action", ""))
                    inputs = payload.get("inputs", {})
                    if not isinstance(inputs, dict):
                        raise ValueError("inputs must be an object")
                    command = build_command(action_id, inputs)
                except Exception as exc:
                    self._send_json({"success": False, "error": str(exc)}, status=400)
                    return

                self.send_response(HTTPStatus.OK)
                self.send_header("content-type", "text/event-stream; charset=utf-8")
                self.send_header("cache-control", "no-cache")
                self.send_header("x-accel-buffering", "no")
                self.end_headers()

                def _sse(event: str, data: str) -> None:
                    for line in data.splitlines():
                        msg = f"event: {event}\ndata: {line}\n\n"
                        self.wfile.write(msg.encode("utf-8"))
                    self.wfile.flush()

                try:
                    proc = subprocess.Popen(
                        command,
                        cwd=working_dir,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        bufsize=1,
                    )
                    _sse("start", json.dumps({"command": list(command)}))
                    for line in proc.stdout:  # type: ignore[union-attr]
                        _sse("line", line.rstrip("\n"))
                    proc.wait()
                    _sse("done", json.dumps({"returncode": proc.returncode}))
                except Exception as exc:
                    _sse("error", str(exc))
                return

            if path == "/api/review/decide":
                try:
                    payload = self._read_json()
                    queue_id = str(payload.get("queue_id", ""))
                    action = str(payload.get("action", "accept"))
                    outcome_override = payload.get("outcome_override")
                    quality_override = payload.get("quality_override")
                    queue_path = working_dir / DEFAULT_QUEUE_PATH
                    human_out_path = working_dir / DEFAULT_HUMAN_OUT_PATH
                    all_entries = list(iter_queue(queue_path))
                    target = next((e for e in all_entries if e.queue_id == queue_id), None)
                    if target is None:
                        self._send_json({"success": False, "error": "entry not found"}, status=404)
                        return
                    if action == "skip":
                        self._send_json({"success": True, "action": "skipped"})
                        return
                    final_outcome = outcome_override or target.auto_outcome_label
                    final_quality = quality_override or target.auto_signal_quality
                    corrected = confirm_label(target, outcome_label=final_outcome, signal_quality=final_quality)
                    human_out_path.parent.mkdir(parents=True, exist_ok=True)
                    with human_out_path.open("a", encoding="utf-8") as fh:
                        fh.write(corrected.model_dump_json() + "\n")
                    for e in all_entries:
                        if e.queue_id == queue_id:
                            e.reviewed = True
                    rewrite_queue(all_entries, queue_path)
                    self._send_json({"success": True, "action": "confirmed", "queue_id": queue_id})
                except Exception as exc:
                    self._send_json({"success": False, "error": str(exc)}, status=400)
                return

            if path != "/api/run":
                self.send_error(HTTPStatus.NOT_FOUND)
                return

            try:
                payload = self._read_json()
                action_id = str(payload.get("action", ""))
                inputs = payload.get("inputs", {})
                if not isinstance(inputs, dict):
                    raise ValueError("inputs must be an object")
                result = run_action(action_id, inputs, cwd=working_dir)
                self._send_json(result.to_dict())
            except Exception as exc:
                self._send_json({"success": False, "error": str(exc)}, status=400)

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("content-length", "0"))
            body = self.rfile.read(length).decode("utf-8")
            if not body:
                return {}
            payload = json.loads(body)
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            return payload

        def _send_html(self, html: str) -> None:
            encoded = html.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("content-type", "text/html; charset=utf-8")
            self.send_header("content-length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _send_json(self, payload: dict[str, Any], *, status: int = 200) -> None:
            encoded = json.dumps(payload, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", "application/json; charset=utf-8")
            self.send_header("content-length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    return GUIRequestHandler


def _overview_payload(working_dir: Path) -> dict[str, Any]:
    settings = get_settings()
    hardware = detect_hardware(probe_network=False)
    queue_path = working_dir / DEFAULT_QUEUE_PATH
    human_out_path = working_dir / DEFAULT_HUMAN_OUT_PATH
    queue_entries = list(iter_queue(queue_path))
    pending_reviews = sum(1 for entry in queue_entries if not entry.reviewed)
    default_examples = working_dir / "logs" / "training_examples.jsonl"
    latest_profile = working_dir / settings.ingestion_profile_path
    return {
        "success": True,
        "generated_at": datetime.now(UTC).isoformat(),
        "working_dir": str(working_dir),
        "providers": settings.provider_status(),
        "hardware": {
            "physical_cores": hardware.physical_cores,
            "logical_cores": hardware.logical_cores,
            "ram_gb": round(hardware.ram_gb, 2),
            "ram_available_gb": round(hardware.ram_available_gb, 2),
            "network_mbps": hardware.network_mbps,
            "network_tier": hardware.network_tier,
            "source_workers": settings.ingestion_source_workers,
            "price_workers": settings.ingestion_price_workers,
            "ticker_workers": settings.ingestion_ticker_workers,
            "http_connections": settings.ingestion_http_connections,
            "write_buffer": settings.ingestion_write_buffer,
        },
        "paths": {
            "training_examples": _display_path(default_examples, working_dir),
            "review_queue": _display_path(queue_path, working_dir),
            "human_labels": _display_path(human_out_path, working_dir),
            "ingestion_profile": _display_path(latest_profile, working_dir),
            "cache_dir": str(settings.ingestion_cache_dir),
        },
        "metrics": {
            "actions": len(action_specs()),
            "review_pending": pending_reviews,
            "review_total": len(queue_entries),
            "training_examples": _count_jsonl(default_examples),
            "human_labels": _count_jsonl(human_out_path),
        },
        "artifacts": _recent_artifacts(working_dir),
    }


def _recent_artifacts(working_dir: Path, *, limit: int = 18) -> list[dict[str, Any]]:
    roots = [working_dir / "logs", working_dir / "data" / "models", working_dir / "data" / "cache"]
    suffixes = {".json", ".jsonl", ".log", ".txt"}
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in suffixes:
                files.append(path)
    files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return [_artifact_summary(path, working_dir) for path in files[:limit]]


def _artifact_summary(path: Path, working_dir: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": _display_path(path, working_dir),
        "name": path.name,
        "size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
        "kind": path.suffix.lower().lstrip(".") or "file",
    }


def _read_artifact(working_dir: Path, path_value: str, *, max_bytes: int = 100_000) -> dict[str, Any]:
    if not path_value.strip():
        raise ValueError("path is required")
    path = _resolve_within(working_dir, path_value)
    if not path.exists() or not path.is_file():
        raise ValueError("artifact not found")
    data = path.read_bytes()[:max_bytes]
    truncated = path.stat().st_size > max_bytes
    return {
        "success": True,
        "artifact": _artifact_summary(path, working_dir),
        "content": data.decode("utf-8", errors="replace"),
        "truncated": truncated,
    }


def _resolve_within(root: Path, path_value: str) -> Path:
    root = root.resolve()
    candidate = Path(path_value)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("path must stay within the repository") from exc
    return candidate


def _display_path(path: Path, working_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(working_dir.resolve())).replace(os.sep, "/")
    except (OSError, ValueError):
        return str(path)


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return 0


_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI Trader Console</title>
  <style>
    :root {
      --bg: #f6f7f8;
      --panel: #ffffff;
      --ink: #172026;
      --muted: #62707b;
      --line: #d8dee4;
      --accent: #0f766e;
      --accent-dark: #115e59;
      --danger: #a43f3f;
      --code: #101418;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family:
        Inter,
        ui-sans-serif,
        system-ui,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;
      letter-spacing: 0;
    }
    .shell {
      display: grid;
      grid-template-columns: 280px minmax(0, 1fr);
      min-height: 100vh;
    }
    aside {
      border-right: 1px solid var(--line);
      background: #eef1f3;
      padding: 18px;
    }
    main {
      display: grid;
      grid-template-rows: auto minmax(260px, 1fr);
      min-width: 0;
      position: relative;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 18px 24px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }
    h1, h2 {
      margin: 0;
      font-size: 18px;
      font-weight: 700;
    }
    .status {
      display: flex;
      align-items: center;
      gap: 10px;
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }
    .dot {
      width: 9px;
      height: 9px;
      border-radius: 999px;
      background: var(--accent);
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 18px;
    }
    .mark {
      display: grid;
      place-items: center;
      width: 34px;
      height: 34px;
      border-radius: 7px;
      color: white;
      background: var(--accent);
      font-weight: 800;
    }
    .group {
      margin: 18px 0 8px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }
    .action-list {
      display: grid;
      gap: 6px;
    }
    .action {
      width: 100%;
      border: 1px solid transparent;
      background: transparent;
      color: var(--ink);
      cursor: pointer;
      border-radius: 7px;
      padding: 9px 10px;
      text-align: left;
      font: inherit;
    }
    .action:hover,
    .action.active {
      border-color: var(--line);
      background: var(--panel);
    }
    .workspace {
      display: grid;
      grid-template-columns: minmax(320px, 520px) minmax(0, 1fr);
      min-height: 0;
    }
    .form-pane {
      padding: 22px 24px;
      border-right: 1px solid var(--line);
      overflow: auto;
    }
    .output-pane {
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      min-width: 0;
      background: var(--code);
      color: #e8edf2;
    }
    form {
      display: grid;
      gap: 14px;
      margin-top: 20px;
    }
    label {
      display: grid;
      gap: 6px;
      font-size: 13px;
      font-weight: 650;
    }
    input,
    textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: var(--panel);
      color: var(--ink);
      padding: 10px 11px;
      font: inherit;
      font-size: 14px;
    }
    textarea {
      min-height: 122px;
      resize: vertical;
      line-height: 1.4;
    }
    .check {
      display: flex;
      align-items: center;
      gap: 10px;
      font-weight: 650;
    }
    .check input {
      width: 18px;
      height: 18px;
    }
    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 6px;
    }
    button.run,
    button.clear {
      border: 0;
      border-radius: 7px;
      padding: 10px 14px;
      cursor: pointer;
      font: inherit;
      font-weight: 700;
    }
    button.run {
      background: var(--accent);
      color: white;
    }
    button.run:hover { background: var(--accent-dark); }
    button.run:disabled {
      cursor: wait;
      background: #8aa9a5;
    }
    button.clear {
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--ink);
    }
    .output-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 14px 18px;
      border-bottom: 1px solid #2b343b;
      color: #cbd5df;
      font-size: 13px;
    }
    pre {
      margin: 0;
      padding: 18px;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: "Cascadia Mono", "SFMono-Regular", Consolas, monospace;
      font-size: 13px;
      line-height: 1.45;
    }
  .progress-bar-wrap {
      padding: 0 18px 14px;
      display: none;
    }
    .progress-bar-wrap.visible { display: block; }
    .progress-bar {
      width: 100%;
      height: 6px;
      background: #2b343b;
      border-radius: 99px;
      overflow: hidden;
    }
    .progress-fill {
      height: 100%;
      width: 100%;
      background: var(--accent);
      border-radius: 99px;
      transform-origin: left;
      animation: pulse 1.4s ease-in-out infinite;
    }
    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.4; }
    }
    .line-count {
      font-size: 12px;
      color: #8a9aaa;
      margin-top: 5px;
    }
    .error { color: #ffb4b4; }
    .ok { color: #a8e6cf; }
    /* ── Review panel ── */
    .review-panel {
      display: none;
      flex-direction: column;
      gap: 0;
      position: absolute;
      inset: 0;
      z-index: 10;
      background: var(--bg);
      overflow: hidden;
    }
    .review-panel.visible { display: flex; }
    .review-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 14px 20px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      flex-shrink: 0;
    }
    .review-header h3 { margin: 0; font-size: 15px; }
    .review-badge {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 24px;
      height: 24px;
      padding: 0 7px;
      border-radius: 999px;
      background: var(--accent);
      color: white;
      font-size: 12px;
      font-weight: 700;
    }
    .review-body {
      overflow-y: auto;
      flex: 1;
      padding: 16px 20px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }
    .review-card {
      border: 1px solid var(--line);
      border-radius: 10px;
      background: var(--panel);
      padding: 16px 18px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .review-card-head {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 8px;
    }
    .review-ticker {
      font-size: 17px;
      font-weight: 800;
      letter-spacing: 0.02em;
    }
    .review-pnl {
      font-size: 13px;
      font-weight: 700;
      padding: 2px 8px;
      border-radius: 6px;
    }
    .review-pnl.pos { background: #d1fae5; color: #065f46; }
    .review-pnl.neg { background: #fee2e2; color: #991b1b; }
    .review-pnl.neu { background: #f1f5f9; color: #475569; }
    .review-meta {
      font-size: 12px;
      color: var(--muted);
      display: flex;
      flex-wrap: wrap;
      gap: 6px 14px;
    }
    .review-signals {
      font-size: 12px;
      color: var(--ink);
      background: var(--bg);
      border-radius: 6px;
      padding: 8px 10px;
      display: flex;
      flex-direction: column;
      gap: 3px;
      max-height: 120px;
      overflow-y: auto;
    }
    .review-labels {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    .review-label-box {
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 8px 10px;
      font-size: 12px;
    }
    .review-label-box strong { display: block; font-size: 11px; color: var(--muted); margin-bottom: 2px; }
    .review-reasons {
      font-size: 11px;
      color: var(--muted);
      display: flex;
      flex-direction: column;
      gap: 2px;
    }
    .review-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 2px;
    }
    .rv-btn {
      border: 0;
      border-radius: 7px;
      padding: 10px 16px;
      cursor: pointer;
      font: inherit;
      font-size: 13px;
      font-weight: 700;
      flex: 1;
      min-width: 80px;
    }
    .rv-btn.accept  { background: var(--accent); color: white; }
    .rv-btn.accept:hover { background: var(--accent-dark); }
    .rv-btn.skip    { background: var(--bg); color: var(--muted); border: 1px solid var(--line); }
    .rv-btn.skip:hover { background: var(--line); }
    .review-override {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .review-override summary {
      cursor: pointer;
      font-size: 12px;
      color: var(--muted);
      user-select: none;
      padding: 2px 0;
    }
    .override-row {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 6px;
    }
    .ov-chip {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--bg);
      padding: 5px 11px;
      font-size: 12px;
      cursor: pointer;
      font: inherit;
    }
    .ov-chip:hover { background: var(--line); }
    .ov-chip.selected { background: var(--accent); color: white; border-color: var(--accent); }
    .review-empty {
      text-align: center;
      color: var(--muted);
      padding: 40px 20px;
      font-size: 14px;
    }
    .review-empty .big { font-size: 36px; margin-bottom: 10px; }
    @media (max-width: 900px) {
      .shell { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
      .workspace { grid-template-columns: 1fr; }
      .form-pane { border-right: 0; border-bottom: 1px solid var(--line); }
      header { align-items: flex-start; flex-direction: column; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <aside>
      <div class="brand">
        <div class="mark">AI</div>
        <h1>Trader Console</h1>
      </div>
      <nav id="nav"></nav>
    </aside>
    <main>
      <header>
        <h2 id="title">Status</h2>
        <div class="status"><span class="dot"></span><span id="state">Ready</span></div>
      </header>
      <section class="workspace">
        <div class="form-pane">
          <form id="form"></form>
        </div>
        <div class="output-pane">
          <div class="output-head">
            <span id="command">No command run yet</span>
            <span id="result"></span>
          </div>
          <div class="progress-bar-wrap" id="progress-wrap">
            <div class="progress-bar"><div class="progress-fill"></div></div>
            <div class="line-count" id="line-count"></div>
          </div>
          <pre id="output"></pre>
        </div>
      </section>
      <!-- ── Label Review Panel (overlays workspace when open) ── -->
      <div class="review-panel" id="review-panel">
        <div class="review-header">
          <h3>📋 Label Review — confirm or override auto-labels</h3>
          <div style="display:flex;gap:10px;align-items:center">
            <span class="review-badge" id="review-badge">0</span>
            <button class="rv-btn skip" style="flex:unset;min-width:unset;padding:6px 12px" onclick="closeReviewPanel()">✕ Close</button>
          </div>
        </div>
        <div class="review-body" id="review-body">
          <div class="review-empty"><div class="big">✅</div>No items pending review.</div>
        </div>
      </div>
    </main>
  </div>
  <script>
    const nav = document.getElementById("nav");
    const form = document.getElementById("form");
    const title = document.getElementById("title");
    const state = document.getElementById("state");
    const output = document.getElementById("output");
    const command = document.getElementById("command");
    const result = document.getElementById("result");
    const progressWrap = document.getElementById("progress-wrap");
    const lineCount = document.getElementById("line-count");
    const reviewPanel = document.getElementById("review-panel");
    const reviewBody = document.getElementById("review-body");
    const reviewBadge = document.getElementById("review-badge");
    let actions = [];
    let active = null;

    // ── Review panel ──────────────────────────────────────────────────────
    const OUTCOME_LABELS = ["strong_win", "win", "neutral", "loss", "strong_loss"];
    const QUALITY_LABELS = ["high", "medium", "low"];
    const PAGE_SIZE = 20;
    let reviewTotal = 0;
    let reviewOffset = 0;
    let reviewLoading = false;

    async function openReviewPanel() {
      reviewPanel.classList.add("visible");
      reviewBody.innerHTML = "";
      reviewTotal = 0;
      reviewOffset = 0;
      await loadReviewPage();
    }
    function closeReviewPanel() {
      reviewPanel.classList.remove("visible");
    }

    async function loadReviewQueue() {
      reviewBody.innerHTML = "";
      reviewTotal = 0;
      reviewOffset = 0;
      await loadReviewPage();
    }

    async function loadReviewPage() {
      if (reviewLoading) return;
      reviewLoading = true;
      const old = document.getElementById("review-load-more");
      if (old) old.remove();
      const loader = document.createElement("div");
      loader.className = "review-empty";
      loader.id = "review-loader";
      loader.innerHTML = '<div class="big">⏳</div>Loading…';
      reviewBody.appendChild(loader);
      try {
        const res = await fetch(`/api/review?limit=${PAGE_SIZE}&offset=${reviewOffset}`);
        const data = await res.json();
        reviewTotal = data.pending || 0;
        updateBadge(reviewTotal);
        loader.remove();
        if (data.entries && data.entries.length > 0) {
          for (const entry of data.entries) {
            reviewBody.appendChild(buildReviewCard(entry));
          }
          reviewOffset += data.entries.length;
        }
        if (reviewOffset === 0 && reviewTotal === 0) {
          reviewBody.innerHTML = '<div class="review-empty"><div class="big">✅</div>Queue is empty — nothing to review!</div>';
        } else if (reviewOffset < reviewTotal) {
          const remaining = reviewTotal - reviewOffset;
          const more = document.createElement("button");
          more.id = "review-load-more";
          more.className = "rv-btn skip";
          more.style.cssText = "width:100%;margin-top:4px";
          more.textContent = `Load next ${Math.min(PAGE_SIZE, remaining)} of ${remaining} remaining`;
          more.onclick = loadReviewPage;
          reviewBody.appendChild(more);
        }
      } catch (e) {
        loader.remove();
        const err = document.createElement("div");
        err.className = "review-empty";
        err.innerHTML = '<div class="big">⚠️</div>Failed to load: ' + e.message;
        reviewBody.appendChild(err);
      } finally {
        reviewLoading = false;
      }
    }

    function updateBadge(n) {
      reviewBadge.textContent = n;
      const nb = document.getElementById("nav-badge");
      if (nb) {
        nb.textContent = n;
        nb.style.display = n > 0 ? "inline-flex" : "none";
      }
    }

    function buildReviewCard(entry) {
      const ex = entry.example;
      const bundle = ex.signal_bundle;
      const plan = ex.trade_plan;
      const pnl = ex.pnl_pct;
      const pnlClass = pnl > 0.01 ? "pos" : pnl < -0.01 ? "neg" : "neu";
      const pnlStr = (pnl >= 0 ? "+" : "") + (pnl * 100).toFixed(1) + "%";

      const card = document.createElement("div");
      card.className = "review-card";
      card.dataset.queueId = entry.queue_id;

      // State for overrides
      let outcomeOverride = null;
      let qualityOverride = null;

      card.innerHTML = `
        <div class="review-card-head">
          <span class="review-ticker">${bundle.ticker}</span>
          <span class="review-pnl ${pnlClass}">${pnlStr}</span>
        </div>
        <div class="review-meta">
          <span>📅 ${bundle.as_of || "—"}</span>
          <span>🎯 direction: <strong>${plan.direction}</strong></span>
          <span>💡 conviction: <strong>${(plan.conviction || 0).toFixed(2)}</strong></span>
          <span>📊 signals: <strong>${(bundle.signals || []).length}</strong></span>
        </div>
        <div class="review-signals" id="sigs-${entry.queue_id}"></div>
        <div class="review-labels">
          <div class="review-label-box">
            <strong>AUTO OUTCOME</strong>
            <span id="ol-${entry.queue_id}">${entry.auto_outcome_label}</span>
          </div>
          <div class="review-label-box">
            <strong>AUTO QUALITY</strong>
            <span id="ql-${entry.queue_id}">${entry.auto_signal_quality}</span>
            <span style="font-size:11px;color:var(--muted)"> (conf ${(entry.auto_label_confidence * 100).toFixed(0)}%)</span>
          </div>
        </div>
        <details class="review-override">
          <summary>✏️ Override labels (optional)</summary>
          <div style="margin-top:8px">
            <div style="font-size:11px;font-weight:700;color:var(--muted);margin-bottom:4px">OUTCOME</div>
            <div class="override-row" id="ov-out-${entry.queue_id}">
              ${OUTCOME_LABELS.map(l => `<button class="ov-chip" data-field="outcome" data-value="${l}" onclick="selectOverride(this,'${entry.queue_id}')">${l}</button>`).join("")}
            </div>
            <div style="font-size:11px;font-weight:700;color:var(--muted);margin:8px 0 4px">SIGNAL QUALITY</div>
            <div class="override-row" id="ov-q-${entry.queue_id}">
              ${QUALITY_LABELS.map(l => `<button class="ov-chip" data-field="quality" data-value="${l}" onclick="selectOverride(this,'${entry.queue_id}')">${l}</button>`).join("")}
            </div>
          </div>
        </details>
        <div class="review-reasons" id="reasons-${entry.queue_id}"></div>
        <div class="review-actions">
          <button class="rv-btn accept" onclick="decide('${entry.queue_id}','accept')">✅ Accept</button>
          <button class="rv-btn skip"   onclick="decide('${entry.queue_id}','skip')">⏭ Skip</button>
        </div>
      `;

      // Fill signals list
      const sigEl = card.querySelector(`#sigs-${entry.queue_id}`);
      for (const s of (bundle.signals || []).slice(0, 6)) {
        const row = document.createElement("div");
        row.textContent = `${s.name}  •  ${s.direction}  •  str ${(s.strength||0).toFixed(2)}  conf ${(s.confidence||0).toFixed(2)}`;
        sigEl.appendChild(row);
      }
      if ((bundle.signals || []).length > 6) {
        const more = document.createElement("div");
        more.style.color = "var(--muted)";
        more.textContent = `… and ${bundle.signals.length - 6} more`;
        sigEl.appendChild(more);
      }

      // Fill reasons
      const reasonsEl = card.querySelector(`#reasons-${entry.queue_id}`);
      for (const r of (entry.review_reasons || [])) {
        const row = document.createElement("div");
        row.textContent = "⚠ " + r;
        reasonsEl.appendChild(row);
      }

      return card;
    }

    function selectOverride(btn, queueId) {
      const field = btn.dataset.field;
      const val = btn.dataset.value;
      const container = document.getElementById(field === "outcome" ? `ov-out-${queueId}` : `ov-q-${queueId}`);
      for (const c of container.querySelectorAll(".ov-chip")) c.classList.remove("selected");
      btn.classList.add("selected");
      // Store on card
      const card = document.querySelector(`[data-queue-id="${queueId}"]`);
      if (field === "outcome") card._outcomeOverride = val;
      else card._qualityOverride = val;
    }

    async function decide(queueId, action) {
      const card = document.querySelector(`[data-queue-id="${queueId}"]`);
      const body = {
        queue_id: queueId,
        action,
        outcome_override: card ? (card._outcomeOverride || null) : null,
        quality_override: card ? (card._qualityOverride || null) : null,
      };
      try {
        const res = await fetch("/api/review/decide", {
          method: "POST",
          headers: {"content-type": "application/json"},
          body: JSON.stringify(body),
        });
        const data = await res.json();
        if (data.success) {
          // Update local count
          reviewTotal = Math.max(0, reviewTotal - 1);
          reviewOffset = Math.max(0, reviewOffset - 1);
          card.style.opacity = "0.4";
          card.style.pointerEvents = "none";
          setTimeout(() => {
            card.remove();
            updateBadge(reviewTotal);
            const remaining = reviewBody.querySelectorAll(".review-card").length;
            if (remaining === 0 && reviewTotal === 0) {
              reviewBody.innerHTML = '<div class="review-empty"><div class="big">\u2705</div>All done! Queue is clear.</div>';
            } else if (remaining === 0) {
              loadReviewPage();
            }
          }, 300);
        } else {
          alert("Error: " + (data.error || "unknown"));
        }
      } catch (e) {
        alert("Request failed: " + e.message);
      }
    }

    // Poll badge count every 60 s
    async function pollBadge() {
      try {
        const res = await fetch("/api/review?limit=0&offset=0");
        const data = await res.json();
        updateBadge(data.pending || 0);
      } catch (_) {}
    }
    pollBadge();
    setInterval(pollBadge, 60000);
    // ── End review panel ──────────────────────────────────────────────────
    let abortController = null;

    async function loadActions() {
      const response = await fetch("/api/actions");
      const payload = await response.json();
      actions = payload.actions;
      renderNav();
      selectAction(actions[0].id);
    }

    function renderNav() {
      nav.innerHTML = "";

      // ── Always-visible Label Review panel button ──
      const rvGroup = document.createElement("div");
      rvGroup.className = "group";
      rvGroup.textContent = "Label Review";
      const rvList = document.createElement("div");
      rvList.className = "action-list";
      const rvBtn = document.createElement("button");
      rvBtn.className = "action";
      rvBtn.type = "button";
      rvBtn.style.cssText = "font-weight:700;display:flex;align-items:center;justify-content:space-between;gap:8px";
      rvBtn.innerHTML = '📋 Label Items <span class="review-badge" id="nav-badge" style="display:none">0</span>';
      rvBtn.onclick = openReviewPanel;
      rvList.appendChild(rvBtn);
      nav.appendChild(rvGroup);
      nav.appendChild(rvList);

      // ── Action groups (skip Label Review — handled above) ──
      const groups = {};
      for (const action of actions) {
        if (action.group === "Label Review") continue;
        groups[action.group] = groups[action.group] || [];
        groups[action.group].push(action);
      }
      for (const [group, items] of Object.entries(groups)) {
        const groupNode = document.createElement("div");
        groupNode.className = "group";
        groupNode.textContent = group;
        nav.appendChild(groupNode);
        const list = document.createElement("div");
        list.className = "action-list";
        for (const action of items) {
          const button = document.createElement("button");
          button.className = "action";
          button.type = "button";
          button.textContent = action.label;
          button.dataset.id = action.id;
          button.onclick = () => selectAction(action.id);
          list.appendChild(button);
        }
        nav.appendChild(list);
      }

      // Sync badge after nav rebuild
      pollBadge();
    }

    function selectAction(id) {
      active = actions.find((action) => action.id === id);
      title.textContent = active.label;
      for (const node of document.querySelectorAll(".action")) {
        node.classList.toggle("active", node.dataset.id === id);
      }
      renderForm();
    }

    function renderForm() {
      form.innerHTML = "";
      for (const field of active.fields) {
        const label = document.createElement("label");
        if (field.type === "checkbox") {
          label.className = "check";
          const input = document.createElement("input");
          input.type = "checkbox";
          input.name = field.name;
          input.checked = Boolean(field.default);
          label.appendChild(input);
          label.appendChild(document.createTextNode(field.label));
          form.appendChild(label);
          continue;
        }
        label.textContent = field.label;
        const input = field.type === "textarea"
          ? document.createElement("textarea")
          : document.createElement("input");
        if (field.type !== "textarea") {
          input.type = field.type === "number" ? "number" : "text";
          if (field.min !== null) input.min = field.min;
          if (field.max !== null) input.max = field.max;
        }
        input.name = field.name;
        input.required = field.required;
        input.placeholder = field.placeholder || "";
        input.value = field.default === null ? "" : field.default;
        label.appendChild(input);
        form.appendChild(label);
      }
      const controls = document.createElement("div");
      controls.className = "controls";
      const run = document.createElement("button");
      run.className = "run";
      run.type = "submit";
      run.textContent = active.streaming ? "Run (Live)" : "Run";
      const clear = document.createElement("button");
      clear.className = "clear";
      clear.type = "button";
      clear.textContent = "Clear Output";
      clear.onclick = () => {
        if (abortController) { abortController.abort(); abortController = null; }
        output.textContent = "";
        command.textContent = "No command run yet";
        result.textContent = "";
        result.className = "";
        progressWrap.classList.remove("visible");
        lineCount.textContent = "";
        state.textContent = "Ready";
        run.disabled = false;
      };
      controls.appendChild(run);
      controls.appendChild(clear);
      form.appendChild(controls);
    }

    function collectInputs() {
      const inputs = {};
      for (const field of active.fields) {
        const node = form.elements[field.name];
        if (!node) continue;
        inputs[field.name] = field.type === "checkbox" ? node.checked : node.value;
      }
      return inputs;
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const runButton = form.querySelector("button.run");
      const inputs = collectInputs();
      state.textContent = "Running";
      runButton.disabled = true;
      result.textContent = "";
      output.textContent = "";
      progressWrap.classList.remove("visible");
      lineCount.textContent = "";

      if (active.streaming) {
        await runStreaming(inputs, runButton);
      } else {
        await runBlocking(inputs, runButton);
      }
    });

    async function runStreaming(inputs, runButton) {
      abortController = new AbortController();
      progressWrap.classList.add("visible");
      let lines = 0;
      try {
        const response = await fetch("/api/stream", {
          method: "POST",
          headers: {"content-type": "application/json"},
          body: JSON.stringify({action: active.id, inputs}),
          signal: abortController.signal,
        });
        if (!response.ok) {
          const err = await response.json().catch(() => ({error: response.statusText}));
          throw new Error(err.error || "Stream request failed");
        }
        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buf = "";
        while (true) {
          const {done, value} = await reader.read();
          if (done) break;
          buf += decoder.decode(value, {stream: true});
          const parts = buf.split("\n\n");
          buf = parts.pop();
          for (const block of parts) {
            let eventType = "line", data = "";
            for (const rawLine of block.split("\n")) {
              if (rawLine.startsWith("event: ")) eventType = rawLine.slice(7).trim();
              else if (rawLine.startsWith("data: ")) data = rawLine.slice(6);
            }
            if (eventType === "start") {
              const payload = JSON.parse(data);
              command.textContent = payload.command.join(" ");
            } else if (eventType === "line") {
              output.textContent += data + "\n";
              output.scrollTop = output.scrollHeight;
              lines++;
              lineCount.textContent = lines + " lines";
            } else if (eventType === "done") {
              const payload = JSON.parse(data);
              const ok = payload.returncode === 0;
              result.textContent = ok ? "ok" : "exit " + payload.returncode;
              result.className = ok ? "ok" : "error";
            } else if (eventType === "error") {
              result.textContent = "error";
              result.className = "error";
              output.textContent += "\n[error] " + data + "\n";
            }
          }
        }
      } catch (err) {
        if (err.name !== "AbortError") {
          result.textContent = "error";
          result.className = "error";
          output.textContent += "\n[error] " + err.message + "\n";
        }
      } finally {
        progressWrap.classList.remove("visible");
        state.textContent = "Ready";
        runButton.disabled = false;
        abortController = null;
      }
    }

    async function runBlocking(inputs, runButton) {
      try {
        const response = await fetch("/api/run", {
          method: "POST",
          headers: {"content-type": "application/json"},
          body: JSON.stringify({action: active.id, inputs})
        });
        const payload = await response.json();
        if (!response.ok || payload.success === false) {
          throw new Error(payload.error || payload.stderr || "Command failed");
        }
        command.textContent = payload.command.join(" ");
        result.textContent = payload.success ? "ok" : "failed";
        result.className = payload.success ? "ok" : "error";
        output.textContent = [
          payload.stdout || "",
          payload.stderr ? "\n[stderr]\n" + payload.stderr : "",
          payload.log_file ? "\n[log]\n" + payload.log_file : ""
        ].join("");
      } catch (err) {
        result.textContent = "error";
        result.className = "error";
        output.textContent = err.message;
      } finally {
        state.textContent = "Ready";
        runButton.disabled = false;
      }
    }

    loadActions();
  </script>
</body>
</html>
"""
