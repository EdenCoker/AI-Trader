from __future__ import annotations

import json
import subprocess
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ai_trader.gui.actions import action_specs, build_command, run_action


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
                self._send_html(_HTML)
                return
            if path == "/api/actions":
                self._send_json({"actions": action_specs()})
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
    let actions = [];
    let active = null;
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
      const groups = {};
      for (const action of actions) {
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
