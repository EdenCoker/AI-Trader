# ruff: noqa: E501
from __future__ import annotations

FRONTEND_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI Trader</title>
  <style>
    :root {
      --bg: #f5f6f8;
      --surface: #ffffff;
      --surface-2: #eef2f5;
      --ink: #18232c;
      --muted: #64727f;
      --line: #d9e0e6;
      --accent: #0f766e;
      --accent-2: #2563eb;
      --warn: #b45309;
      --danger: #b42318;
      --success: #047857;
      --terminal: #0f1419;
      --terminal-line: #26313a;
      --terminal-text: #e7edf3;
      --shadow: 0 10px 24px rgba(24, 35, 44, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    button, input, textarea, select {
      font: inherit;
      letter-spacing: 0;
    }
    button { cursor: pointer; }
    .app {
      display: grid;
      grid-template-columns: 276px minmax(0, 1fr);
      min-height: 100vh;
    }
    .sidebar {
      display: grid;
      grid-template-rows: auto auto minmax(0, 1fr);
      gap: 14px;
      padding: 18px;
      border-right: 1px solid var(--line);
      background: #edf1f4;
      min-height: 0;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 10px;
      min-height: 38px;
    }
    .brand-mark {
      display: grid;
      place-items: center;
      width: 36px;
      height: 36px;
      border-radius: 7px;
      background: var(--ink);
      color: #fff;
      font-weight: 800;
    }
    .brand-title {
      margin: 0;
      font-size: 17px;
      line-height: 1.1;
    }
    .brand-subtitle {
      margin-top: 2px;
      color: var(--muted);
      font-size: 12px;
    }
    .search {
      width: 100%;
      height: 38px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: var(--surface);
      color: var(--ink);
      padding: 0 11px;
    }
    .nav-scroll {
      min-height: 0;
      overflow: auto;
      padding-right: 2px;
    }
    .nav-section {
      margin: 15px 0 7px;
      color: var(--muted);
      font-size: 11px;
      font-weight: 800;
      text-transform: uppercase;
    }
    .nav-list {
      display: grid;
      gap: 5px;
    }
    .nav-button {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      width: 100%;
      min-height: 38px;
      border: 1px solid transparent;
      border-radius: 7px;
      background: transparent;
      color: var(--ink);
      padding: 8px 10px;
      text-align: left;
    }
    .nav-button:hover,
    .nav-button.active {
      background: var(--surface);
      border-color: var(--line);
    }
    .badge {
      min-width: 22px;
      height: 22px;
      display: inline-grid;
      place-items: center;
      padding: 0 7px;
      border-radius: 99px;
      background: var(--accent);
      color: #fff;
      font-size: 12px;
      font-weight: 800;
    }
    .main {
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      min-width: 0;
      min-height: 100vh;
    }
    .topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
      padding: 16px 22px;
      border-bottom: 1px solid var(--line);
      background: var(--surface);
    }
    .title-block {
      min-width: 0;
    }
    .page-title {
      margin: 0;
      font-size: 20px;
      font-weight: 800;
      line-height: 1.2;
    }
    .page-meta {
      margin-top: 3px;
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .top-actions {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }
    .state-pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      min-height: 34px;
      border: 1px solid var(--line);
      border-radius: 99px;
      background: var(--surface-2);
      padding: 0 12px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 700;
    }
    .state-dot {
      width: 9px;
      height: 9px;
      border-radius: 99px;
      background: var(--success);
    }
    .state-dot.running { background: var(--warn); }
    .state-dot.failed { background: var(--danger); }
    .button {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 7px;
      min-height: 36px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: var(--surface);
      color: var(--ink);
      padding: 0 12px;
      font-weight: 800;
    }
    .button:hover { background: var(--surface-2); }
    .button.primary {
      border-color: var(--accent);
      background: var(--accent);
      color: #fff;
    }
    .button.primary:hover { background: #115e59; }
    .button.danger {
      border-color: var(--danger);
      color: var(--danger);
    }
    .button:disabled {
      opacity: 0.58;
      cursor: wait;
    }
    .content {
      min-height: 0;
      overflow: auto;
    }
    .view {
      display: none;
      padding: 22px;
    }
    .view.active { display: block; }
    .view.flush { padding: 0; height: 100%; }
    .stat-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(160px, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }
    .stat-card,
    .panel,
    .artifact-row,
    .review-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
      box-shadow: var(--shadow);
    }
    .stat-card {
      min-height: 98px;
      padding: 14px;
    }
    .stat-label {
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
      text-transform: uppercase;
    }
    .stat-value {
      margin-top: 8px;
      font-size: 26px;
      font-weight: 850;
      line-height: 1;
    }
    .stat-foot {
      margin-top: 8px;
      color: var(--muted);
      font-size: 12px;
      overflow-wrap: anywhere;
    }
    .dashboard-grid {
      display: grid;
      grid-template-columns: minmax(320px, 1.1fr) minmax(280px, 0.9fr);
      gap: 14px;
      align-items: start;
    }
    .panel {
      overflow: hidden;
    }
    .panel-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      min-height: 48px;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
    }
    .panel-title {
      margin: 0;
      font-size: 15px;
      font-weight: 850;
    }
    .panel-body {
      padding: 14px;
    }
    .provider-grid,
    .hardware-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }
    .mini-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #fbfcfd;
      padding: 8px 10px;
      font-size: 13px;
    }
    .mini-row strong {
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .status-text {
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
      white-space: nowrap;
    }
    .status-text.ok { color: var(--success); }
    .status-text.off { color: var(--danger); }
    .workbench {
      display: grid;
      grid-template-columns: minmax(340px, 500px) minmax(0, 1fr);
      height: 100%;
      min-height: 0;
    }
    .form-pane {
      min-height: 0;
      overflow: auto;
      border-right: 1px solid var(--line);
      background: var(--bg);
      padding: 22px;
    }
    .terminal-pane {
      display: grid;
      grid-template-rows: auto auto minmax(0, 1fr);
      min-width: 0;
      min-height: 0;
      background: var(--terminal);
      color: var(--terminal-text);
    }
    .form-grid {
      display: grid;
      gap: 13px;
      margin-top: 16px;
    }
    label.field {
      display: grid;
      gap: 6px;
      font-size: 13px;
      font-weight: 750;
    }
    .field-hint {
      color: var(--muted);
      font-size: 12px;
      font-weight: 500;
    }
    input[type="text"],
    input[type="number"],
    textarea,
    select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: var(--surface);
      color: var(--ink);
      padding: 10px 11px;
      font-size: 14px;
    }
    textarea {
      min-height: 118px;
      resize: vertical;
      line-height: 1.4;
    }
    .check-row {
      display: flex;
      align-items: center;
      gap: 10px;
      min-height: 38px;
      font-size: 13px;
      font-weight: 750;
    }
    .check-row input {
      width: 18px;
      height: 18px;
      accent-color: var(--accent);
    }
    .form-actions {
      display: flex;
      gap: 9px;
      flex-wrap: wrap;
      padding-top: 4px;
    }
    .terminal-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      min-height: 48px;
      padding: 12px 15px;
      border-bottom: 1px solid var(--terminal-line);
      color: #c9d3dc;
      font-size: 13px;
    }
    .command-line {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-family: "Cascadia Mono", "SFMono-Regular", Consolas, monospace;
    }
    .terminal-meta {
      display: flex;
      align-items: center;
      gap: 10px;
      white-space: nowrap;
    }
    .progress-wrap {
      display: none;
      padding: 0 15px 12px;
      border-bottom: 1px solid var(--terminal-line);
    }
    .progress-wrap.visible { display: block; }
    .progress-bar {
      height: 6px;
      overflow: hidden;
      border-radius: 99px;
      background: #28333d;
    }
    .progress-fill {
      width: 100%;
      height: 100%;
      border-radius: 99px;
      background: var(--accent);
      transform-origin: left;
      animation: throb 1.3s ease-in-out infinite;
    }
    @keyframes throb {
      0%, 100% { opacity: 1; transform: scaleX(0.96); }
      50% { opacity: 0.42; transform: scaleX(0.32); }
    }
    .terminal-output {
      margin: 0;
      min-height: 0;
      overflow: auto;
      padding: 16px;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: "Cascadia Mono", "SFMono-Regular", Consolas, monospace;
      font-size: 13px;
      line-height: 1.45;
    }
    .terminal-ok { color: #a7f3d0; }
    .terminal-error { color: #fecaca; }
    .review-layout,
    .artifact-layout {
      display: grid;
      grid-template-columns: minmax(320px, 450px) minmax(0, 1fr);
      gap: 14px;
      align-items: start;
    }
    .review-list,
    .artifact-list {
      display: grid;
      gap: 10px;
    }
    .review-card {
      padding: 14px;
      display: grid;
      gap: 10px;
    }
    .review-card.done {
      opacity: 0.5;
      pointer-events: none;
    }
    .review-head,
    .artifact-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }
    .review-symbol {
      font-size: 17px;
      font-weight: 850;
    }
    .pnl {
      border-radius: 7px;
      padding: 4px 8px;
      font-size: 12px;
      font-weight: 850;
    }
    .pnl.pos { background: #d1fae5; color: #065f46; }
    .pnl.neg { background: #fee2e2; color: #991b1b; }
    .pnl.neu { background: #e5e7eb; color: #475569; }
    .meta-line {
      display: flex;
      flex-wrap: wrap;
      gap: 6px 12px;
      color: var(--muted);
      font-size: 12px;
    }
    .signal-list {
      display: grid;
      gap: 5px;
      max-height: 126px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #fbfcfd;
      padding: 8px;
      font-size: 12px;
    }
    .label-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }
    .label-box {
      min-height: 50px;
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 8px;
      font-size: 13px;
    }
    .label-box span {
      display: block;
      color: var(--muted);
      font-size: 11px;
      font-weight: 850;
      text-transform: uppercase;
    }
    .chip-row {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 7px;
    }
    .chip {
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #fbfcfd;
      color: var(--ink);
      min-height: 30px;
      padding: 0 10px;
      font-size: 12px;
      font-weight: 750;
    }
    .chip.selected {
      border-color: var(--accent);
      background: var(--accent);
      color: #fff;
    }
    .card-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .empty-state {
      border: 1px dashed var(--line);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.55);
      color: var(--muted);
      padding: 28px;
      text-align: center;
      font-weight: 700;
    }
    .artifact-row {
      min-height: 58px;
      padding: 10px 12px;
      box-shadow: none;
    }
    .artifact-row:hover {
      border-color: #b8c4ce;
      background: #fbfcfd;
    }
    .artifact-name {
      font-weight: 850;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .artifact-path {
      color: var(--muted);
      font-size: 12px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      max-width: 100%;
    }
    .artifact-preview {
      margin: 0;
      min-height: 420px;
      max-height: calc(100vh - 230px);
      overflow: auto;
      border-radius: 8px;
      background: var(--terminal);
      color: var(--terminal-text);
      padding: 15px;
      font-family: "Cascadia Mono", "SFMono-Regular", Consolas, monospace;
      font-size: 13px;
      line-height: 1.45;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .split-stack {
      display: grid;
      gap: 14px;
    }
    .toast {
      position: fixed;
      right: 18px;
      bottom: 18px;
      z-index: 50;
      display: none;
      min-width: 240px;
      max-width: 420px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
      box-shadow: var(--shadow);
      padding: 12px 14px;
      font-size: 13px;
      font-weight: 750;
    }
    .toast.visible { display: block; }
    @media (max-width: 1120px) {
      .stat-grid { grid-template-columns: repeat(2, minmax(160px, 1fr)); }
      .dashboard-grid,
      .workbench,
      .review-layout,
      .artifact-layout {
        grid-template-columns: 1fr;
      }
      .form-pane { border-right: 0; border-bottom: 1px solid var(--line); }
      .terminal-pane { min-height: 520px; }
    }
    @media (max-width: 760px) {
      .app { grid-template-columns: 1fr; }
      .sidebar { min-height: auto; border-right: 0; border-bottom: 1px solid var(--line); }
      .nav-scroll { max-height: 320px; }
      .topbar { align-items: flex-start; flex-direction: column; }
      .top-actions { justify-content: flex-start; }
      .view { padding: 14px; }
      .stat-grid,
      .provider-grid,
      .hardware-grid,
      .label-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside class="sidebar">
      <div class="brand">
        <div class="brand-mark">AT</div>
        <div>
          <h1 class="brand-title">AI Trader</h1>
          <div class="brand-subtitle">Local control surface</div>
        </div>
      </div>
      <input id="action-search" class="search" type="text" placeholder="Filter actions">
      <div id="nav" class="nav-scroll"></div>
    </aside>
    <main class="main">
      <header class="topbar">
        <div class="title-block">
          <h2 id="page-title" class="page-title">Dashboard</h2>
          <div id="page-meta" class="page-meta">Loading workspace state</div>
        </div>
        <div class="top-actions">
          <div class="state-pill"><span id="state-dot" class="state-dot"></span><span id="state-text">Ready</span></div>
          <button id="refresh-button" class="button" type="button">Refresh</button>
        </div>
      </header>
      <div class="content">
        <section id="view-dashboard" class="view active">
          <div class="stat-grid" id="stat-grid"></div>
          <div class="dashboard-grid">
            <div class="split-stack">
              <div class="panel">
                <div class="panel-head">
                  <h3 class="panel-title">Provider Readiness</h3>
                  <button class="button" type="button" data-view-target="workbench">Run Status</button>
                </div>
                <div class="panel-body">
                  <div id="provider-grid" class="provider-grid"></div>
                </div>
              </div>
              <div class="panel">
                <div class="panel-head">
                  <h3 class="panel-title">Recent Artifacts</h3>
                  <button class="button" type="button" data-view-target="artifacts">Open</button>
                </div>
                <div class="panel-body">
                  <div id="dashboard-artifacts" class="artifact-list"></div>
                </div>
              </div>
            </div>
            <div class="panel">
              <div class="panel-head">
                <h3 class="panel-title">Ingestion Tuning</h3>
              </div>
              <div class="panel-body">
                <div id="hardware-grid" class="hardware-grid"></div>
              </div>
            </div>
          </div>
        </section>

        <section id="view-workbench" class="view flush">
          <div class="workbench">
            <div class="form-pane">
              <div class="panel">
                <div class="panel-head">
                  <h3 id="action-title" class="panel-title">Action</h3>
                </div>
                <div class="panel-body">
                  <form id="action-form" class="form-grid"></form>
                </div>
              </div>
            </div>
            <div class="terminal-pane">
              <div class="terminal-head">
                <div id="command-line" class="command-line">No command has run</div>
                <div class="terminal-meta">
                  <span id="line-count">0 lines</span>
                  <span id="run-result"></span>
                </div>
              </div>
              <div id="progress-wrap" class="progress-wrap">
                <div class="progress-bar"><div class="progress-fill"></div></div>
              </div>
              <pre id="terminal-output" class="terminal-output"></pre>
            </div>
          </div>
        </section>

        <section id="view-review" class="view">
          <div class="review-layout">
            <div class="panel">
              <div class="panel-head">
                <h3 class="panel-title">Queue</h3>
                <button id="load-review-button" class="button" type="button">Reload</button>
              </div>
              <div class="panel-body">
                <div id="review-list" class="review-list"></div>
              </div>
            </div>
            <div class="panel">
              <div class="panel-head">
                <h3 class="panel-title">Review Detail</h3>
              </div>
              <div class="panel-body">
                <div id="review-detail" class="empty-state">Select a queued item</div>
              </div>
            </div>
          </div>
        </section>

        <section id="view-artifacts" class="view">
          <div class="artifact-layout">
            <div class="panel">
              <div class="panel-head">
                <h3 class="panel-title">Artifacts</h3>
                <button id="artifact-refresh" class="button" type="button">Reload</button>
              </div>
              <div class="panel-body">
                <div id="artifact-list" class="artifact-list"></div>
              </div>
            </div>
            <div class="panel">
              <div class="panel-head">
                <h3 id="artifact-title" class="panel-title">Preview</h3>
              </div>
              <div class="panel-body">
                <pre id="artifact-preview" class="artifact-preview">Select an artifact</pre>
              </div>
            </div>
          </div>
        </section>
      </div>
    </main>
  </div>
  <div id="toast" class="toast"></div>

  <script>
    const $ = (id) => document.getElementById(id);
    const nav = $("nav");
    const actionSearch = $("action-search");
    const pageTitle = $("page-title");
    const pageMeta = $("page-meta");
    const stateDot = $("state-dot");
    const stateText = $("state-text");
    const refreshButton = $("refresh-button");
    const statGrid = $("stat-grid");
    const providerGrid = $("provider-grid");
    const hardwareGrid = $("hardware-grid");
    const dashboardArtifacts = $("dashboard-artifacts");
    const actionForm = $("action-form");
    const actionTitle = $("action-title");
    const commandLine = $("command-line");
    const terminalOutput = $("terminal-output");
    const runResult = $("run-result");
    const lineCount = $("line-count");
    const progressWrap = $("progress-wrap");
    const reviewList = $("review-list");
    const reviewDetail = $("review-detail");
    const artifactList = $("artifact-list");
    const artifactPreview = $("artifact-preview");
    const artifactTitle = $("artifact-title");
    const toast = $("toast");

    let actions = [];
    let overview = null;
    let activeAction = null;
    let activeView = "dashboard";
    let abortController = null;
    let lastReviewEntries = [];

    const pageLabels = {
      dashboard: "Dashboard",
      workbench: "Workbench",
      review: "Label Review",
      artifacts: "Artifacts"
    };

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }

    function formatBytes(bytes) {
      const value = Number(bytes || 0);
      if (value < 1024) return value + " B";
      if (value < 1024 * 1024) return (value / 1024).toFixed(1) + " KB";
      return (value / (1024 * 1024)).toFixed(1) + " MB";
    }

    function formatDate(value) {
      if (!value) return "";
      try {
        return new Intl.DateTimeFormat(undefined, {
          month: "short",
          day: "2-digit",
          hour: "2-digit",
          minute: "2-digit"
        }).format(new Date(value));
      } catch (_) {
        return value;
      }
    }

    function showToast(message) {
      toast.textContent = message;
      toast.classList.add("visible");
      setTimeout(() => toast.classList.remove("visible"), 2600);
    }

    function setRunState(mode, text) {
      stateText.textContent = text;
      stateDot.classList.remove("running", "failed");
      if (mode === "running") stateDot.classList.add("running");
      if (mode === "failed") stateDot.classList.add("failed");
    }

    async function fetchJson(url, options) {
      const response = await fetch(url, options);
      const payload = await response.json().catch(() => ({}));
      if (!response.ok || payload.success === false) {
        throw new Error(payload.error || response.statusText);
      }
      return payload;
    }

    async function boot() {
      await Promise.all([loadActions(), refreshOverview()]);
      renderNav();
      selectAction("status");
      await loadReviewQueue();
      wireStaticButtons();
      showView("dashboard");
    }

    async function loadActions() {
      const payload = await fetchJson("/api/actions");
      actions = payload.actions || [];
    }

    async function refreshOverview() {
      overview = await fetchJson("/api/overview");
      renderDashboard();
      renderArtifacts();
      updateBadges();
      pageMeta.textContent = overview.working_dir || "";
    }

    function wireStaticButtons() {
      refreshButton.onclick = async () => {
        await refreshOverview();
        showToast("Workspace refreshed");
      };
      $("load-review-button").onclick = loadReviewQueue;
      $("artifact-refresh").onclick = refreshOverview;
      actionSearch.addEventListener("input", renderNav);
      document.querySelectorAll("[data-view-target]").forEach((button) => {
        button.addEventListener("click", () => {
          if (button.dataset.viewTarget === "workbench") selectAction("status");
          showView(button.dataset.viewTarget);
        });
      });
    }

    function showView(name) {
      activeView = name;
      document.querySelectorAll(".view").forEach((node) => {
        node.classList.toggle("active", node.id === "view-" + name);
      });
      pageTitle.textContent = pageLabels[name] || "AI Trader";
      document.querySelectorAll(".nav-button").forEach((node) => {
        const isAction = Boolean(node.dataset.actionId);
        const isActiveAction = node.dataset.actionId === activeAction?.id;
        node.classList.toggle("active", node.dataset.view === name && (!isAction || isActiveAction));
      });
      if (name === "review") loadReviewQueue();
      if (name === "artifacts") renderArtifacts();
    }

    function renderNav() {
      const filter = actionSearch.value.trim().toLowerCase();
      nav.innerHTML = "";
      const staticItems = [
        ["dashboard", "Dashboard", overview?.metrics?.training_examples ?? 0],
        ["review", "Label Review", overview?.metrics?.review_pending ?? 0],
        ["artifacts", "Artifacts", overview?.artifacts?.length ?? 0]
      ];
      const staticSection = document.createElement("div");
      staticSection.className = "nav-section";
      staticSection.textContent = "Workspace";
      nav.appendChild(staticSection);
      const staticList = document.createElement("div");
      staticList.className = "nav-list";
      for (const [view, label, count] of staticItems) {
        const button = document.createElement("button");
        button.className = "nav-button";
        button.type = "button";
        button.dataset.view = view;
        button.innerHTML = `<span>${escapeHtml(label)}</span><span class="badge">${count}</span>`;
        button.onclick = () => showView(view);
        staticList.appendChild(button);
      }
      nav.appendChild(staticList);

      const grouped = {};
      for (const action of actions) {
        const haystack = `${action.label} ${action.group}`.toLowerCase();
        if (filter && !haystack.includes(filter)) continue;
        grouped[action.group] = grouped[action.group] || [];
        grouped[action.group].push(action);
      }
      for (const [group, items] of Object.entries(grouped)) {
        const header = document.createElement("div");
        header.className = "nav-section";
        header.textContent = group;
        nav.appendChild(header);
        const list = document.createElement("div");
        list.className = "nav-list";
        for (const action of items) {
          const button = document.createElement("button");
          button.className = "nav-button";
          button.type = "button";
          button.dataset.actionId = action.id;
          button.dataset.view = "workbench";
          button.innerHTML = `<span>${escapeHtml(action.label)}</span>`;
          button.onclick = () => {
            selectAction(action.id);
            showView("workbench");
          };
          list.appendChild(button);
        }
        nav.appendChild(list);
      }
      document.querySelectorAll(".nav-button").forEach((node) => {
        const isAction = Boolean(node.dataset.actionId);
        const isActiveAction = node.dataset.actionId === activeAction?.id;
        node.classList.toggle(
          "active",
          node.dataset.view === activeView && (!isAction || isActiveAction)
        );
      });
    }

    function updateBadges() {
      renderNav();
    }

    function renderDashboard() {
      if (!overview) return;
      const metrics = overview.metrics || {};
      const hardware = overview.hardware || {};
      statGrid.innerHTML = [
        statCard("Training Examples", metrics.training_examples, overview.paths?.training_examples),
        statCard("Review Pending", metrics.review_pending, overview.paths?.review_queue),
        statCard("Actions", metrics.actions, "Whitelisted local workflows"),
        statCard("Network", hardware.network_tier || "unknown", hardware.network_mbps ? hardware.network_mbps.toFixed(1) + " Mbps" : "cached or disabled")
      ].join("");

      const providers = overview.providers || {};
      providerGrid.innerHTML = Object.entries(providers)
        .map(([name, ready]) => `
          <div class="mini-row">
            <strong>${escapeHtml(name.replaceAll("_", " "))}</strong>
            <span class="status-text ${ready ? "ok" : "off"}">${ready ? "ready" : "missing"}</span>
          </div>`)
        .join("");

      hardwareGrid.innerHTML = [
        ["CPU", `${hardware.physical_cores ?? 0} physical / ${hardware.logical_cores ?? 0} logical`],
        ["RAM", `${hardware.ram_available_gb ?? 0} GB free / ${hardware.ram_gb ?? 0} GB total`],
        ["Source Workers", hardware.source_workers],
        ["Price Workers", hardware.price_workers],
        ["Ticker Workers", hardware.ticker_workers],
        ["HTTP Connections", hardware.http_connections],
        ["Write Buffer", hardware.write_buffer],
        ["Cache", overview.paths?.cache_dir || ""]
      ].map(([label, value]) => `
        <div class="mini-row">
          <strong>${escapeHtml(label)}</strong>
          <span class="status-text">${escapeHtml(value)}</span>
        </div>`).join("");

      const artifacts = overview.artifacts || [];
      dashboardArtifacts.innerHTML = artifacts.slice(0, 6).map(artifactRowHtml).join("") ||
        `<div class="empty-state">No artifacts found</div>`;
      dashboardArtifacts.querySelectorAll("[data-artifact-path]").forEach((row) => {
        row.addEventListener("click", () => {
          showView("artifacts");
          loadArtifact(row.dataset.artifactPath);
        });
      });
    }

    function statCard(label, value, foot) {
      return `
        <div class="stat-card">
          <div class="stat-label">${escapeHtml(label)}</div>
          <div class="stat-value">${escapeHtml(value ?? 0)}</div>
          <div class="stat-foot">${escapeHtml(foot || "")}</div>
        </div>`;
    }

    function selectAction(id) {
      activeAction = actions.find((action) => action.id === id) || actions[0];
      if (!activeAction) return;
      actionTitle.textContent = activeAction.label;
      renderActionForm();
      renderNav();
    }

    function renderActionForm() {
      actionForm.innerHTML = "";
      for (const field of activeAction.fields || []) {
        if (field.type === "checkbox") {
          const label = document.createElement("label");
          label.className = "check-row";
          const input = document.createElement("input");
          input.type = "checkbox";
          input.name = field.name;
          input.checked = Boolean(field.default);
          label.appendChild(input);
          label.appendChild(document.createTextNode(field.label));
          actionForm.appendChild(label);
          continue;
        }
        const label = document.createElement("label");
        label.className = "field";
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
        input.required = Boolean(field.required);
        input.placeholder = field.placeholder || "";
        input.value = field.default === null || field.default === undefined ? "" : field.default;
        label.appendChild(input);
        actionForm.appendChild(label);
      }
      const controls = document.createElement("div");
      controls.className = "form-actions";
      controls.innerHTML = `
        <button id="run-button" class="button primary" type="submit">${activeAction.streaming ? "Run Live" : "Run"}</button>
        <button id="clear-button" class="button" type="button">Clear</button>
        <button id="cancel-button" class="button danger" type="button">Cancel</button>
      `;
      actionForm.appendChild(controls);
      $("clear-button").onclick = clearTerminal;
      $("cancel-button").onclick = () => {
        if (abortController) abortController.abort();
      };
    }

    function collectInputs() {
      const inputs = {};
      for (const field of activeAction.fields || []) {
        const node = actionForm.elements[field.name];
        if (!node) continue;
        inputs[field.name] = field.type === "checkbox" ? node.checked : node.value;
      }
      return inputs;
    }

    actionForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (!activeAction) return;
      const button = $("run-button");
      button.disabled = true;
      setRunState("running", "Running");
      runResult.textContent = "";
      terminalOutput.textContent = "";
      lineCount.textContent = "0 lines";
      try {
        if (activeAction.streaming) {
          await runStreaming(collectInputs());
        } else {
          await runBlocking(collectInputs());
        }
        await refreshOverview();
      } finally {
        button.disabled = false;
        progressWrap.classList.remove("visible");
      }
    });

    function clearTerminal() {
      terminalOutput.textContent = "";
      commandLine.textContent = "No command has run";
      runResult.textContent = "";
      lineCount.textContent = "0 lines";
      setRunState("ready", "Ready");
    }

    async function runBlocking(inputs) {
      try {
        const payload = await fetchJson("/api/run", {
          method: "POST",
          headers: {"content-type": "application/json"},
          body: JSON.stringify({action: activeAction.id, inputs})
        });
        commandLine.textContent = payload.command.join(" ");
        terminalOutput.textContent = [
          payload.stdout || "",
          payload.stderr ? "\n[stderr]\n" + payload.stderr : "",
          payload.log_file ? "\n[log]\n" + payload.log_file : ""
        ].join("");
        runResult.textContent = payload.success ? "ok" : "exit " + payload.returncode;
        runResult.className = payload.success ? "terminal-ok" : "terminal-error";
        setRunState(payload.success ? "ready" : "failed", payload.success ? "Ready" : "Failed");
      } catch (error) {
        runResult.textContent = "error";
        runResult.className = "terminal-error";
        terminalOutput.textContent = error.message;
        setRunState("failed", "Failed");
      }
    }

    async function runStreaming(inputs) {
      progressWrap.classList.add("visible");
      abortController = new AbortController();
      let lines = 0;
      try {
        const response = await fetch("/api/stream", {
          method: "POST",
          headers: {"content-type": "application/json"},
          body: JSON.stringify({action: activeAction.id, inputs}),
          signal: abortController.signal
        });
        if (!response.ok) {
          const err = await response.json().catch(() => ({error: response.statusText}));
          throw new Error(err.error || "Stream failed");
        }
        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = "";
        while (true) {
          const {done, value} = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, {stream: true});
          const blocks = buffer.split("\n\n");
          buffer = blocks.pop() || "";
          for (const block of blocks) {
            const parsed = parseSseBlock(block);
            if (!parsed) continue;
            if (parsed.event === "start") {
              const payload = JSON.parse(parsed.data);
              commandLine.textContent = payload.command.join(" ");
            } else if (parsed.event === "line") {
              terminalOutput.textContent += parsed.data + "\n";
              terminalOutput.scrollTop = terminalOutput.scrollHeight;
              lines += 1;
              lineCount.textContent = lines + " lines";
            } else if (parsed.event === "done") {
              const payload = JSON.parse(parsed.data);
              const ok = payload.returncode === 0;
              runResult.textContent = ok ? "ok" : "exit " + payload.returncode;
              runResult.className = ok ? "terminal-ok" : "terminal-error";
              setRunState(ok ? "ready" : "failed", ok ? "Ready" : "Failed");
            } else if (parsed.event === "error") {
              terminalOutput.textContent += "\n[error] " + parsed.data + "\n";
              runResult.textContent = "error";
              runResult.className = "terminal-error";
              setRunState("failed", "Failed");
            }
          }
        }
      } catch (error) {
        if (error.name !== "AbortError") {
          terminalOutput.textContent += "\n[error] " + error.message + "\n";
          runResult.textContent = "error";
          runResult.className = "terminal-error";
          setRunState("failed", "Failed");
        } else {
          terminalOutput.textContent += "\n[cancelled]\n";
          runResult.textContent = "cancelled";
          runResult.className = "terminal-error";
          setRunState("ready", "Ready");
        }
      } finally {
        abortController = null;
        progressWrap.classList.remove("visible");
      }
    }

    function parseSseBlock(block) {
      let event = "message";
      const data = [];
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
      }
      return {event, data: data.join("\n")};
    }

    async function loadReviewQueue() {
      if (!reviewList) return;
      reviewList.innerHTML = `<div class="empty-state">Loading queue</div>`;
      try {
        const payload = await fetchJson("/api/review?limit=50&offset=0");
        lastReviewEntries = payload.entries || [];
        reviewList.innerHTML = lastReviewEntries.map(reviewCardHtml).join("") ||
          `<div class="empty-state">Queue is empty</div>`;
        reviewList.querySelectorAll("[data-review-id]").forEach((card) => {
          card.addEventListener("click", () => showReviewDetail(card.dataset.reviewId));
        });
        reviewList.querySelectorAll("[data-review-action]").forEach((button) => {
          button.addEventListener("click", (event) => {
            event.stopPropagation();
            decideReview(button.dataset.reviewId, button.dataset.reviewAction);
          });
        });
        reviewList.querySelectorAll("[data-override]").forEach((button) => {
          button.addEventListener("click", (event) => {
            event.stopPropagation();
            const field = button.dataset.override;
            const id = button.dataset.reviewId;
            const card = reviewList.querySelector(`[data-review-id="${CSS.escape(id)}"]`);
            card.dataset[field] = button.dataset.value;
            card.querySelectorAll(`[data-override="${field}"]`).forEach((node) => {
              node.classList.toggle("selected", node === button);
            });
          });
        });
      } catch (error) {
        reviewList.innerHTML = `<div class="empty-state">${escapeHtml(error.message)}</div>`;
      }
    }

    function reviewCardHtml(entry) {
      const example = entry.example || {};
      const bundle = example.signal_bundle || {};
      const plan = example.trade_plan || {};
      const pnl = Number(example.pnl_pct || 0);
      const pnlClass = pnl > 0.01 ? "pos" : pnl < -0.01 ? "neg" : "neu";
      const pnlText = (pnl >= 0 ? "+" : "") + (pnl * 100).toFixed(1) + "%";
      return `
        <div class="review-card" data-review-id="${escapeHtml(entry.queue_id)}">
          <div class="review-head">
            <div class="review-symbol">${escapeHtml(bundle.ticker || "TICKER")}</div>
            <div class="pnl ${pnlClass}">${pnlText}</div>
          </div>
          <div class="meta-line">
            <span>${escapeHtml(bundle.as_of || "")}</span>
            <span>${escapeHtml(plan.direction || "")}</span>
            <span>conviction ${(Number(plan.conviction || 0)).toFixed(2)}</span>
          </div>
          <div class="label-grid">
            <div class="label-box"><span>Outcome</span>${escapeHtml(entry.auto_outcome_label)}</div>
            <div class="label-box"><span>Quality</span>${escapeHtml(entry.auto_signal_quality)}</div>
          </div>
          <div class="chip-row">
            ${["strong_win", "win", "neutral", "loss", "strong_loss"].map((value) => `<button class="chip" type="button" data-review-id="${escapeHtml(entry.queue_id)}" data-override="outcomeOverride" data-value="${value}">${value}</button>`).join("")}
          </div>
          <div class="chip-row">
            ${["high", "medium", "low"].map((value) => `<button class="chip" type="button" data-review-id="${escapeHtml(entry.queue_id)}" data-override="qualityOverride" data-value="${value}">${value}</button>`).join("")}
          </div>
          <div class="card-actions">
            <button class="button primary" type="button" data-review-id="${escapeHtml(entry.queue_id)}" data-review-action="accept">Accept</button>
            <button class="button" type="button" data-review-id="${escapeHtml(entry.queue_id)}" data-review-action="skip">Skip</button>
          </div>
        </div>`;
    }

    function showReviewDetail(id) {
      const entry = lastReviewEntries.find((item) => item.queue_id === id);
      if (!entry) return;
      const example = entry.example || {};
      const bundle = example.signal_bundle || {};
      const signals = bundle.signals || [];
      reviewDetail.innerHTML = `
        <div class="split-stack">
          <div class="mini-row"><strong>${escapeHtml(bundle.ticker || "")}</strong><span class="status-text">${escapeHtml(bundle.as_of || "")}</span></div>
          <div class="signal-list">
            ${signals.map((signal) => `
              <div>${escapeHtml(signal.name)} | ${escapeHtml(signal.direction)} | strength ${(Number(signal.strength || 0)).toFixed(2)} | confidence ${(Number(signal.confidence || 0)).toFixed(2)}</div>
            `).join("") || "<div>No signals</div>"}
          </div>
          <div class="signal-list">
            ${(entry.review_reasons || []).map((reason) => `<div>${escapeHtml(reason)}</div>`).join("") || "<div>No review reasons</div>"}
          </div>
        </div>`;
    }

    async function decideReview(id, action) {
      const card = reviewList.querySelector(`[data-review-id="${CSS.escape(id)}"]`);
      const body = {
        queue_id: id,
        action,
        outcome_override: card?.dataset?.outcomeOverride || null,
        quality_override: card?.dataset?.qualityOverride || null
      };
      try {
        await fetchJson("/api/review/decide", {
          method: "POST",
          headers: {"content-type": "application/json"},
          body: JSON.stringify(body)
        });
        if (card) {
          card.classList.add("done");
          setTimeout(() => card.remove(), 220);
        }
        await Promise.all([refreshOverview(), loadReviewQueue()]);
        showToast(action === "skip" ? "Item skipped" : "Label saved");
      } catch (error) {
        showToast(error.message);
      }
    }

    function renderArtifacts() {
      const artifacts = overview?.artifacts || [];
      artifactList.innerHTML = artifacts.map(artifactRowHtml).join("") ||
        `<div class="empty-state">No artifacts found</div>`;
      artifactList.querySelectorAll("[data-artifact-path]").forEach((row) => {
        row.addEventListener("click", () => loadArtifact(row.dataset.artifactPath));
      });
    }

    function artifactRowHtml(item) {
      return `
        <button class="artifact-row" type="button" data-artifact-path="${escapeHtml(item.path)}">
          <span style="min-width:0">
            <span class="artifact-name">${escapeHtml(item.name)}</span>
            <span class="artifact-path">${escapeHtml(item.path)}</span>
          </span>
          <span class="status-text">${formatBytes(item.size_bytes)}</span>
        </button>`;
    }

    async function loadArtifact(path) {
      try {
        const payload = await fetchJson("/api/artifact?path=" + encodeURIComponent(path));
        artifactTitle.textContent = payload.artifact.name;
        artifactPreview.textContent = formatArtifactContent(payload.content, payload.artifact.kind);
        if (payload.truncated) artifactPreview.textContent += "\n\n[truncated]";
      } catch (error) {
        artifactTitle.textContent = "Preview";
        artifactPreview.textContent = error.message;
      }
    }

    function formatArtifactContent(content, kind) {
      if (kind === "json") {
        try {
          return JSON.stringify(JSON.parse(content), null, 2);
        } catch (_) {
          return content;
        }
      }
      if (kind === "jsonl") {
        const lines = content.split(/\r?\n/).filter(Boolean).slice(0, 80);
        return lines.map((line) => {
          try {
            return JSON.stringify(JSON.parse(line), null, 2);
          } catch (_) {
            return line;
          }
        }).join("\n");
      }
      return content;
    }

    boot().catch((error) => {
      setRunState("failed", "Failed");
      pageMeta.textContent = error.message;
    });
  </script>
</body>
</html>
"""
