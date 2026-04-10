use crate::history::{self, EventRow, RunRow};
use anyhow::Result;
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{Html, IntoResponse};
use axum::routing::get;
use axum::{Json, Router};
use serde::Deserialize;
use std::net::SocketAddr;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct DashboardArgs {
    pub host: String,
    pub port: u16,
}

#[derive(Clone)]
struct DashboardState {
    store: Arc<history::HistoryStore>,
}

#[derive(Debug, Deserialize)]
struct LimitQuery {
    limit: Option<usize>,
}

pub async fn run_dashboard(args: DashboardArgs) -> Result<()> {
    let Some(store) = history::global() else {
        anyhow::bail!("history store is unavailable; cannot launch dashboard");
    };
    let addr: SocketAddr = format!("{}:{}", args.host, args.port).parse()?;

    let app = Router::new()
        .route("/", get(index))
        .route("/api/runs", get(list_runs))
        .route("/api/runs/{run_id}/events", get(list_events))
        .route("/api/artifacts/{id}", get(get_artifact))
        .with_state(DashboardState { store });

    eprintln!("Dashboard listening on http://{}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn index() -> Html<&'static str> {
    Html(INDEX_HTML)
}

async fn list_runs(
    State(state): State<DashboardState>,
    Query(query): Query<LimitQuery>,
) -> Result<Json<Vec<RunRow>>, StatusCode> {
    let limit = query.limit.unwrap_or(100).clamp(1, 500);
    state
        .store
        .list_runs(limit)
        .map(Json)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

async fn list_events(
    State(state): State<DashboardState>,
    Path(run_id): Path<String>,
    Query(query): Query<LimitQuery>,
) -> Result<Json<Vec<EventRow>>, StatusCode> {
    let limit = query.limit.unwrap_or(5000).clamp(1, 20_000);
    state
        .store
        .list_events(&run_id, limit)
        .map(Json)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

async fn get_artifact(
    State(state): State<DashboardState>,
    Path(id): Path<i64>,
) -> impl IntoResponse {
    match state.store.get_artifact(id) {
        Ok(Some(artifact)) => Json(artifact).into_response(),
        Ok(None) => StatusCode::NOT_FOUND.into_response(),
        Err(_) => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    }
}

const INDEX_HTML: &str = r#"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>lambda_rlm dashboard</title>
  <style>
    :root {
      --bg: #f5f4ef;
      --panel: #ffffff;
      --ink: #1a1a1a;
      --muted: #6a6a66;
      --line: #ddd9cf;
      --accent: #1f6f5f;
      --warn: #b46215;
      --err: #9f2a2a;
      --mono: "SF Mono", "Menlo", "Consolas", monospace;
      --sans: "Avenir Next", "Gill Sans", "Trebuchet MS", sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: var(--sans);
      color: var(--ink);
      background: linear-gradient(160deg, #f5f4ef 0%, #ece8da 100%);
      min-height: 100vh;
    }
    header {
      padding: 14px 18px;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.8);
      backdrop-filter: blur(4px);
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    h1 { margin: 0; font-size: 18px; letter-spacing: 0.02em; }
    .hint { color: var(--muted); font-size: 12px; }
    .grid {
      display: grid;
      grid-template-columns: 320px 1fr 1.1fr;
      gap: 12px;
      padding: 12px;
      height: calc(100vh - 56px);
    }
    .panel {
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 10px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      min-height: 0;
    }
    .panel h2 {
      margin: 0;
      font-size: 13px;
      color: var(--muted);
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .scroll { overflow: auto; min-height: 0; }
    .run {
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      cursor: pointer;
    }
    .run:hover { background: #f7f7f3; }
    .run.active { background: #edf6f3; border-left: 3px solid var(--accent); }
    .run .top { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
    .run .id { font-family: var(--mono); font-size: 11px; color: var(--muted); }
    .badge {
      font-size: 10px;
      border-radius: 999px;
      padding: 2px 8px;
      background: #e8ece8;
      text-transform: uppercase;
      letter-spacing: 0.07em;
    }
    .badge.running { background: #e8f4ef; color: var(--accent); }
    .badge.error { background: #fdeeee; color: var(--err); }
    .badge.ok { background: #edf6f1; color: #2a6a3c; }
    .run .q { margin-top: 6px; font-size: 13px; line-height: 1.35; }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }
    th, td {
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }
    th { position: sticky; top: 0; background: #faf9f5; z-index: 1; color: var(--muted); font-weight: 600; }
    tr:hover td { background: #f8f8f4; }
    tr.active td { background: #edf6f3; }
    .mono { font-family: var(--mono); }
    .pill {
      display: inline-block;
      padding: 2px 7px;
      border-radius: 999px;
      background: #ecebe4;
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }
    .pill.warn { color: var(--warn); background: #fff3e6; }
    .pill.error { color: var(--err); background: #fdeeee; }
    pre {
      margin: 0;
      padding: 12px;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: var(--mono);
      font-size: 12px;
      line-height: 1.45;
    }
    .meta {
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      font-size: 12px;
    }
    .meta .k { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; }
    .meta .v { font-family: var(--mono); font-size: 11px; }
    @media (max-width: 1000px) {
      .grid { grid-template-columns: 1fr; height: auto; }
      .panel { min-height: 280px; }
    }
  </style>
</head>
<body>
  <header>
    <h1>lambda_rlm dashboard</h1>
    <div class="hint">History-backed view from ~/.lambda-rlm/history.db</div>
  </header>
  <div class="grid">
    <section class="panel">
      <h2>Runs</h2>
      <div id="runs" class="scroll"></div>
    </section>
    <section class="panel">
      <h2>Timeline</h2>
      <div class="scroll">
        <table>
          <thead>
            <tr>
              <th>Seq</th>
              <th>Iter</th>
              <th>Component</th>
              <th>Kind</th>
              <th>Status</th>
              <th>Message</th>
            </tr>
          </thead>
          <tbody id="events"></tbody>
        </table>
      </div>
    </section>
    <section class="panel">
      <h2>Details</h2>
      <div id="detail-meta" class="meta"></div>
      <div class="scroll"><pre id="detail">Select an event</pre></div>
    </section>
  </div>
  <script>
    const state = { runs: [], events: [], selectedRunId: null, selectedEventId: null };

    async function fetchJson(url) {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res.json();
    }

    function badge(status) {
      const cls = status === 'error' ? 'error' : status === 'running' ? 'running' : 'ok';
      return `<span class="badge ${cls}">${status}</span>`;
    }

    function renderRuns() {
      const el = document.getElementById('runs');
      el.innerHTML = state.runs.map(run => `
        <div class="run ${run.run_id === state.selectedRunId ? 'active' : ''}" data-id="${run.run_id}">
          <div class="top">
            <span class="id">${run.run_id}</span>
            ${badge(run.status)}
          </div>
          <div class="q">${(run.question || '(no question)').replace(/</g, '&lt;')}</div>
        </div>
      `).join('');

      for (const row of el.querySelectorAll('.run')) {
        row.onclick = () => selectRun(row.dataset.id);
      }
    }

    function statusPill(status) {
      if (status === 'error') return '<span class="pill error">error</span>';
      if (status === 'warn') return '<span class="pill warn">warn</span>';
      return `<span class="pill">${status}</span>`;
    }

    function renderEvents() {
      const tbody = document.getElementById('events');
      tbody.innerHTML = state.events.map(event => `
        <tr class="${event.id === state.selectedEventId ? 'active' : ''}" data-id="${event.id}">
          <td class="mono">${event.seq}</td>
          <td class="mono">${event.iteration}</td>
          <td>${event.component}</td>
          <td>${event.kind}</td>
          <td>${statusPill(event.status)}</td>
          <td>${(event.message || '').replace(/</g, '&lt;')}</td>
        </tr>
      `).join('');

      for (const row of tbody.querySelectorAll('tr')) {
        row.onclick = () => selectEvent(Number(row.dataset.id));
      }
    }

    function metaItem(k, v) {
      return `<div><div class="k">${k}</div><div class="v">${(v ?? '').toString().replace(/</g, '&lt;')}</div></div>`;
    }

    async function selectEvent(eventId) {
      state.selectedEventId = eventId;
      renderEvents();
      const event = state.events.find(e => e.id === eventId);
      if (!event) return;

      const detailMeta = document.getElementById('detail-meta');
      detailMeta.innerHTML = [
        metaItem('run', event.run_id),
        metaItem('seq', event.seq),
        metaItem('iteration', event.iteration),
        metaItem('component', event.component),
        metaItem('kind', event.kind),
        metaItem('status', event.status),
        metaItem('call', event.call_no || ''),
        metaItem('attempt', event.attempt || ''),
      ].join('');

      const sections = [];
      sections.push(`message:\n${event.message || ''}`);

      if (event.request_artifact_id) {
        const req = await fetchJson(`/api/artifacts/${event.request_artifact_id}`);
        sections.push(`request:\n${req.content}`);
      }
      if (event.response_artifact_id) {
        const res = await fetchJson(`/api/artifacts/${event.response_artifact_id}`);
        sections.push(`response:\n${res.content}`);
      }
      if (event.error_artifact_id) {
        const err = await fetchJson(`/api/artifacts/${event.error_artifact_id}`);
        sections.push(`error:\n${err.content}`);
      }
      if (event.extra_json) {
        sections.push(`extra:\n${event.extra_json}`);
      }

      document.getElementById('detail').textContent = sections.join('\n\n---\n\n');
    }

    async function selectRun(runId) {
      state.selectedRunId = runId;
      state.selectedEventId = null;
      renderRuns();
      state.events = await fetchJson(`/api/runs/${runId}/events?limit=20000`);
      renderEvents();
      document.getElementById('detail').textContent = 'Select an event';
      document.getElementById('detail-meta').innerHTML = '';
      if (state.events.length > 0) {
        await selectEvent(state.events[0].id);
      }
    }

    async function boot() {
      state.runs = await fetchJson('/api/runs?limit=200');
      renderRuns();
      if (state.runs.length > 0) {
        await selectRun(state.runs[0].run_id);
      }
    }

    boot().catch(err => {
      document.getElementById('detail').textContent = `dashboard error: ${err.message}`;
    });
  </script>
</body>
</html>
"#;
