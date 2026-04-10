use anyhow::{Context, Result};
use rusqlite::{params, Connection, OptionalExtension};
use serde::Serialize;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock, RwLock};

const STORE_DIR_NAME: &str = ".lambda-rlm";
const DB_FILE_NAME: &str = "history.db";

static STORE: OnceLock<Arc<HistoryStore>> = OnceLock::new();

#[derive(Clone, Debug)]
struct ActiveContext {
    run_id: String,
    iteration: usize,
    seq: u64,
}

static ACTIVE_CONTEXT: OnceLock<RwLock<Option<ActiveContext>>> = OnceLock::new();

fn active_context() -> &'static RwLock<Option<ActiveContext>> {
    ACTIVE_CONTEXT.get_or_init(|| RwLock::new(None))
}

fn next_context_seq() -> Option<(String, usize, u64)> {
    let mut guard = active_context().write().ok()?;
    let ctx = guard.as_mut()?;
    ctx.seq = ctx.seq.saturating_add(1);
    Some((ctx.run_id.clone(), ctx.iteration, ctx.seq))
}

#[derive(Clone, Debug)]
pub struct HistoryStore {
    db_path: PathBuf,
}

#[derive(Clone, Debug)]
pub struct RunStart {
    pub mode: String,
    pub path: Option<String>,
    pub question: Option<String>,
    pub task: String,
    pub generator: Option<String>,
    pub model: String,
    pub config_fingerprint: String,
}

#[derive(Clone, Debug)]
pub struct EventRecord {
    pub component: String,
    pub kind: String,
    pub status: String,
    pub message: String,
    pub call_no: Option<u64>,
    pub attempt: Option<usize>,
    pub latency_ms: Option<u64>,
    pub cache_hit: Option<bool>,
    pub idempotency_key: Option<String>,
    pub request_payload: Option<String>,
    pub response_payload: Option<String>,
    pub error_payload: Option<String>,
    pub extra_json: Option<String>,
    pub iteration_override: Option<usize>,
}

#[derive(Clone, Debug, Serialize)]
pub struct RunRow {
    pub run_id: String,
    pub started_at: String,
    pub finished_at: Option<String>,
    pub status: String,
    pub mode: String,
    pub path: Option<String>,
    pub question: Option<String>,
    pub task: String,
    pub generator: Option<String>,
    pub model: String,
    pub config_fingerprint: String,
    pub error_text: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct EventRow {
    pub id: i64,
    pub run_id: String,
    pub seq: u64,
    pub ts: String,
    pub iteration: usize,
    pub component: String,
    pub kind: String,
    pub status: String,
    pub message: String,
    pub call_no: Option<u64>,
    pub attempt: Option<usize>,
    pub latency_ms: Option<u64>,
    pub cache_hit: Option<bool>,
    pub idempotency_key: Option<String>,
    pub request_artifact_id: Option<i64>,
    pub response_artifact_id: Option<i64>,
    pub error_artifact_id: Option<i64>,
    pub extra_json: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ArtifactRow {
    pub id: i64,
    pub run_id: String,
    pub kind: String,
    pub content: String,
    pub created_at: String,
}

impl HistoryStore {
    pub fn initialize_global() -> Option<Arc<Self>> {
        if let Some(existing) = STORE.get() {
            return Some(Arc::clone(existing));
        }

        match Self::new() {
            Ok(store) => {
                let store = Arc::new(store);
                let _ = STORE.set(Arc::clone(&store));
                Some(store)
            }
            Err(error) => {
                tracing::warn!(error = %error, "history store unavailable; continuing without persistence");
                None
            }
        }
    }

    pub fn global() -> Option<Arc<Self>> {
        STORE.get().cloned()
    }

    fn new() -> Result<Self> {
        let dir = global_store_dir().context("failed to resolve history home")?;
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("failed to create {}", dir.display()))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700));
        }

        let db_path = dir.join(DB_FILE_NAME);
        let conn = Connection::open(&db_path)
            .with_context(|| format!("failed to open {}", db_path.display()))?;
        Self::initialize_schema(&conn)?;
        Ok(Self { db_path })
    }

    fn connect(&self) -> Result<Connection> {
        Connection::open(&self.db_path)
            .with_context(|| format!("failed to open {}", self.db_path.display()))
    }

    fn initialize_schema(conn: &Connection) -> Result<()> {
        conn.execute_batch(
            "
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;
            PRAGMA foreign_keys = ON;

            CREATE TABLE IF NOT EXISTS schema_meta (
                version INTEGER NOT NULL
            );

            INSERT INTO schema_meta(version)
            SELECT 1
            WHERE NOT EXISTS (SELECT 1 FROM schema_meta);

            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                finished_at TEXT,
                status TEXT NOT NULL,
                mode TEXT NOT NULL,
                path TEXT,
                question TEXT,
                task TEXT NOT NULL,
                generator TEXT,
                model TEXT NOT NULL,
                config_fingerprint TEXT NOT NULL,
                final_artifact_id INTEGER,
                error_text TEXT
            );

            CREATE TABLE IF NOT EXISTS artifacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                ts TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                iteration INTEGER NOT NULL,
                component TEXT NOT NULL,
                kind TEXT NOT NULL,
                status TEXT NOT NULL,
                message TEXT NOT NULL,
                call_no INTEGER,
                attempt INTEGER,
                latency_ms INTEGER,
                cache_hit INTEGER,
                idempotency_key TEXT,
                request_artifact_id INTEGER,
                response_artifact_id INTEGER,
                error_artifact_id INTEGER,
                extra_json TEXT,
                FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_runs_started_at ON runs(started_at DESC);
            CREATE INDEX IF NOT EXISTS idx_events_run_seq ON events(run_id, seq);
            CREATE INDEX IF NOT EXISTS idx_artifacts_run ON artifacts(run_id, id);
            ",
        )
        .context("failed to initialize history schema")?;
        Ok(())
    }

    fn put_artifact(conn: &Connection, run_id: &str, kind: &str, content: &str) -> Result<i64> {
        conn.execute(
            "INSERT INTO artifacts (run_id, kind, content) VALUES (?1, ?2, ?3)",
            params![run_id, kind, content],
        )?;
        Ok(conn.last_insert_rowid())
    }

    pub fn start_run(&self, payload: &RunStart) -> Result<String> {
        let run_id = next_run_id();
        let conn = self.connect()?;
        conn.execute(
            "INSERT INTO runs (run_id, status, mode, path, question, task, generator, model, config_fingerprint)
             VALUES (?1, 'running', ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                run_id,
                payload.mode,
                payload.path,
                payload.question,
                payload.task,
                payload.generator,
                payload.model,
                payload.config_fingerprint,
            ],
        )?;

        if let Ok(mut guard) = active_context().write() {
            *guard = Some(ActiveContext {
                run_id: run_id.clone(),
                iteration: 0,
                seq: 0,
            });
        }
        Ok(run_id)
    }

    pub fn set_iteration(iteration: usize) {
        if let Ok(mut guard) = active_context().write() {
            if let Some(ctx) = guard.as_mut() {
                ctx.iteration = iteration;
            }
        }
    }

    pub fn clear_active_context() {
        if let Ok(mut guard) = active_context().write() {
            *guard = None;
        }
    }

    pub fn finish_run(
        &self,
        run_id: &str,
        status: &str,
        final_output: Option<&str>,
        error_text: Option<&str>,
    ) -> Result<()> {
        let conn = self.connect()?;
        let final_artifact_id = if let Some(text) = final_output {
            Some(Self::put_artifact(&conn, run_id, "run.final_output", text)?)
        } else {
            None
        };

        conn.execute(
            "UPDATE runs
             SET status = ?2,
                 finished_at = strftime('%Y-%m-%dT%H:%M:%fZ','now'),
                 final_artifact_id = ?3,
                 error_text = ?4
             WHERE run_id = ?1",
            params![run_id, status, final_artifact_id, error_text],
        )?;
        Ok(())
    }

    pub fn record_event(&self, event: EventRecord) -> Result<()> {
        let Some((run_id, ctx_iteration, seq)) = next_context_seq() else {
            return Ok(());
        };

        let iteration = event.iteration_override.unwrap_or(ctx_iteration);
        let conn = self.connect()?;
        let request_artifact_id = match event.request_payload {
            Some(payload) => Some(Self::put_artifact(
                &conn,
                &run_id,
                "event.request",
                &payload,
            )?),
            None => None,
        };
        let response_artifact_id = match event.response_payload {
            Some(payload) => Some(Self::put_artifact(
                &conn,
                &run_id,
                "event.response",
                &payload,
            )?),
            None => None,
        };
        let error_artifact_id = match event.error_payload {
            Some(payload) => Some(Self::put_artifact(&conn, &run_id, "event.error", &payload)?),
            None => None,
        };

        conn.execute(
            "INSERT INTO events (
                run_id, seq, iteration, component, kind, status, message,
                call_no, attempt, latency_ms, cache_hit, idempotency_key,
                request_artifact_id, response_artifact_id, error_artifact_id, extra_json
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16)",
            params![
                run_id,
                seq as i64,
                iteration as i64,
                event.component,
                event.kind,
                event.status,
                event.message,
                event.call_no.map(|v| v as i64),
                event.attempt.map(|v| v as i64),
                event.latency_ms.map(|v| v as i64),
                event.cache_hit.map(|v| if v { 1i64 } else { 0i64 }),
                event.idempotency_key,
                request_artifact_id,
                response_artifact_id,
                error_artifact_id,
                event.extra_json,
            ],
        )?;
        Ok(())
    }

    pub fn list_runs(&self, limit: usize) -> Result<Vec<RunRow>> {
        let conn = self.connect()?;
        let mut stmt = conn.prepare(
            "SELECT run_id, started_at, finished_at, status, mode, path, question, task, generator, model, config_fingerprint, error_text
             FROM runs
             ORDER BY started_at DESC
             LIMIT ?1",
        )?;
        let rows = stmt
            .query_map(params![limit as i64], |row| {
                Ok(RunRow {
                    run_id: row.get(0)?,
                    started_at: row.get(1)?,
                    finished_at: row.get(2)?,
                    status: row.get(3)?,
                    mode: row.get(4)?,
                    path: row.get(5)?,
                    question: row.get(6)?,
                    task: row.get(7)?,
                    generator: row.get(8)?,
                    model: row.get(9)?,
                    config_fingerprint: row.get(10)?,
                    error_text: row.get(11)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    pub fn list_events(&self, run_id: &str, limit: usize) -> Result<Vec<EventRow>> {
        let conn = self.connect()?;
        let mut stmt = conn.prepare(
            "SELECT id, run_id, seq, ts, iteration, component, kind, status, message,
                    call_no, attempt, latency_ms, cache_hit, idempotency_key,
                    request_artifact_id, response_artifact_id, error_artifact_id, extra_json
             FROM events
             WHERE run_id = ?1
             ORDER BY seq ASC
             LIMIT ?2",
        )?;
        let rows = stmt
            .query_map(params![run_id, limit as i64], |row| {
                Ok(EventRow {
                    id: row.get(0)?,
                    run_id: row.get(1)?,
                    seq: row.get::<_, i64>(2)? as u64,
                    ts: row.get(3)?,
                    iteration: row.get::<_, i64>(4)? as usize,
                    component: row.get(5)?,
                    kind: row.get(6)?,
                    status: row.get(7)?,
                    message: row.get(8)?,
                    call_no: row.get::<_, Option<i64>>(9)?.map(|v| v as u64),
                    attempt: row.get::<_, Option<i64>>(10)?.map(|v| v as usize),
                    latency_ms: row.get::<_, Option<i64>>(11)?.map(|v| v as u64),
                    cache_hit: row.get::<_, Option<i64>>(12)?.map(|v| v != 0),
                    idempotency_key: row.get(13)?,
                    request_artifact_id: row.get(14)?,
                    response_artifact_id: row.get(15)?,
                    error_artifact_id: row.get(16)?,
                    extra_json: row.get(17)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    pub fn get_artifact(&self, id: i64) -> Result<Option<ArtifactRow>> {
        let conn = self.connect()?;
        conn.query_row(
            "SELECT id, run_id, kind, content, created_at
             FROM artifacts
             WHERE id = ?1",
            params![id],
            |row| {
                Ok(ArtifactRow {
                    id: row.get(0)?,
                    run_id: row.get(1)?,
                    kind: row.get(2)?,
                    content: row.get(3)?,
                    created_at: row.get(4)?,
                })
            },
        )
        .optional()
        .context("failed to fetch artifact")
    }
}

pub fn initialize_global() -> Option<Arc<HistoryStore>> {
    HistoryStore::initialize_global()
}

pub fn global() -> Option<Arc<HistoryStore>> {
    HistoryStore::global()
}

pub fn set_iteration(iteration: usize) {
    HistoryStore::set_iteration(iteration);
}

pub fn clear_active_context() {
    HistoryStore::clear_active_context();
}

pub fn start_run(payload: &RunStart) -> Option<String> {
    let store = global()?;
    match store.start_run(payload) {
        Ok(run_id) => Some(run_id),
        Err(error) => {
            tracing::warn!(error = %error, "failed to persist run start");
            None
        }
    }
}

pub fn finish_run(
    run_id: &str,
    status: &str,
    final_output: Option<&str>,
    error_text: Option<&str>,
) {
    if let Some(store) = global() {
        if let Err(error) = store.finish_run(run_id, status, final_output, error_text) {
            tracing::warn!(error = %error, run_id, "failed to persist run completion");
        }
    }
}

pub fn record_event(event: EventRecord) {
    if let Some(store) = global() {
        if let Err(error) = store.record_event(event) {
            tracing::warn!(error = %error, "failed to persist history event");
        }
    }
}

fn global_store_dir() -> Result<PathBuf> {
    if let Some(home) = std::env::var_os("HOME") {
        return Ok(PathBuf::from(home).join(STORE_DIR_NAME));
    }
    if let Some(profile) = std::env::var_os("USERPROFILE") {
        return Ok(PathBuf::from(profile).join(STORE_DIR_NAME));
    }
    let cwd = std::env::current_dir().context("failed to resolve current directory")?;
    Ok(cwd.join(STORE_DIR_NAME))
}

fn next_run_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static COUNTER: AtomicU64 = AtomicU64::new(1);
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let mut hasher = blake3::Hasher::new();
    hasher.update(&now.to_le_bytes());
    hasher.update(&(std::process::id() as u64).to_le_bytes());
    hasher.update(&seq.to_le_bytes());
    format!("run_{}", &hasher.finalize().to_hex()[..20])
}
