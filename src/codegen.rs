//! Post-RLM code generation orchestrator.
//!
//! Dispatches to Claude CLI or OpenCode CLI after the RLM analysis
//! produces `.lambda-rlm-result.md`. Both generators use the same
//! prompt format and output contract: read findings, fix code, commit,
//! output a summary line (or "CLEAN" if nothing actionable).

use anyhow::{Context, Result};
use dashmap::DashMap;
use portable_pty::{CommandBuilder, NativePtySystem, PtySize, PtySystem};
use regex::Regex;
use std::io::IsTerminal;
use std::io::Read;
use std::io::Write;
use std::time::Instant;
use std::{
    path::Path,
    process::Stdio,
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
    sync::Arc,
    sync::OnceLock,
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::sync::{watch, Semaphore};
use tokio::task::JoinHandle;
use tokio::time::Duration;

use crate::oracle::Oracle;
use crate::resilience::CircuitBreaker;
use crate::types::CodeGenerator;

const RESULT_FILE_NAME: &str = ".lambda-rlm-result.md";
const MAX_RESULT_BYTES: usize = 8 * 1024 * 1024;
const MAX_LOG_FILES_PER_GENERATOR: usize = 10;
const CODEGEN_BUDGET_UNITS: usize = 50;
const RESULT_WRITE_TIMEOUT: Duration = Duration::from_secs(15);
const LOG_WRITE_TIMEOUT: Duration = Duration::from_secs(10);
const MAX_INLINE_ANALYSIS_BYTES: usize = 128 * 1024;
const CLAUDE_CB_THRESHOLD: usize = 3;
const CLAUDE_CB_COOLDOWN: Duration = Duration::from_secs(60);
const OPENCODE_CB_THRESHOLD: usize = 2;
const OPENCODE_CB_COOLDOWN: Duration = Duration::from_secs(120);
const CLAUDE_MAX_CONCURRENT: usize = 4;
const OPENCODE_MAX_CONCURRENT: usize = 2;
const MAX_CODEGEN_BULKHEAD_PERMITS: usize = 64;
const CODEGEN_CACHE_SCHEMA_VERSION: u8 = 1;
const CODEGEN_SINGLE_FLIGHT_TTL: Duration = Duration::from_secs(5 * 60);
const CODEGEN_SINGLE_FLIGHT_SCAVENGE_INTERVAL: Duration = Duration::from_secs(60);
const CODEGEN_SINGLE_FLIGHT_WAIT_TIMEOUT: Duration = Duration::from_secs(30);
const CODEGEN_SINGLE_FLIGHT_MAX_ENTRIES: usize = 1_000;
const INTERACTIVE_STARTUP_TIMEOUT_MIN: Duration = Duration::from_secs(5);
const INTERACTIVE_STARTUP_TIMEOUT_MAX: Duration = Duration::from_secs(30);
const INTERACTIVE_SESSION_TIMEOUT: Duration = Duration::from_secs(30 * 60);
const OPENCODE_INTERACTIVE_GLOBAL_TIMEOUT: Duration = Duration::from_secs(300);
const CHILD_REAP_TIMEOUT: Duration = Duration::from_secs(5);
const CHILD_SHUTDOWN_REAP_TIMEOUT: Duration = Duration::from_millis(100);
const CHILD_DROP_REAP_WAIT_TIMEOUT: Duration = Duration::from_secs(5);
const RESOURCE_ACQUIRE_TIMEOUT: Duration = Duration::from_secs(30);
const INTERACTIVE_STARTUP_PROBE_INTERVAL: Duration = Duration::from_millis(50);
const INTERACTIVE_IO_JOIN_TIMEOUT: Duration = Duration::from_secs(1);
static CODEGEN_BUDGET_GUARD_LIVE_COUNT: AtomicU64 = AtomicU64::new(0);
static CODEGEN_FLIGHT_GUARD_LIVE_COUNT: AtomicU64 = AtomicU64::new(0);
static CHILD_CLEANUP_GUARD_LIVE_COUNT: AtomicU64 = AtomicU64::new(0);
static PTY_SESSION_GUARD_LIVE_COUNT: AtomicU64 = AtomicU64::new(0);
static LIVE_CODEGEN_CHILDREN: OnceLock<DashMap<u32, ()>> = OnceLock::new();
type CodegenFlightResult = Result<Arc<str>, Arc<str>>;
type CodegenFlightSender = Arc<watch::Sender<Option<CodegenFlightResult>>>;
static CODEGEN_SINGLE_FLIGHT: OnceLock<DashMap<String, CodegenFlightEntry>> = OnceLock::new();
static CODEGEN_SINGLE_FLIGHT_LAST_SCAVENGE_SECS: AtomicU64 = AtomicU64::new(0);
static CODEGEN_SINGLE_FLIGHT_SCAVENGE_COUNT: AtomicU64 = AtomicU64::new(0);
const CODEGEN_SINGLE_FLIGHT_SHRINK_EVERY: u64 = 100;
const INTERACTIVE_STDIN_POLL_TIMEOUT: Duration = Duration::from_millis(100);

struct CodegenFlightEntry {
    sender: CodegenFlightSender,
    inserted_at: Instant,
}

pub struct CodegenGuardLiveCounts {
    pub budget: u64,
    pub flight: u64,
    pub child_cleanup: u64,
    pub pty_session: u64,
}

pub fn codegen_guard_live_counts() -> CodegenGuardLiveCounts {
    CodegenGuardLiveCounts {
        budget: CODEGEN_BUDGET_GUARD_LIVE_COUNT.load(Ordering::Acquire),
        flight: CODEGEN_FLIGHT_GUARD_LIVE_COUNT.load(Ordering::Acquire),
        child_cleanup: CHILD_CLEANUP_GUARD_LIVE_COUNT.load(Ordering::Acquire),
        pty_session: PTY_SESSION_GUARD_LIVE_COUNT.load(Ordering::Acquire),
    }
}

#[must_use = "PtySessionGuard tracks interactive PTY lifecycle"]
struct PtySessionGuard;

impl PtySessionGuard {
    fn new() -> Self {
        PTY_SESSION_GUARD_LIVE_COUNT.fetch_add(1, Ordering::AcqRel);
        Self
    }
}

impl Drop for PtySessionGuard {
    fn drop(&mut self) {
        PTY_SESSION_GUARD_LIVE_COUNT.fetch_sub(1, Ordering::AcqRel);
    }
}

async fn join_interactive_io_task(task: JoinHandle<()>, label: &'static str) {
    let mut task = task;
    match tokio::time::timeout(INTERACTIVE_IO_JOIN_TIMEOUT, &mut task).await {
        Ok(Ok(())) => {}
        Ok(Err(join_error)) => {
            tracing::warn!(label, error = ?join_error, "interactive I/O task join failed");
        }
        Err(_) => {
            task.abort();
            tracing::warn!(
                label,
                timeout_ms = INTERACTIVE_IO_JOIN_TIMEOUT.as_millis(),
                "interactive I/O task did not stop before join timeout"
            );
        }
    }
}

fn is_sensitive_env_var_name(name: &str) -> bool {
    static SENSITIVE_ENV_NAME_RE: OnceLock<Regex> = OnceLock::new();
    SENSITIVE_ENV_NAME_RE
        .get_or_init(|| {
            Regex::new(r"(?i)^[a-z_][a-z0-9_]*(?:key|token|secret|api)[a-z0-9_]*$")
                .expect("valid sensitive env-name regex")
        })
        .is_match(name)
}

fn tracked_codegen_children() -> &'static DashMap<u32, ()> {
    LIVE_CODEGEN_CHILDREN.get_or_init(DashMap::new)
}

fn track_codegen_child_pid(pid: Option<u32>) {
    if let Some(pid) = pid {
        tracked_codegen_children().insert(pid, ());
    }
}

fn untrack_codegen_child_pid(pid: Option<u32>) {
    if let Some(pid) = pid {
        let _ = tracked_codegen_children().remove(&pid);
    }
}

#[cfg(unix)]
fn kill_process_group_sigkill(pid: Option<u32>) {
    if let Some(pid) = pid {
        if let Ok(pgid) = i32::try_from(pid) {
            let target = format!("-{pgid}");
            let _ = std::process::Command::new("kill")
                .arg("-KILL")
                .arg("--")
                .arg(target)
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status();
        }
    }
}

#[cfg(unix)]
fn kill_process_group_sigterm(pid: Option<u32>) {
    if let Some(pid) = pid {
        if let Ok(pgid) = i32::try_from(pid) {
            let target = format!("-{pgid}");
            let _ = std::process::Command::new("kill")
                .arg("-TERM")
                .arg("--")
                .arg(target)
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status();
        }
    }
}

pub fn force_kill_tracked_codegen_children() -> usize {
    let pids: Vec<u32> = tracked_codegen_children()
        .iter()
        .map(|entry| *entry.key())
        .collect();

    for pid in &pids {
        #[cfg(unix)]
        {
            kill_process_group_sigkill(Some(*pid));
            let _ = std::process::Command::new("kill")
                .arg("-9")
                .arg(pid.to_string())
                .status();
        }
        untrack_codegen_child_pid(Some(*pid));
    }

    pids.len()
}

#[must_use = "dropping CodegenBudgetGuard without commit() restores budget"]
struct CodegenBudgetGuard<'a> {
    oracle: &'a Oracle,
    units: usize,
    committed: bool,
}

impl<'a> CodegenBudgetGuard<'a> {
    fn acquire(oracle: &'a Oracle, units: usize) -> Result<Self> {
        anyhow::ensure!(
            oracle.budget_try_reserve(units),
            "code generation budget exhausted"
        );
        CODEGEN_BUDGET_GUARD_LIVE_COUNT.fetch_add(1, Ordering::AcqRel);
        Ok(Self {
            oracle,
            units,
            committed: false,
        })
    }

    fn commit(&mut self) {
        self.committed = true;
    }
}

impl Drop for CodegenBudgetGuard<'_> {
    fn drop(&mut self) {
        CODEGEN_BUDGET_GUARD_LIVE_COUNT.fetch_sub(1, Ordering::AcqRel);
        if !self.committed {
            if !std::thread::panicking() {
                tracing::error!(
                    units = self.units,
                    "CodegenBudgetGuard dropped without commit() — restoring reserved budget"
                );
            }
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.oracle.budget_unreserve(self.units);
            }));
        }
    }
}

fn codegen_circuit(
    generator: &CodeGenerator,
    interactive: bool,
) -> Result<&'static CircuitBreaker> {
    static CLAUDE_CIRCUIT: OnceLock<CircuitBreaker> = OnceLock::new();
    static CLAUDE_INTERACTIVE_CIRCUIT: OnceLock<CircuitBreaker> = OnceLock::new();
    // OpenCode is still less stable under sustained load in our testing, so it
    // uses a stricter threshold/longer cooldown to isolate faults faster.
    static OPENCODE_CIRCUIT: OnceLock<CircuitBreaker> = OnceLock::new();
    static OPENCODE_INTERACTIVE_CIRCUIT: OnceLock<CircuitBreaker> = OnceLock::new();
    let circuit = match (generator, interactive) {
        (CodeGenerator::Claude, false) => CLAUDE_CIRCUIT
            .get_or_init(|| CircuitBreaker::new(CLAUDE_CB_THRESHOLD, CLAUDE_CB_COOLDOWN)),
        (CodeGenerator::Claude, true) => CLAUDE_INTERACTIVE_CIRCUIT
            .get_or_init(|| CircuitBreaker::new(CLAUDE_CB_THRESHOLD, CLAUDE_CB_COOLDOWN)),
        (CodeGenerator::Opencode, false) => OPENCODE_CIRCUIT
            .get_or_init(|| CircuitBreaker::new(OPENCODE_CB_THRESHOLD, OPENCODE_CB_COOLDOWN)),
        (CodeGenerator::Opencode, true) => OPENCODE_INTERACTIVE_CIRCUIT
            .get_or_init(|| CircuitBreaker::new(OPENCODE_CB_THRESHOLD, OPENCODE_CB_COOLDOWN)),
    };
    Ok(circuit)
}

/// Build the fix-loop prompt shared by all code generators.
///
/// PRE: question is non-empty, iteration >= 1
/// POST: prompt instructs the generator to read .lambda-rlm-result.md and act
fn build_prompt(question: &str, iteration: usize) -> String {
    format!(
        "Read the analysis in .lambda-rlm-result.md — it's iteration {iteration} of a λ-RLM \
         fix loop for the question: \"{question}\".\n\n\
         Your job:\n\
         1. Read the findings carefully.\n\
         2. Act on every actionable item — fix bugs, refactor code, add missing pieces.\n\
         3. When done, output a single line summary of what you changed.\n\
         4. Do a proper git commit(signed commit preferred)\n\
         5. Update readme with commit id and change-log\n\
         5. If there is nothing actionable (only informational notes or the analysis is clean), \
            output exactly: CLEAN\n\n\
         Start by reading .lambda-rlm-result.md now.",
    )
}

fn build_prompt_with_inline_result(question: &str, iteration: usize, result: &str) -> String {
    let mut inline = result.to_owned();
    if inline.len() > MAX_INLINE_ANALYSIS_BYTES {
        let mut end = MAX_INLINE_ANALYSIS_BYTES;
        while end > 0 && !inline.is_char_boundary(end) {
            end -= 1;
        }
        inline.truncate(end);
        inline.push_str("\n\n[TRUNCATED: source analysis clipped due to storage constraints]");
    }

    format!(
        "Read this analysis directly (result file unavailable due to disk constraints) — it's iteration {iteration} of a λ-RLM \
         fix loop for the question: \"{question}\".\n\n\
         --- BEGIN ANALYSIS ---\n{inline}\n--- END ANALYSIS ---\n\n\
         Your job:\n\
         1. Read the findings carefully.\n\
         2. Act on every actionable item — fix bugs, refactor code, add missing pieces.\n\
         3. When done, output a single line summary of what you changed.\n\
         4. Do a proper git commit(signed commit preferred)\n\
         5. Update readme with commit id and change-log\n\
         5. If there is nothing actionable (only informational notes or the analysis is clean), \
            output exactly: CLEAN"
    )
}

/// Dispatch to the configured code generator.
///
/// PRE: work_dir exists and is writable
/// POST: generator process has exited, returns summary line
pub async fn run_code_generator(
    oracle: &Oracle,
    generator: &CodeGenerator,
    result: &str,
    work_dir: &Path,
    question: &str,
    iteration: usize,
    interactive: bool,
    interactive_timeout: Duration,
    shutdown: watch::Receiver<bool>,
) -> Result<Arc<str>> {
    validate_codegen_work_dir(work_dir)?;
    if !single_flight_enabled(interactive) {
        return run_code_generator_once(
            oracle,
            generator,
            result,
            work_dir,
            question,
            iteration,
            interactive,
            interactive_timeout,
            shutdown,
        )
        .await;
    }

    maybe_scavenge_codegen_single_flight();
    let flight_key = codegen_flight_key(
        generator,
        work_dir,
        question,
        iteration,
        result,
        interactive,
    );
    let flights = CODEGEN_SINGLE_FLIGHT.get_or_init(DashMap::new);
    let _ = evict_stale_codegen_flights(flights);

    loop {
        match flights.entry(flight_key.clone()) {
            dashmap::mapref::entry::Entry::Occupied(entry) => {
                if entry.get().inserted_at.elapsed() > CODEGEN_SINGLE_FLIGHT_TTL {
                    let _ = flights.remove(&flight_key);
                    continue;
                }
                let mut rx = entry.get().sender.subscribe();
                let existing = rx.borrow().clone();
                drop(entry);

                if let Some(outcome) = existing {
                    return flight_outcome_to_result(outcome, generator);
                }

                match tokio::time::timeout(CODEGEN_SINGLE_FLIGHT_WAIT_TIMEOUT, rx.changed()).await {
                    Ok(Ok(())) => {}
                    Ok(Err(_)) | Err(_) => {
                        let _ = flights.remove(&flight_key);
                        continue;
                    }
                }

                let resolved = { rx.borrow().clone() };
                if let Some(outcome) = resolved {
                    return flight_outcome_to_result(outcome, generator);
                }
                let _ = flights.remove(&flight_key);
                continue;
            }
            dashmap::mapref::entry::Entry::Vacant(entry) => {
                let (tx, _rx) = watch::channel::<Option<CodegenFlightResult>>(None);
                let tx: CodegenFlightSender = Arc::new(tx);
                entry.insert(CodegenFlightEntry {
                    sender: Arc::clone(&tx),
                    inserted_at: Instant::now(),
                });
                if flights.len() > CODEGEN_SINGLE_FLIGHT_MAX_ENTRIES {
                    let removed = evict_stale_codegen_flights(flights);
                    if removed > 0 {
                        tracing::warn!(
                            removed,
                            size = flights.len(),
                            threshold = CODEGEN_SINGLE_FLIGHT_MAX_ENTRIES,
                            "evicted stale single-flight entries on pressure"
                        );
                    }
                }

                let _flight_guard = CodegenFlightGuard::new(flights, flight_key.clone());

                let outcome = run_code_generator_once(
                    oracle,
                    generator,
                    result,
                    work_dir,
                    question,
                    iteration,
                    interactive,
                    interactive_timeout,
                    shutdown.clone(),
                )
                .await
                .map_err(|error| {
                    tracing::error!(generator = %generator, iteration, error = ?error, "single-flight leader failed");
                    Arc::<str>::from("code generation unavailable")
                });

                let _ = tx.send(Some(outcome.clone()));
                return flight_outcome_to_result(outcome, generator);
            }
        }
    }
}

#[inline]
fn single_flight_enabled(interactive: bool) -> bool {
    !interactive
}

fn validate_codegen_work_dir(work_dir: &Path) -> Result<()> {
    let meta = std::fs::symlink_metadata(work_dir)
        .with_context(|| format!("failed to inspect work directory {}", work_dir.display()))?;
    anyhow::ensure!(meta.is_dir(), "work directory is not a directory");
    anyhow::ensure!(
        !meta.file_type().is_symlink(),
        "refusing symlink work directory"
    );

    #[cfg(unix)]
    {
        use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
        let _ = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(work_dir)
            .with_context(|| format!("refusing unsafe work directory {}", work_dir.display()))?;

        if let Ok(tmp_meta) = std::fs::symlink_metadata("/tmp") {
            anyhow::ensure!(
                meta.dev() == tmp_meta.dev(),
                "work directory must be on the same filesystem as /tmp for atomic writes"
            );
        }
    }

    Ok(())
}

async fn run_code_generator_once(
    oracle: &Oracle,
    generator: &CodeGenerator,
    result: &str,
    work_dir: &Path,
    question: &str,
    iteration: usize,
    interactive: bool,
    interactive_timeout: Duration,
    shutdown: watch::Receiver<bool>,
) -> Result<Arc<str>> {
    let cache_key = codegen_cache_key(generator, result, interactive);
    if !interactive {
        if let Some(cached) = oracle
            .cache()
            .get_validated(&cache_key, |value| validate_codegen_summary(value).is_ok())
        {
            tracing::debug!(generator = %generator, iteration, "codegen replay cache hit");
            return Ok(Arc::from(cached));
        }
    }

    let circuit = codegen_circuit(generator, interactive)?;
    anyhow::ensure!(
        circuit.allow_request(),
        "code generation temporarily unavailable"
    );

    if interactive {
        let interactive_timeout = clamp_interactive_startup_timeout(interactive_timeout);
        let started = Instant::now();
        let run = tokio::time::timeout(INTERACTIVE_SESSION_TIMEOUT, async {
            match generator {
                CodeGenerator::Claude => {
                    run_claude_interactive(
                        result,
                        work_dir,
                        question,
                        iteration,
                        interactive_timeout,
                        INTERACTIVE_SESSION_TIMEOUT,
                        shutdown,
                    )
                    .await
                }
                CodeGenerator::Opencode => tokio::time::timeout(
                    OPENCODE_INTERACTIVE_GLOBAL_TIMEOUT,
                    run_opencode_interactive(
                        result,
                        work_dir,
                        question,
                        iteration,
                        interactive_timeout,
                        INTERACTIVE_SESSION_TIMEOUT,
                        shutdown,
                    ),
                )
                .await
                .map_err(|_| {
                    anyhow::anyhow!(
                        "interactive opencode global timeout ({}s)",
                        OPENCODE_INTERACTIVE_GLOBAL_TIMEOUT.as_secs()
                    )
                    .context(CodegenErrorMarker(CodegenErrorKind::Retryable))
                })?,
            }
        })
        .await
        .map_err(|_| {
            anyhow::anyhow!("interactive session timeout")
                .context(CodegenErrorMarker(CodegenErrorKind::Retryable))
        })?;
        oracle.record_codegen_call(generator, started.elapsed());

        return match run {
            Ok(summary) => {
                circuit.record_success();
                Ok(summary)
            }
            Err(error) => {
                if error_has_context(&error, "codegen_fatal") {
                    circuit.record_failure();
                    circuit.record_failure();
                    tracing::error!(generator = %generator, iteration, error = ?error, "code generation fatal failure");
                } else if error_has_context(&error, "codegen_retryable") {
                    circuit.record_failure();
                    tracing::warn!(generator = %generator, iteration, error = ?error, "code generation retryable failure");
                } else {
                    circuit.record_failure();
                    tracing::error!(generator = %generator, iteration, error = ?error, "code generation execution failed");
                }
                Err(
                    anyhow::anyhow!("code generation unavailable").context(match generator {
                        CodeGenerator::Claude => "claude_service_unavailable",
                        CodeGenerator::Opencode => "opencode_service_unavailable",
                    }),
                )
            }
        };
    }

    let mut budget_guard = CodegenBudgetGuard::acquire(oracle, CODEGEN_BUDGET_UNITS)?;
    let _bulkhead = tokio::time::timeout(
        RESOURCE_ACQUIRE_TIMEOUT,
        codegen_bulkhead(generator).acquire_owned(),
    )
    .await
    .with_context(|| format!("{generator} bulkhead acquisition timed out"))?
    .with_context(|| format!("{generator} bulkhead saturated"))?;
    let started = Instant::now();
    let run = match generator {
        CodeGenerator::Claude => run_claude(result, work_dir, question, iteration).await,
        CodeGenerator::Opencode => run_opencode(result, work_dir, question, iteration).await,
    };
    oracle.record_codegen_call(generator, started.elapsed());
    budget_guard.commit();

    match run {
        Ok(summary) => {
            if !interactive {
                if let Err(error) = oracle.cache().put(&cache_key, summary.as_ref()) {
                    tracing::warn!(generator = %generator, iteration, error = %error, "failed to persist codegen replay cache entry");
                }
            }
            circuit.record_success();
            Ok(summary)
        }
        Err(error) => {
            if error_has_context(&error, "codegen_fatal") {
                circuit.record_failure();
                circuit.record_failure();
                tracing::error!(generator = %generator, iteration, error = ?error, "code generation fatal failure");
            } else if error_has_context(&error, "codegen_retryable") {
                circuit.record_failure();
                tracing::warn!(generator = %generator, iteration, error = ?error, "code generation retryable failure");
            } else {
                circuit.record_failure();
                tracing::error!(generator = %generator, iteration, error = ?error, "code generation execution failed");
            }
            Err(
                anyhow::anyhow!("code generation unavailable").context(match generator {
                    CodeGenerator::Claude => "claude_service_unavailable",
                    CodeGenerator::Opencode => "opencode_service_unavailable",
                }),
            )
        }
    }
}

fn clamp_interactive_startup_timeout(timeout: Duration) -> Duration {
    timeout.clamp(
        INTERACTIVE_STARTUP_TIMEOUT_MIN,
        INTERACTIVE_STARTUP_TIMEOUT_MAX,
    )
}

#[must_use = "CodegenFlightGuard must stay alive to release single-flight ownership"]
struct CodegenFlightGuard<'a> {
    flights: &'a DashMap<String, CodegenFlightEntry>,
    key: String,
}

impl<'a> CodegenFlightGuard<'a> {
    fn new(flights: &'a DashMap<String, CodegenFlightEntry>, key: String) -> Self {
        CODEGEN_FLIGHT_GUARD_LIVE_COUNT.fetch_add(1, Ordering::AcqRel);
        Self { flights, key }
    }
}

impl Drop for CodegenFlightGuard<'_> {
    fn drop(&mut self) {
        CODEGEN_FLIGHT_GUARD_LIVE_COUNT.fetch_sub(1, Ordering::AcqRel);
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = self.flights.remove(&self.key);
        }));
    }
}

fn codegen_flight_key(
    generator: &CodeGenerator,
    work_dir: &Path,
    question: &str,
    iteration: usize,
    result: &str,
    interactive: bool,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(generator.to_string().as_bytes());
    hasher.update(work_dir.as_os_str().to_string_lossy().as_bytes());
    hasher.update(question.as_bytes());
    hasher.update(&iteration.to_le_bytes());
    hasher.update(&[interactive as u8]);
    hasher.update(result.as_bytes());
    format!("{}:{}", generator, hasher.finalize().to_hex())
}

fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

fn evict_stale_codegen_flights(flights: &DashMap<String, CodegenFlightEntry>) -> usize {
    let mut removed = 0usize;
    flights.retain(|_key, entry| {
        let keep = entry.inserted_at.elapsed() <= CODEGEN_SINGLE_FLIGHT_TTL;
        if !keep {
            removed += 1;
        }
        keep
    });
    removed
}

fn maybe_scavenge_codegen_single_flight() {
    let now = now_unix_secs();
    let last = CODEGEN_SINGLE_FLIGHT_LAST_SCAVENGE_SECS.load(Ordering::Acquire);
    if now.saturating_sub(last) < CODEGEN_SINGLE_FLIGHT_SCAVENGE_INTERVAL.as_secs() {
        return;
    }
    if CODEGEN_SINGLE_FLIGHT_LAST_SCAVENGE_SECS
        .compare_exchange(last, now, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return;
    }

    let flights = CODEGEN_SINGLE_FLIGHT.get_or_init(DashMap::new);
    let removed = evict_stale_codegen_flights(flights);
    if removed > 0 {
        tracing::warn!(removed, "evicted stale codegen single-flight entries");
    }
    let scavenge_count = CODEGEN_SINGLE_FLIGHT_SCAVENGE_COUNT.fetch_add(1, Ordering::AcqRel) + 1;
    if scavenge_count % CODEGEN_SINGLE_FLIGHT_SHRINK_EVERY == 0 {
        flights.shrink_to_fit();
    }
}

pub fn run_codegen_maintenance() {
    maybe_scavenge_codegen_single_flight();
}

fn flight_outcome_to_result(
    outcome: CodegenFlightResult,
    generator: &CodeGenerator,
) -> Result<Arc<str>> {
    match outcome {
        Ok(summary) => Ok(summary),
        Err(_) => Err(
            anyhow::anyhow!("code generation unavailable").context(match generator {
                CodeGenerator::Claude => "claude_service_unavailable",
                CodeGenerator::Opencode => "opencode_service_unavailable",
            }),
        ),
    }
}

fn error_has_context(error: &anyhow::Error, marker: &str) -> bool {
    match marker {
        "codegen_fatal" => {
            codegen_error_kind(error).is_some_and(|kind| kind == CodegenErrorKind::Fatal)
        }
        "codegen_retryable" => {
            codegen_error_kind(error).is_some_and(|kind| kind == CodegenErrorKind::Retryable)
        }
        _ => false,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CodegenErrorKind {
    Retryable,
    Fatal,
}

#[derive(Debug)]
struct CodegenErrorMarker(CodegenErrorKind);

impl std::fmt::Display for CodegenErrorMarker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("code generation failed")
    }
}

impl std::error::Error for CodegenErrorMarker {}

fn codegen_error_kind(error: &anyhow::Error) -> Option<CodegenErrorKind> {
    if let Some(marker) = error.downcast_ref::<CodegenErrorMarker>() {
        return Some(marker.0);
    }

    for cause in error.chain() {
        if let Some(marker) = cause.downcast_ref::<CodegenErrorMarker>() {
            return Some(marker.0);
        }
    }
    None
}

fn codegen_bulkhead(generator: &CodeGenerator) -> Arc<Semaphore> {
    static CLAUDE_BULKHEAD: OnceLock<Arc<Semaphore>> = OnceLock::new();
    static OPENCODE_BULKHEAD: OnceLock<Arc<Semaphore>> = OnceLock::new();

    match generator {
        CodeGenerator::Claude => Arc::clone(CLAUDE_BULKHEAD.get_or_init(|| {
            Arc::new(Semaphore::new(
                CLAUDE_MAX_CONCURRENT.clamp(1, MAX_CODEGEN_BULKHEAD_PERMITS),
            ))
        })),
        CodeGenerator::Opencode => Arc::clone(OPENCODE_BULKHEAD.get_or_init(|| {
            Arc::new(Semaphore::new(
                OPENCODE_MAX_CONCURRENT.clamp(1, MAX_CODEGEN_BULKHEAD_PERMITS),
            ))
        })),
    }
}

fn codegen_cache_key(generator: &CodeGenerator, result: &str, interactive: bool) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&[CODEGEN_CACHE_SCHEMA_VERSION]);
    hasher.update(b"codegen\x00");
    hasher.update(generator.to_string().as_bytes());
    hasher.update(b"\x00");
    hasher.update(env!("CARGO_PKG_VERSION").as_bytes());
    hasher.update(b"\x00");
    hasher.update(&[interactive as u8]);
    hasher.update(b"\x00");
    hasher.update(result.as_bytes());
    format!("b3-codegen_{}", hasher.finalize().to_hex())
}

fn validate_codegen_summary(summary: &str) -> Result<()> {
    anyhow::ensure!(!summary.trim().is_empty(), "codegen summary is empty");
    anyhow::ensure!(
        summary.len() <= MAX_RESULT_BYTES,
        "codegen summary exceeds {} bytes",
        MAX_RESULT_BYTES
    );
    anyhow::ensure!(
        !summary
            .bytes()
            .any(|b| b < 0x20 && b != b'\n' && b != b'\t' && b != b'\r'),
        "codegen summary contains control characters"
    );
    Ok(())
}

fn validate_analysis_result_content(result: &str) -> Result<()> {
    anyhow::ensure!(!result.trim().is_empty(), "result is empty");
    anyhow::ensure!(
        result.len() <= MAX_RESULT_BYTES,
        "result exceeds {} bytes",
        MAX_RESULT_BYTES
    );
    anyhow::ensure!(
        !result.as_bytes().contains(&0),
        "result contains invalid NUL byte"
    );
    Ok(())
}

fn validate_generator_output(generator: &CodeGenerator, output: &str) -> Result<()> {
    anyhow::ensure!(!output.trim().is_empty(), "{generator} output is empty");
    anyhow::ensure!(
        output.len() <= MAX_RESULT_BYTES,
        "{generator} output exceeds {} bytes",
        MAX_RESULT_BYTES
    );
    anyhow::ensure!(
        !output.as_bytes().contains(&0),
        "{generator} output contains invalid NUL byte"
    );
    Ok(())
}

#[cfg(unix)]
fn open_file_no_follow(path: &Path) -> Result<(std::fs::File, u64, u64, u64)> {
    use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
    let file = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(path)
        .with_context(|| format!("refusing unsafe result path: {}", path.display()))?;
    let meta = file.metadata()?;
    Ok((file, meta.len(), meta.dev(), meta.ino()))
}

#[cfg(windows)]
fn open_file_no_follow(path: &Path) -> Result<(std::fs::File, u64, u64, u64)> {
    use std::os::windows::fs::OpenOptionsExt;
    const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;
    const FILE_FLAG_BACKUP_SEMANTICS: u32 = 0x0200_0000;

    let file = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS)
        .open(path)
        .with_context(|| format!("refusing unsafe result path: {}", path.display()))?;
    let meta = file.metadata()?;
    anyhow::ensure!(
        !meta.file_type().is_symlink(),
        "refusing symlink result file"
    );
    Ok((file, meta.len(), 0, 0))
}

#[cfg(all(not(unix), not(windows)))]
fn open_file_no_follow(path: &Path) -> Result<(std::fs::File, u64, u64, u64)> {
    let meta = std::fs::symlink_metadata(path)
        .with_context(|| format!("failed to inspect result path: {}", path.display()))?;
    anyhow::ensure!(
        !meta.file_type().is_symlink(),
        "refusing symlink result file"
    );
    let file = std::fs::File::open(path)?;
    Ok((file, meta.len(), 0, 0))
}

async fn write_result_file_atomically(
    work_dir: &Path,
    result: &str,
    _iteration: usize,
) -> Result<()> {
    validate_analysis_result_content(result)?;

    let result_file = work_dir.join(RESULT_FILE_NAME);
    match std::fs::symlink_metadata(&result_file) {
        Ok(meta) => {
            anyhow::ensure!(
                !meta.file_type().is_symlink(),
                "refusing symlink result file"
            );
            let (_existing, size, _dev, _ino) = open_file_no_follow(&result_file)?;
            anyhow::ensure!(
                size <= MAX_RESULT_BYTES as u64,
                "existing result file exceeds safety bound"
            );
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(error).with_context(|| {
                format!("failed to inspect result target {}", result_file.display())
            });
        }
    }

    let work_dir = work_dir.to_path_buf();
    let result_file_clone = result_file.clone();
    let result_owned = result.to_owned();
    tokio::time::timeout(
        RESULT_WRITE_TIMEOUT,
        tokio::task::spawn_blocking(move || -> Result<()> {
            let mut tmp = tempfile::NamedTempFile::new_in(&work_dir).with_context(|| {
                format!(
                    "Failed to create temporary result file in {}",
                    work_dir.display()
                )
            })?;
            tmp.write_all(result_owned.as_bytes()).with_context(|| {
                format!(
                    "Failed to write temporary result file in {}",
                    work_dir.display()
                )
            })?;
            tmp.as_file_mut()
                .sync_all()
                .context("Failed to fsync temporary result file")?;
            let persisted = tmp.persist(&result_file_clone).map_err(|e| {
                anyhow::anyhow!(
                    "Failed to move {}: {}",
                    result_file_clone.display(),
                    e.error
                )
            })?;
            persisted
                .sync_all()
                .context("Failed to fsync persisted result file")?;
            let dir = std::fs::File::open(&work_dir)
                .with_context(|| format!("Failed to open {} for fsync", work_dir.display()))?;
            dir.sync_all().context("Failed to fsync result directory")?;
            Ok(())
        }),
    )
    .await
    .context("Result file writer task timed out")?
    .context("Result file writer task failed")??;

    Ok(())
}

fn is_storage_full_error(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        if let Some(io_err) = cause.downcast_ref::<std::io::Error>() {
            return io_err.kind() == std::io::ErrorKind::StorageFull;
        }
        let lower = cause.to_string().to_ascii_lowercase();
        lower.contains("no space left") || lower.contains("disk full")
    })
}

async fn write_log_file_atomically(log_file: &Path, content: &str) -> Result<()> {
    let log_path = log_file.to_path_buf();
    let parent = log_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("log file has no parent"))?
        .to_path_buf();
    let content = content.to_owned();

    tokio::time::timeout(
        LOG_WRITE_TIMEOUT,
        tokio::task::spawn_blocking(move || -> Result<()> {
            if log_path.exists() {
                let _ = open_file_no_follow(&log_path)?;
            }
            let mut tmp = tempfile::NamedTempFile::new_in(&parent)?;
            tmp.write_all(content.as_bytes())?;
            tmp.as_file_mut().sync_all()?;
            let persisted = tmp.persist(&log_path).map_err(|e| {
                anyhow::anyhow!("Failed to move {}: {}", log_path.display(), e.error)
            })?;
            persisted.sync_all()?;
            if let Ok(dir) = std::fs::File::open(&parent) {
                let _ = dir.sync_all();
            }
            Ok(())
        }),
    )
    .await
    .context("log file writer task timed out")?
    .context("log file writer task failed")??;

    Ok(())
}

async fn remove_result_file(work_dir: &Path) {
    let result_file = work_dir.join(RESULT_FILE_NAME);
    if let Err(error) = tokio::fs::remove_file(&result_file).await {
        if error.kind() != std::io::ErrorKind::NotFound {
            tracing::warn!(path = %result_file.display(), error = %error, "failed to remove result file");
        }
    }
}

fn first_non_empty_line(text: &str) -> Option<&str> {
    text.lines().find(|line| !line.trim().is_empty())
}

fn last_non_empty_line(text: &str) -> Option<&str> {
    text.lines().rev().find(|line| !line.trim().is_empty())
}

fn is_opencode_preamble_line(line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return true;
    }
    if trimmed.starts_with("```")
        || trimmed.starts_with('#')
        || trimmed.starts_with('[')
        || trimmed.starts_with('>')
    {
        return true;
    }

    let lower = trimmed.to_ascii_lowercase();
    lower.starts_with("opencode")
        || lower.starts_with("model:")
        || lower.starts_with("provider:")
        || lower.starts_with("tokens:")
        || lower.starts_with("cost:")
        || lower.starts_with("duration:")
        || lower.starts_with("status:")
}

fn extract_opencode_summary_line(text: &str) -> Option<&str> {
    if text
        .bytes()
        .any(|byte| byte < 0x20 && byte != b'\n' && byte != b'\t' && byte != b'\r')
    {
        return None;
    }
    text.lines()
        .find(|line| !is_opencode_preamble_line(line))
        .or_else(|| extract_opencode_summary_line_regex(text))
        .or_else(|| first_non_empty_line(text))
}

fn extract_opencode_summary_line_regex(text: &str) -> Option<&str> {
    static FALLBACK_RE: OnceLock<Regex> = OnceLock::new();
    let re = FALLBACK_RE.get_or_init(|| {
        Regex::new(r"(?m)^\s*(?:>>>\s*)?(?P<summary>CLEAN|(?:[A-Za-z0-9][^\n]{8,240}))\s*$")
            .expect("fallback regex must compile")
    });
    re.captures_iter(text).find_map(|captures| {
        let candidate = captures.name("summary")?.as_str().trim();
        if candidate.is_empty() || is_opencode_preamble_line(candidate) {
            return None;
        }
        Some(candidate)
    })
}

pub fn validate_opencode_binary_version() -> Result<()> {
    let output = std::process::Command::new("opencode")
        .arg("--version")
        .output()
        .context("failed to run `opencode --version`")?;
    anyhow::ensure!(
        output.status.success(),
        "`opencode --version` exited with {}",
        output.status
    );
    let version = String::from_utf8_lossy(&output.stdout);
    let major = parse_opencode_major_version(&version).ok_or_else(|| {
        anyhow::anyhow!(
            "unable to parse OpenCode version output: {}",
            version.trim()
        )
    })?;
    anyhow::ensure!(
        major == 1,
        "OpenCode v1.x required, found: {}",
        version.trim()
    );
    Ok(())
}

fn parse_opencode_major_version(version_output: &str) -> Option<u64> {
    for raw_token in version_output.split_whitespace() {
        let token = raw_token.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '.');
        if token.eq_ignore_ascii_case("opencode") {
            continue;
        }

        let token = token.strip_prefix('v').unwrap_or(token);
        let mut parts = token.split('.');
        let major = parts.next()?;
        let minor = parts.next();
        if minor.is_none() {
            continue;
        }
        if !major.chars().all(|c| c.is_ascii_digit()) {
            continue;
        }

        return major.parse::<u64>().ok();
    }
    None
}

impl CodeGenerator {
    fn extract_result(&self, output: &[u8]) -> Result<Arc<str>> {
        let stdout = std::str::from_utf8(output)
            .with_context(|| format!("{self} subprocess produced invalid UTF-8 on stdout"))?;
        if stdout.trim().is_empty() {
            tracing::warn!(generator = %self, "generator completed with empty stdout; treating as degraded");
            return Ok(Arc::from("(degraded: empty generator output)"));
        }
        validate_generator_output(self, &stdout)?;

        let summary = match self {
            CodeGenerator::Claude => last_non_empty_line(&stdout),
            CodeGenerator::Opencode => extract_opencode_summary_line(&stdout),
        };

        if let Some(summary) = summary {
            return Ok(Arc::from(summary));
        }

        if matches!(self, CodeGenerator::Opencode) {
            tracing::warn!("opencode output summary line missing; degrading output contract");
            return Ok(Arc::from("(degraded: unparseable opencode output)"));
        }

        Err(anyhow::anyhow!(
            "{self} output did not contain a non-empty summary line"
        ))
    }
}

fn cleanup_old_logs(work_dir: &Path, generator: &CodeGenerator) {
    let mut entries: Vec<(SystemTime, std::path::PathBuf)> = Vec::new();
    let prefix = match generator {
        CodeGenerator::Claude => ".lambda-rlm-claude-",
        CodeGenerator::Opencode => ".lambda-rlm-opencode-",
    };

    let read_dir = match std::fs::read_dir(work_dir) {
        Ok(read_dir) => read_dir,
        Err(error) => {
            tracing::warn!(path = %work_dir.display(), error = %error, "failed to scan log directory");
            return;
        }
    };

    for entry in read_dir.filter_map(|entry| entry.ok()) {
        let file_name = entry.file_name();
        let file_name = file_name.to_string_lossy();
        if !file_name.starts_with(prefix) || !file_name.ends_with(".log") {
            continue;
        }

        match entry.metadata().and_then(|meta| meta.modified()) {
            Ok(mtime) => entries.push((mtime, entry.path())),
            Err(error) => {
                tracing::warn!(path = %entry.path().display(), error = %error, "failed to read log metadata")
            }
        }
    }

    if entries.len() <= MAX_LOG_FILES_PER_GENERATOR {
        return;
    }

    entries.sort_by_key(|(mtime, _)| *mtime);
    let to_remove = entries.len() - MAX_LOG_FILES_PER_GENERATOR;
    for (_, path) in entries.into_iter().take(to_remove) {
        if let Err(error) = std::fs::remove_file(&path) {
            tracing::warn!(path = %path.display(), error = %error, "failed to evict old log file");
        }
    }
}

struct GeneratorOutput {
    status: std::process::ExitStatus,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

#[must_use = "ChildCleanup must be kept until process exits or is explicitly reaped"]
struct ChildCleanup {
    child: Option<tokio::process::Child>,
}

impl ChildCleanup {
    fn new(child: tokio::process::Child) -> Self {
        CHILD_CLEANUP_GUARD_LIVE_COUNT.fetch_add(1, Ordering::AcqRel);
        track_codegen_child_pid(child.id());
        Self { child: Some(child) }
    }

    fn child_mut(&mut self) -> Result<&mut tokio::process::Child> {
        self.child
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("child process handle missing"))
    }

    async fn kill_and_reap(&mut self, wait_timeout: Duration) {
        if let Some(child) = self.child.as_mut() {
            let pid = child.id();
            #[cfg(unix)]
            kill_process_group_sigkill(pid);
            let _ = child.start_kill();
            let _ = tokio::time::timeout(wait_timeout, child.wait()).await;
            untrack_codegen_child_pid(pid);
        }
        self.child = None;
    }

    async fn terminate_with_escalation(&mut self, wait_timeout: Duration) {
        if let Some(child) = self.child.as_mut() {
            let pid = child.id();
            #[cfg(unix)]
            kill_process_group_sigterm(pid);
            #[cfg(not(unix))]
            {
                let _ = child.start_kill();
            }

            let exited_cleanly = matches!(
                tokio::time::timeout(wait_timeout, child.wait()).await,
                Ok(Ok(_))
            );
            if !exited_cleanly {
                #[cfg(unix)]
                kill_process_group_sigkill(pid);
                let _ = child.start_kill();
                let _ = tokio::time::timeout(wait_timeout, child.wait()).await;
            }
            untrack_codegen_child_pid(pid);
        }
        self.child = None;
    }

    fn disarm(&mut self) {
        let pid = self.child.as_ref().and_then(tokio::process::Child::id);
        untrack_codegen_child_pid(pid);
        self.child = None;
    }
}

impl Drop for ChildCleanup {
    fn drop(&mut self) {
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if let Some(mut child) = self.child.take() {
                let pid = child.id();
                match child.try_wait() {
                    Ok(Some(_)) => {
                        untrack_codegen_child_pid(pid);
                    }
                    Ok(None) | Err(_) => {
                        untrack_codegen_child_pid(pid);
                        let (tx, rx) = std::sync::mpsc::channel();
                        spawn_background_child_reap(child, Some(tx));
                        let _ = rx.recv_timeout(CHILD_DROP_REAP_WAIT_TIMEOUT);
                    }
                }
            }
        }));
        CHILD_CLEANUP_GUARD_LIVE_COUNT.fetch_sub(1, Ordering::Release);
    }
}

async fn kill_and_reap_owned(mut child: tokio::process::Child, wait_timeout: Duration) {
    let pid = child.id();
    #[cfg(unix)]
    kill_process_group_sigkill(pid);
    let _ = child.start_kill();
    let _ = tokio::time::timeout(wait_timeout, child.wait()).await;
    untrack_codegen_child_pid(pid);
}

fn spawn_background_child_reap(
    mut child: tokio::process::Child,
    completion_tx: Option<std::sync::mpsc::Sender<()>>,
) {
    if completion_tx.is_none() {
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                kill_and_reap_owned(child, CHILD_REAP_TIMEOUT).await;
            });
            return;
        }
    }

    let _ = std::thread::Builder::new()
        .name("codegen-child-reaper".to_owned())
        .spawn(move || {
            if let Ok(rt) = tokio::runtime::Runtime::new() {
                rt.block_on(kill_and_reap_owned(child, CHILD_REAP_TIMEOUT));
            } else {
                let _ = child.start_kill();
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
                loop {
                    match child.try_wait() {
                        Ok(Some(_)) => break,
                        Ok(None) => {
                            if std::time::Instant::now() >= deadline {
                                break;
                            }
                            std::thread::sleep(std::time::Duration::from_millis(20));
                        }
                        Err(_) => break,
                    }
                }
                let _ = child.kill();
                let _ = child.wait();
            }
            if let Some(tx) = completion_tx {
                let _ = tx.send(());
            }
        });
}

async fn read_stream_bounded<R>(mut reader: R, max_bytes: usize) -> std::io::Result<Vec<u8>>
where
    R: AsyncRead + Unpin,
{
    let mut out = Vec::new();
    let mut chunk = [0u8; 8192];
    loop {
        let n = reader.read(&mut chunk).await?;
        if n == 0 {
            break;
        }
        if out.len().saturating_add(n) > max_bytes {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("generator output exceeds {max_bytes} bytes"),
            ));
        }
        out.extend_from_slice(&chunk[..n]);
    }
    Ok(out)
}

fn classify_non_success_exit(generator_name: &str, status: &ProcessExitStatus) -> anyhow::Error {
    match status.code() {
        Some(127) => anyhow::anyhow!("{generator_name} command not found (exit 127)")
            .context(CodegenErrorMarker(CodegenErrorKind::Fatal)),
        Some(1) => anyhow::anyhow!("{generator_name} exited with retryable CLI error (exit 1)")
            .context(CodegenErrorMarker(CodegenErrorKind::Retryable)),
        Some(code) => anyhow::anyhow!("{generator_name} exited with code {code}")
            .context(CodegenErrorMarker(CodegenErrorKind::Retryable)),
        None => anyhow::anyhow!("{generator_name} terminated by signal")
            .context(CodegenErrorMarker(CodegenErrorKind::Retryable)),
    }
}

#[derive(Debug)]
enum ProcessExitStatus {
    Std(std::process::ExitStatus),
    Pty(portable_pty::ExitStatus),
}

impl ProcessExitStatus {
    fn code(&self) -> Option<i32> {
        match self {
            Self::Std(status) => status.code(),
            Self::Pty(status) => i32::try_from(status.exit_code()).ok(),
        }
    }
}

fn sanitize_generator_stderr(input: &str) -> String {
    let mut out = input.to_owned();
    for (name, value) in std::env::vars() {
        if is_sensitive_env_var_name(&name) && !value.is_empty() {
            out = out.replace(&value, "[REDACTED]");
        }
    }
    out
}

async fn stop_interactive_pty_io(
    stop_io: &Arc<AtomicBool>,
    pty_master: &mut Option<Box<dyn portable_pty::MasterPty + Send>>,
    output_task: JoinHandle<()>,
    input_task: Option<JoinHandle<()>>,
    child_pid: Option<u32>,
) {
    stop_io.store(true, Ordering::Release);
    let _ = pty_master.take();
    untrack_codegen_child_pid(child_pid);
    join_interactive_io_task(output_task, "opencode-pty-output").await;
    if let Some(task) = input_task {
        join_interactive_io_task(task, "opencode-pty-input").await;
    }
}

async fn run_generator_process(
    mut cmd: tokio::process::Command,
    generator: &CodeGenerator,
) -> Result<GeneratorOutput> {
    let generator_name = match generator {
        CodeGenerator::Claude => "claude",
        CodeGenerator::Opencode => "opencode",
    };
    let child = cmd.spawn().with_context(|| {
        format!("Failed to spawn `{generator_name}` — is it installed and on PATH?")
    })?;
    let mut child = ChildCleanup::new(child);

    let stdout = match child.child_mut()?.stdout.take() {
        Some(stdout) => stdout,
        None => {
            child.kill_and_reap(CHILD_REAP_TIMEOUT).await;
            anyhow::bail!("failed to capture generator stdout");
        }
    };
    let stderr = match child.child_mut()?.stderr.take() {
        Some(stderr) => stderr,
        None => {
            child.kill_and_reap(CHILD_REAP_TIMEOUT).await;
            anyhow::bail!("failed to capture generator stderr");
        }
    };

    let stdout_task = tokio::spawn(async move {
        read_stream_bounded(stdout, MAX_RESULT_BYTES)
            .await
            .map_err(anyhow::Error::from)
    });
    let stderr_task = tokio::spawn(async move {
        read_stream_bounded(stderr, MAX_RESULT_BYTES)
            .await
            .map_err(anyhow::Error::from)
    });

    let status = match child.child_mut()?.wait().await {
        Ok(status) => status,
        Err(error) => {
            child.kill_and_reap(CHILD_REAP_TIMEOUT).await;
            return Err(error).with_context(|| format!("Failed to run {generator_name}"));
        }
    };
    child.disarm();

    let stdout = stdout_task
        .await
        .context("stdout reader task panicked")?
        .context("failed reading generator stdout")?;
    let stderr = stderr_task
        .await
        .context("stderr reader task panicked")?
        .context("failed reading generator stderr")?;

    Ok(GeneratorOutput {
        status,
        stdout,
        stderr,
    })
}

fn configure_generator_command(cmd: &mut tokio::process::Command, work_dir: &Path) {
    cmd.current_dir(work_dir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    #[cfg(unix)]
    {
        cmd.process_group(0);
    }
}

fn configure_interactive_generator_command(cmd: &mut tokio::process::Command, work_dir: &Path) {
    cmd.current_dir(work_dir)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .kill_on_drop(true);
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum InteractiveWatchdogEvent {
    StartupTimeout,
    SessionTimeout,
    SessionHardTimeout,
}

fn spawn_interactive_watchdog(
    pid: Option<u32>,
    startup_timeout: Duration,
    session_timeout: Duration,
    mut startup_ready_rx: tokio::sync::oneshot::Receiver<()>,
    mut stop_rx: tokio::sync::oneshot::Receiver<()>,
) -> (
    JoinHandle<()>,
    tokio::sync::mpsc::UnboundedReceiver<InteractiveWatchdogEvent>,
) {
    let (event_tx, event_rx) = tokio::sync::mpsc::unbounded_channel();
    let handle = tokio::spawn(async move {
        tokio::select! {
            _ = &mut stop_rx => return,
            startup_ready = &mut startup_ready_rx => {
                if startup_ready.is_err() {
                    let _ = event_tx.send(InteractiveWatchdogEvent::StartupTimeout);
                    return;
                }
            }
            _ = tokio::time::sleep(startup_timeout) => {
                let _ = event_tx.send(InteractiveWatchdogEvent::StartupTimeout);
                return;
            }
        }

        tokio::select! {
            _ = &mut stop_rx => return,
            _ = tokio::time::sleep(session_timeout) => {
                #[cfg(unix)]
                kill_process_group_sigterm(pid);
                let _ = event_tx.send(InteractiveWatchdogEvent::SessionTimeout);
            }
        }

        tokio::select! {
            _ = &mut stop_rx => return,
            _ = tokio::time::sleep(Duration::from_secs(30)) => {
                #[cfg(unix)]
                kill_process_group_sigkill(pid);
                let _ = event_tx.send(InteractiveWatchdogEvent::SessionHardTimeout);
            }
        }
    });

    (handle, event_rx)
}

async fn stop_interactive_watchdog(
    stop_tx: tokio::sync::oneshot::Sender<()>,
    handle: JoinHandle<()>,
) {
    let _ = stop_tx.send(());
    let _ = tokio::time::timeout(Duration::from_millis(250), handle).await;
}

struct InteractiveWatchdogGuard {
    stop_tx: Option<tokio::sync::oneshot::Sender<()>>,
}

impl InteractiveWatchdogGuard {
    fn new(stop_tx: tokio::sync::oneshot::Sender<()>) -> Self {
        Self {
            stop_tx: Some(stop_tx),
        }
    }
}

impl Drop for InteractiveWatchdogGuard {
    fn drop(&mut self) {
        if let Some(stop_tx) = self.stop_tx.take() {
            let _ = stop_tx.send(());
        }
    }
}

#[must_use = "PtyChildCleanup must be kept until process exits or is explicitly reaped"]
struct PtyChildCleanup {
    child: Option<Box<dyn portable_pty::Child + Send>>,
    pid: Option<u32>,
}

impl PtyChildCleanup {
    fn new(child: Box<dyn portable_pty::Child + Send>) -> Self {
        let pid = child.process_id();
        Self {
            child: Some(child),
            pid,
        }
    }

    fn child_mut(&mut self) -> Result<&mut Box<dyn portable_pty::Child + Send>> {
        self.child
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("PTY child already reaped"))
    }

    fn disarm(&mut self) {
        untrack_codegen_child_pid(self.pid);
        self.child = None;
    }
}

impl Drop for PtyChildCleanup {
    fn drop(&mut self) {
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if let Some(mut child) = self.child.take() {
                match child.try_wait() {
                    Ok(Some(_)) => {}
                    Ok(None) | Err(_) => {
                        #[cfg(unix)]
                        kill_process_group_sigkill(self.pid);
                        let _ = child.kill();
                        let deadline = std::time::Instant::now() + CHILD_DROP_REAP_WAIT_TIMEOUT;
                        loop {
                            match child.try_wait() {
                                Ok(Some(_)) => break,
                                Ok(None) => {
                                    if std::time::Instant::now() >= deadline {
                                        break;
                                    }
                                    std::thread::sleep(std::time::Duration::from_millis(20));
                                }
                                Err(_) => break,
                            }
                        }
                    }
                }
                untrack_codegen_child_pid(self.pid);
            }
        }));
    }
}

async fn stop_interactive_watchdog_once(
    watchdog_guard: &mut InteractiveWatchdogGuard,
    handle: &mut Option<JoinHandle<()>>,
) {
    if let (Some(stop_tx), Some(handle)) = (watchdog_guard.stop_tx.take(), handle.take()) {
        stop_interactive_watchdog(stop_tx, handle).await;
    }
}

async fn run_generator_interactive_process(
    mut cmd: tokio::process::Command,
    generator: &CodeGenerator,
    startup_timeout: Duration,
    session_timeout: Duration,
    mut shutdown_rx: watch::Receiver<bool>,
) -> Result<()> {
    let generator_name = match generator {
        CodeGenerator::Claude => "claude",
        CodeGenerator::Opencode => "opencode",
    };
    let child = cmd
        .spawn()
        .map_err(anyhow::Error::from)
        .with_context(|| {
            format!("Failed to spawn `{generator_name}` — is it installed and on PATH?")
        })
        .context(CodegenErrorMarker(CodegenErrorKind::Fatal))?;
    let mut child = ChildCleanup::new(child);
    let child_pid = child.child_mut()?.id();
    let (startup_ready_tx, startup_ready_rx) = tokio::sync::oneshot::channel();
    let (watchdog_stop_tx, watchdog_stop_rx) = tokio::sync::oneshot::channel();
    let (watchdog_handle, mut watchdog_events) = spawn_interactive_watchdog(
        child_pid,
        startup_timeout,
        session_timeout,
        startup_ready_rx,
        watchdog_stop_rx,
    );
    let _ = startup_ready_tx.send(());
    let mut watchdog_guard = InteractiveWatchdogGuard::new(watchdog_stop_tx);
    let mut watchdog_handle = Some(watchdog_handle);

    loop {
        let status = child
            .child_mut()?
            .try_wait()
            .map_err(anyhow::Error::from)
            .with_context(|| format!("Failed to poll {generator_name} status"))
            .context(CodegenErrorMarker(CodegenErrorKind::Retryable))?;
        if let Some(status) = status {
            child.disarm();
            stop_interactive_watchdog_once(&mut watchdog_guard, &mut watchdog_handle).await;
            if !status.success() {
                let classified =
                    classify_non_success_exit(generator_name, &ProcessExitStatus::Std(status));
                return Err(classified);
            }
            return Ok(());
        }

        tokio::select! {
            changed = shutdown_rx.changed() => {
                if changed.is_ok() || *shutdown_rx.borrow() {
                    tracing::info!(generator = generator_name, "shutdown received; terminating interactive generator process");
                }
                child
                    .terminate_with_escalation(CHILD_SHUTDOWN_REAP_TIMEOUT)
                    .await;
                stop_interactive_watchdog_once(&mut watchdog_guard, &mut watchdog_handle).await;
                return Err(anyhow::anyhow!("interactive session interrupted by shutdown").context(CodegenErrorMarker(CodegenErrorKind::Retryable)));
            }
            event = watchdog_events.recv() => {
                match event {
                    Some(InteractiveWatchdogEvent::StartupTimeout) => {
                        tracing::warn!(
                            generator = generator_name,
                            startup_timeout_secs = startup_timeout.as_secs(),
                            "interactive generator startup timed out"
                        );
                        child.terminate_with_escalation(CHILD_REAP_TIMEOUT).await;
                        stop_interactive_watchdog_once(&mut watchdog_guard, &mut watchdog_handle).await;
                        return Err(anyhow::anyhow!("interactive startup timeout").context(CodegenErrorMarker(CodegenErrorKind::Retryable)));
                    }
                    Some(InteractiveWatchdogEvent::SessionTimeout) => {
                        tracing::warn!(
                            generator = generator_name,
                            timeout_secs = session_timeout.as_secs(),
                            "interactive session exceeded timeout; sending SIGTERM with SIGKILL escalation"
                        );
                        child.terminate_with_escalation(CHILD_REAP_TIMEOUT).await;
                        stop_interactive_watchdog_once(&mut watchdog_guard, &mut watchdog_handle).await;
                        return Err(anyhow::anyhow!("interactive session timeout").context(CodegenErrorMarker(CodegenErrorKind::Retryable)));
                    }
                    Some(InteractiveWatchdogEvent::SessionHardTimeout) => {
                        tracing::error!(
                            generator = generator_name,
                            hard_timeout_secs = session_timeout.saturating_add(Duration::from_secs(30)).as_secs(),
                            "interactive session exceeded hard deadline; force-killing child process group"
                        );
                        child.kill_and_reap(CHILD_REAP_TIMEOUT).await;
                        stop_interactive_watchdog_once(&mut watchdog_guard, &mut watchdog_handle).await;
                        return Err(anyhow::anyhow!("interactive session hard timeout").context(CodegenErrorMarker(CodegenErrorKind::Retryable)));
                    }
                    None => {
                        child.terminate_with_escalation(CHILD_REAP_TIMEOUT).await;
                        stop_interactive_watchdog_once(&mut watchdog_guard, &mut watchdog_handle).await;
                        return Err(anyhow::anyhow!("interactive watchdog terminated unexpectedly").context(CodegenErrorMarker(CodegenErrorKind::Retryable)));
                    }
                }
            }
            _ = tokio::time::sleep(INTERACTIVE_STARTUP_PROBE_INTERVAL) => {}
        }
    }
}

fn pty_size_from_terminal() -> PtySize {
    let cols = std::env::var("COLUMNS")
        .ok()
        .and_then(|v| v.parse::<u16>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(80);
    let rows = std::env::var("LINES")
        .ok()
        .and_then(|v| v.parse::<u16>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(24);
    PtySize {
        rows,
        cols,
        pixel_width: 0,
        pixel_height: 0,
    }
}

async fn wait_pty_child_exit(
    child: &mut Box<dyn portable_pty::Child + Send>,
    timeout: Duration,
) -> Option<ProcessExitStatus> {
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Some(ProcessExitStatus::Pty(status)),
            Ok(None) => {
                if tokio::time::Instant::now() >= deadline {
                    return None;
                }
                tokio::time::sleep(Duration::from_millis(25)).await;
            }
            Err(_) => return None,
        }
    }
}

async fn terminate_pty_child_with_escalation(
    child: &mut Box<dyn portable_pty::Child + Send>,
    pid: Option<u32>,
) {
    #[cfg(unix)]
    kill_process_group_sigterm(pid);
    let _ = child.kill();
    if wait_pty_child_exit(child, CHILD_REAP_TIMEOUT)
        .await
        .is_some()
    {
        return;
    }

    #[cfg(unix)]
    kill_process_group_sigkill(pid);
    let _ = child.kill();
    let _ = wait_pty_child_exit(child, CHILD_REAP_TIMEOUT).await;
}

async fn run_opencode_interactive_pty(
    work_dir: &Path,
    question: &str,
    iteration: usize,
    startup_timeout: Duration,
    session_timeout: Duration,
    mut shutdown_rx: watch::Receiver<bool>,
) -> Result<()> {
    let _pty_session_guard = PtySessionGuard::new();
    eprintln!(
        ">>> Iteration {iteration}: launching interactive opencode in {} ...",
        work_dir.display()
    );
    eprintln!(">>> Start by opening .lambda-rlm-result.md, then apply fixes for: \"{question}\"");

    let pty_system = NativePtySystem::default();
    let pair = pty_system
        .openpty(pty_size_from_terminal())
        .context("failed to allocate pseudo-terminal for interactive opencode")
        .context(CodegenErrorMarker(CodegenErrorKind::Retryable))?;
    let mut pty_master = Some(pair.master);

    let mut pty_cmd = CommandBuilder::new("opencode");
    pty_cmd.cwd(work_dir);
    pty_cmd.arg("run");
    pty_cmd.arg("--file");
    pty_cmd.arg(RESULT_FILE_NAME);
    pty_cmd.arg("--");
    pty_cmd.arg(interactive_prompt(question, iteration));
    let child = pair
        .slave
        .spawn_command(pty_cmd)
        .context("failed to spawn opencode inside pseudo-terminal")
        .context(CodegenErrorMarker(CodegenErrorKind::Fatal))?;
    let child_pid = child.process_id();
    track_codegen_child_pid(child_pid);
    let mut child = PtyChildCleanup::new(child);
    drop(pair.slave);

    let (startup_ready_tx, startup_ready_rx) = tokio::sync::oneshot::channel();
    let (watchdog_stop_tx, watchdog_stop_rx) = tokio::sync::oneshot::channel();
    let (watchdog_handle, mut watchdog_events) = spawn_interactive_watchdog(
        child_pid,
        startup_timeout,
        session_timeout,
        startup_ready_rx,
        watchdog_stop_rx,
    );
    let mut watchdog_guard = InteractiveWatchdogGuard::new(watchdog_stop_tx);
    let mut watchdog_handle = Some(watchdog_handle);

    let stop_io = Arc::new(AtomicBool::new(false));

    let mut pty_reader = pty_master
        .as_mut()
        .ok_or_else(|| anyhow::anyhow!("PTY master handle missing"))?
        .try_clone_reader()
        .context("failed to clone PTY reader")
        .context(CodegenErrorMarker(CodegenErrorKind::Retryable))?;
    let stop_output = Arc::clone(&stop_io);
    let output_task = tokio::task::spawn_blocking(move || {
        let mut stdout = std::io::stdout();
        let mut buf = [0u8; 8192];
        while !stop_output.load(Ordering::Acquire) {
            match pty_reader.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    if stdout.write_all(&buf[..n]).is_err() {
                        break;
                    }
                    if stdout.flush().is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    });

    let mut input_task = None;
    if std::io::stdin().is_terminal() {
        let mut pty_writer = pty_master
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("PTY master handle missing"))?
            .take_writer()
            .context("failed to open PTY writer")?;
        let stop_input = Arc::clone(&stop_io);
        #[cfg(unix)]
        {
            input_task = Some(tokio::spawn(async move {
                let stdin_file = match std::fs::File::open("/dev/stdin") {
                    Ok(file) => file,
                    Err(_) => return,
                };
                let async_stdin = match tokio::io::unix::AsyncFd::new(stdin_file) {
                    Ok(fd) => fd,
                    Err(_) => return,
                };
                let mut buf = [0u8; 8192];
                while !stop_input.load(Ordering::Acquire) {
                    let mut ready = match tokio::time::timeout(
                        INTERACTIVE_STDIN_POLL_TIMEOUT,
                        async_stdin.readable(),
                    )
                    .await
                    {
                        Ok(Ok(ready)) => ready,
                        Ok(Err(_)) => break,
                        Err(_) => continue,
                    };

                    match ready.try_io(|inner| inner.get_ref().read(&mut buf)) {
                        Ok(Ok(0)) => {
                            drop(pty_writer);
                            break;
                        }
                        Ok(Ok(n)) => {
                            if pty_writer.write_all(&buf[..n]).is_err() {
                                break;
                            }
                        }
                        Ok(Err(_)) => break,
                        Err(_would_block) => continue,
                    }
                }
            }));
        }
        #[cfg(not(unix))]
        {
            input_task = Some(tokio::task::spawn_blocking(move || {
                let mut stdin = std::io::stdin();
                let mut buf = [0u8; 8192];
                while !stop_input.load(Ordering::Acquire) {
                    match stdin.read(&mut buf) {
                        Ok(0) => {
                            drop(pty_writer);
                            break;
                        }
                        Ok(n) => {
                            if pty_writer.write_all(&buf[..n]).is_err() {
                                break;
                            }
                        }
                        Err(_) => break,
                    }
                }
            }));
        }
    }

    let _ = startup_ready_tx.send(());

    let deadline = tokio::time::Instant::now() + session_timeout;
    loop {
        match child.child_mut()?.try_wait() {
            Ok(Some(status)) => {
                stop_interactive_pty_io(
                    &stop_io,
                    &mut pty_master,
                    output_task,
                    input_task,
                    child_pid,
                )
                .await;
                stop_interactive_watchdog_once(&mut watchdog_guard, &mut watchdog_handle).await;
                child.disarm();
                if !status.success() {
                    return Err(classify_non_success_exit(
                        "opencode",
                        &ProcessExitStatus::Pty(status),
                    ));
                }
                return Ok(());
            }
            Ok(None) => {}
            Err(error) => {
                stop_interactive_pty_io(
                    &stop_io,
                    &mut pty_master,
                    output_task,
                    input_task,
                    child_pid,
                )
                .await;
                stop_interactive_watchdog_once(&mut watchdog_guard, &mut watchdog_handle).await;
                child.disarm();
                return Err(anyhow::Error::from(error))
                    .context("Failed to poll opencode interactive status")
                    .context(CodegenErrorMarker(CodegenErrorKind::Retryable));
            }
        }

        if tokio::time::Instant::now() >= deadline {
            tracing::warn!(
                timeout_secs = session_timeout.as_secs(),
                "interactive opencode session exceeded timeout; terminating process group"
            );
            terminate_pty_child_with_escalation(child.child_mut()?, child_pid).await;
            stop_interactive_pty_io(
                &stop_io,
                &mut pty_master,
                output_task,
                input_task,
                child_pid,
            )
            .await;
            stop_interactive_watchdog_once(&mut watchdog_guard, &mut watchdog_handle).await;
            child.disarm();
            return Err(anyhow::anyhow!("interactive session timeout")
                .context(CodegenErrorMarker(CodegenErrorKind::Retryable)));
        }

        tokio::select! {
            changed = shutdown_rx.changed() => {
                if changed.is_ok() || *shutdown_rx.borrow() {
                    tracing::info!("shutdown received; terminating interactive opencode PTY process");
                }
                terminate_pty_child_with_escalation(child.child_mut()?, child_pid).await;
                stop_interactive_pty_io(
                    &stop_io,
                    &mut pty_master,
                    output_task,
                    input_task,
                    child_pid,
                )
                .await;
                stop_interactive_watchdog_once(&mut watchdog_guard, &mut watchdog_handle).await;
                child.disarm();
                return Err(anyhow::anyhow!("interactive session interrupted by shutdown").context(CodegenErrorMarker(CodegenErrorKind::Retryable)));
            }
            event = watchdog_events.recv() => {
                match event {
                    Some(InteractiveWatchdogEvent::StartupTimeout) => {
                        tracing::warn!(
                            startup_timeout_secs = startup_timeout.as_secs(),
                            "interactive opencode PTY startup timed out"
                        );
                        terminate_pty_child_with_escalation(child.child_mut()?, child_pid).await;
                        stop_interactive_pty_io(
                            &stop_io,
                            &mut pty_master,
                            output_task,
                            input_task,
                            child_pid,
                        )
                        .await;
                        stop_interactive_watchdog_once(&mut watchdog_guard, &mut watchdog_handle).await;
                        child.disarm();
                        return Err(anyhow::anyhow!("interactive startup timeout").context(CodegenErrorMarker(CodegenErrorKind::Retryable)));
                    }
                    Some(InteractiveWatchdogEvent::SessionTimeout) => {
                        tracing::warn!(
                            timeout_secs = session_timeout.as_secs(),
                            "interactive opencode PTY session exceeded timeout; sending SIGTERM with SIGKILL escalation"
                        );
                        terminate_pty_child_with_escalation(child.child_mut()?, child_pid).await;
                        stop_interactive_pty_io(
                            &stop_io,
                            &mut pty_master,
                            output_task,
                            input_task,
                            child_pid,
                        )
                        .await;
                        stop_interactive_watchdog_once(&mut watchdog_guard, &mut watchdog_handle).await;
                        child.disarm();
                        return Err(anyhow::anyhow!("interactive session timeout").context(CodegenErrorMarker(CodegenErrorKind::Retryable)));
                    }
                    Some(InteractiveWatchdogEvent::SessionHardTimeout) => {
                        tracing::error!(
                            hard_timeout_secs = session_timeout.saturating_add(Duration::from_secs(30)).as_secs(),
                            "interactive opencode PTY exceeded hard deadline; force-killing child process group"
                        );
                        terminate_pty_child_with_escalation(child.child_mut()?, child_pid).await;
                        stop_interactive_pty_io(
                            &stop_io,
                            &mut pty_master,
                            output_task,
                            input_task,
                            child_pid,
                        )
                        .await;
                        stop_interactive_watchdog_once(&mut watchdog_guard, &mut watchdog_handle).await;
                        child.disarm();
                        return Err(anyhow::anyhow!("interactive session hard timeout").context(CodegenErrorMarker(CodegenErrorKind::Retryable)));
                    }
                    None => {
                        terminate_pty_child_with_escalation(child.child_mut()?, child_pid).await;
                        stop_interactive_pty_io(
                            &stop_io,
                            &mut pty_master,
                            output_task,
                            input_task,
                            child_pid,
                        )
                        .await;
                        stop_interactive_watchdog_once(&mut watchdog_guard, &mut watchdog_handle).await;
                        child.disarm();
                        return Err(anyhow::anyhow!("interactive watchdog terminated unexpectedly").context(CodegenErrorMarker(CodegenErrorKind::Retryable)));
                    }
                }
            }
            _ = tokio::time::sleep(Duration::from_millis(25)) => {}
        }
    }
}

fn interactive_prompt(question: &str, iteration: usize) -> String {
    format!(
        "You are running interactive mode for lambda-RLM iteration {iteration}.\n\
         Read .lambda-rlm-result.md first, then execute fixes for: \"{question}\".\n\
         Commit your changes when done and print a one-line summary (or CLEAN)."
    )
}

/// Spawn claude in print mode and wait for it to finish.
/// Returns Claude's output summary (last non-empty line).
///
/// PRE: work_dir exists and is writable
/// POST: claude process has exited
async fn run_claude(
    result: &str,
    work_dir: &Path,
    question: &str,
    iteration: usize,
) -> Result<Arc<str>> {
    let mut wrote_result_file = true;
    if let Err(error) = write_result_file_atomically(work_dir, result, iteration).await {
        if is_storage_full_error(&error) {
            tracing::warn!(iteration, error = %error, "disk full writing result file; falling back to in-memory prompt");
            wrote_result_file = false;
        } else {
            return Err(error);
        }
    }

    let run = async {
        let log_file = work_dir.join(format!(".lambda-rlm-claude-{iteration}.log"));

        eprintln!(
            ">>> Iteration {iteration}: launching claude in {} ...",
            work_dir.display()
        );
        let prompt = if wrote_result_file {
            build_prompt(question, iteration)
        } else {
            build_prompt_with_inline_result(question, iteration, result)
        };

        let mut cmd = tokio::process::Command::new("claude");
        cmd.arg("--dangerously-skip-permissions")
            .arg("-p")
            .arg(&prompt);
        configure_generator_command(&mut cmd, work_dir);

        let output = run_generator_process(cmd, &CodeGenerator::Claude).await?;

        let stdout = std::str::from_utf8(&output.stdout)
            .context("claude subprocess produced invalid UTF-8 on stdout")?;

        // Save full output to log — warn on failure rather than swallowing
        if let Err(e) = write_log_file_atomically(&log_file, stdout).await {
            tracing::warn!(path = %log_file.display(), error = %e, "failed to write iteration log");
        }

        if !output.status.success() {
            let stderr = sanitize_generator_stderr(&String::from_utf8_lossy(&output.stderr));
            tracing::error!(status = %output.status, stderr = %stderr, "claude exited with failure status");
            return Err(classify_non_success_exit(
                "claude",
                &ProcessExitStatus::Std(output.status),
            ));
        }

        let summary = CodeGenerator::Claude.extract_result(&output.stdout)?;

        Ok(summary)
    }
    .await;

    remove_result_file(work_dir).await;
    cleanup_old_logs(work_dir, &CodeGenerator::Claude);
    run
}

async fn run_claude_interactive(
    result: &str,
    work_dir: &Path,
    question: &str,
    iteration: usize,
    startup_timeout: Duration,
    session_timeout: Duration,
    shutdown_rx: watch::Receiver<bool>,
) -> Result<Arc<str>> {
    write_result_file_atomically(work_dir, result, iteration).await?;

    let run = async {
        eprintln!(
            ">>> Iteration {iteration}: launching interactive claude in {} ...",
            work_dir.display()
        );
        let mut cmd = tokio::process::Command::new("claude");
        cmd.arg("--dangerously-skip-permissions")
            .arg(interactive_prompt(question, iteration));
        configure_interactive_generator_command(&mut cmd, work_dir);
        run_generator_interactive_process(
            cmd,
            &CodeGenerator::Claude,
            startup_timeout,
            session_timeout,
            shutdown_rx,
        )
        .await?;
        Ok(Arc::<str>::from("interactive session completed"))
    }
    .await;

    remove_result_file(work_dir).await;
    run
}

/// Spawn opencode in non-interactive mode and wait for it to finish.
/// Returns OpenCode's output summary (first non-empty line).
///
/// PRE: work_dir exists and is writable
/// POST: opencode process has exited
async fn run_opencode(
    result: &str,
    work_dir: &Path,
    question: &str,
    iteration: usize,
) -> Result<Arc<str>> {
    let mut wrote_result_file = true;
    if let Err(error) = write_result_file_atomically(work_dir, result, iteration).await {
        if is_storage_full_error(&error) {
            tracing::warn!(iteration, error = %error, "disk full writing result file; falling back to in-memory prompt");
            wrote_result_file = false;
        } else {
            return Err(error);
        }
    }

    let run = async {
        let log_file = work_dir.join(format!(".lambda-rlm-opencode-{iteration}.log"));

        eprintln!(
            ">>> Iteration {iteration}: launching opencode in {} ...",
            work_dir.display()
        );
        let prompt = if wrote_result_file {
            build_prompt(question, iteration)
        } else {
            build_prompt_with_inline_result(question, iteration, result)
        };

        let mut cmd = tokio::process::Command::new("opencode");
        cmd.arg("run");
        if wrote_result_file {
            cmd.arg("--file").arg(".lambda-rlm-result.md");
        }
        cmd.arg("--").arg(&prompt);
        configure_generator_command(&mut cmd, work_dir);

        let output = run_generator_process(cmd, &CodeGenerator::Opencode).await?;

        let stdout = std::str::from_utf8(&output.stdout)
            .context("opencode subprocess produced invalid UTF-8 on stdout")?;

        if let Err(e) = write_log_file_atomically(&log_file, stdout).await {
            tracing::warn!(path = %log_file.display(), error = %e, "failed to write iteration log");
        }

        if !output.status.success() {
            let stderr = sanitize_generator_stderr(&String::from_utf8_lossy(&output.stderr));
            tracing::error!(status = %output.status, stderr = %stderr, "opencode exited with failure status");
            return Err(classify_non_success_exit(
                "opencode",
                &ProcessExitStatus::Std(output.status),
            ));
        }

        let summary = CodeGenerator::Opencode.extract_result(&output.stdout)?;

        Ok(summary)
    }
    .await;

    remove_result_file(work_dir).await;
    cleanup_old_logs(work_dir, &CodeGenerator::Opencode);
    run
}

async fn run_opencode_interactive(
    result: &str,
    work_dir: &Path,
    question: &str,
    iteration: usize,
    startup_timeout: Duration,
    session_timeout: Duration,
    shutdown_rx: watch::Receiver<bool>,
) -> Result<Arc<str>> {
    write_result_file_atomically(work_dir, result, iteration).await?;

    let run = async {
        run_opencode_interactive_pty(
            work_dir,
            question,
            iteration,
            startup_timeout,
            session_timeout,
            shutdown_rx,
        )
        .await?;
        Ok(Arc::<str>::from("interactive session completed"))
    }
    .await;

    remove_result_file(work_dir).await;
    run
}

/// Log file name for the given generator and iteration.
pub fn log_file_name(generator: &CodeGenerator, iteration: usize) -> String {
    match generator {
        CodeGenerator::Claude => format!(".lambda-rlm-claude-{iteration}.log"),
        CodeGenerator::Opencode => format!(".lambda-rlm-opencode-{iteration}.log"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use loom::sync::atomic::{AtomicUsize as LoomAtomicUsize, Ordering as LoomOrdering};
    use loom::sync::Arc as LoomArc;
    use loom::sync::Mutex as LoomMutex;
    use loom::thread as loom_thread;
    use std::time::Duration as StdDuration;
    use std::time::Instant as StdInstant;

    fn test_pid_alive(pid: u32) -> bool {
        std::process::Command::new("kill")
            .arg("-0")
            .arg(pid.to_string())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|status| status.success())
            .unwrap_or(false)
    }

    fn unique_pid_file_path() -> std::path::PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!("lambda-rlm-interactive-pid-{stamp}.txt"))
    }

    #[test]
    fn result_validation_rejects_oversized_payload() {
        let huge = "x".repeat(MAX_RESULT_BYTES + 1);
        assert!(validate_analysis_result_content(&huge).is_err());
    }

    #[test]
    fn result_validation_rejects_empty_payload() {
        assert!(validate_analysis_result_content("   \n\n").is_err());
    }

    #[test]
    fn generator_output_validation_rejects_nul_bytes() {
        let out = "ok\0bad";
        assert!(validate_generator_output(&CodeGenerator::Claude, out).is_err());
    }

    #[test]
    fn log_file_name_is_stable_for_all_generators() {
        assert_eq!(
            log_file_name(&CodeGenerator::Claude, 7),
            ".lambda-rlm-claude-7.log"
        );
        assert_eq!(
            log_file_name(&CodeGenerator::Opencode, 7),
            ".lambda-rlm-opencode-7.log"
        );
    }

    #[test]
    fn last_non_empty_line_extracts_final_signal() {
        let output = "line one\n\nline two\n\n";
        assert_eq!(last_non_empty_line(output), Some("line two"));
    }

    #[test]
    fn extract_result_uses_last_line_for_claude() {
        let output = b"thinking...\n\nfix done\n";
        let summary = CodeGenerator::Claude
            .extract_result(output)
            .expect("summary");
        assert_eq!(summary.as_ref(), "fix done");
    }

    #[test]
    fn extract_result_uses_first_line_for_opencode() {
        let output = b"metadata\nfix done\n";
        let summary = CodeGenerator::Opencode
            .extract_result(output)
            .expect("summary");
        assert_eq!(summary.as_ref(), "metadata");
    }

    #[test]
    fn parse_opencode_major_version_accepts_prefixed_output() {
        assert_eq!(parse_opencode_major_version("opencode 1.3.13"), Some(1));
    }

    #[test]
    fn parse_opencode_major_version_accepts_plain_semver_output() {
        assert_eq!(parse_opencode_major_version("1.3.13"), Some(1));
    }

    #[test]
    fn parse_opencode_major_version_rejects_unparseable_output() {
        assert_eq!(
            parse_opencode_major_version("opencode version unknown"),
            None
        );
    }

    #[test]
    fn extract_result_skips_opencode_headers() {
        let output =
            b"OpenCode v0.6.0\nModel: gpt-5\nStatus: done\nFixed input validation and added tests\n";
        let summary = CodeGenerator::Opencode
            .extract_result(output)
            .expect("summary");
        assert_eq!(summary.as_ref(), "Fixed input validation and added tests");
    }

    #[test]
    fn extract_result_rejects_control_chars_for_opencode() {
        let output = b"OpenCode v1.0\nStatus: done\n\x01bad";
        let summary = CodeGenerator::Opencode
            .extract_result(output)
            .expect("degraded summary");
        assert_eq!(summary.as_ref(), "(degraded: unparseable opencode output)");
    }

    #[test]
    fn opencode_regex_fallback_extracts_summary() {
        let output = "Status: done\n>>> CLEAN\n";
        let summary = extract_opencode_summary_line(output).expect("summary line");
        assert_eq!(summary, "CLEAN");
    }

    #[test]
    fn codegen_cache_key_changes_with_generator() {
        let result = "fix critical issue";
        let claude = codegen_cache_key(&CodeGenerator::Claude, result, false);
        let opencode = codegen_cache_key(&CodeGenerator::Opencode, result, false);
        assert_ne!(claude, opencode);
        assert!(claude.starts_with("b3-codegen_"));
        assert!(opencode.starts_with("b3-codegen_"));
    }

    #[test]
    fn codegen_cache_key_differs_for_interactive_mode() {
        let result = "fix critical issue";
        let headless = codegen_cache_key(&CodeGenerator::Claude, result, false);
        let interactive = codegen_cache_key(&CodeGenerator::Claude, result, true);
        assert_ne!(headless, interactive);
    }

    #[test]
    fn codegen_flight_key_is_generator_prefixed() {
        let key = codegen_flight_key(
            &CodeGenerator::Opencode,
            Path::new("/tmp"),
            "q",
            1,
            "r",
            false,
        );
        assert!(key.starts_with("opencode:"));
    }

    #[test]
    fn interactive_mode_disables_single_flight() {
        assert!(single_flight_enabled(false));
        assert!(!single_flight_enabled(true));
    }

    #[test]
    fn stale_single_flight_entries_are_evicted() {
        let flights = DashMap::new();
        let (tx, _rx) = watch::channel::<Option<CodegenFlightResult>>(None);
        flights.insert(
            "stale".to_string(),
            CodegenFlightEntry {
                sender: Arc::new(tx),
                inserted_at: Instant::now() - (CODEGEN_SINGLE_FLIGHT_TTL + Duration::from_secs(1)),
            },
        );

        let removed = evict_stale_codegen_flights(&flights);
        assert_eq!(removed, 1);
        assert!(flights.is_empty());
    }

    #[test]
    fn extract_result_treats_empty_stdout_as_degraded() {
        let summary = CodeGenerator::Opencode
            .extract_result(b"   \n\n")
            .expect("summary");
        assert_eq!(summary.as_ref(), "(degraded: empty generator output)");
    }

    #[test]
    fn classify_exit_codes_for_codegen_resilience() {
        let retryable = classify_non_success_exit(
            "opencode",
            &ProcessExitStatus::Std(
                std::process::Command::new("sh")
                    .arg("-c")
                    .arg("exit 1")
                    .status()
                    .expect("status"),
            ),
        );
        let fatal = classify_non_success_exit(
            "opencode",
            &ProcessExitStatus::Std(
                std::process::Command::new("sh")
                    .arg("-c")
                    .arg("exit 127")
                    .status()
                    .expect("status"),
            ),
        );

        assert!(error_has_context(&retryable, "codegen_retryable"));
        assert!(error_has_context(&fatal, "codegen_fatal"));
    }

    #[test]
    fn sanitize_generator_stderr_redacts_claude_api_key() {
        let needle = "unit-test-claude-secret";
        std::env::set_var("CLAUDE_API_KEY", needle);
        let sanitized = sanitize_generator_stderr(&format!("boom {needle}"));
        std::env::remove_var("CLAUDE_API_KEY");
        assert!(!sanitized.contains(needle));
        assert!(sanitized.contains("[REDACTED]"));
    }

    #[test]
    fn sanitize_generator_stderr_redacts_pattern_matched_secret() {
        let needle = "unit-test-custom-secret";
        std::env::set_var("INTERNAL_DEPLOY_TOKEN", needle);
        let sanitized = sanitize_generator_stderr(&format!("boom {needle}"));
        std::env::remove_var("INTERNAL_DEPLOY_TOKEN");
        assert!(!sanitized.contains(needle));
        assert!(sanitized.contains("[REDACTED]"));
    }

    #[test]
    fn is_storage_full_error_matches_io_kind() {
        let io = std::io::Error::from(std::io::ErrorKind::StorageFull);
        let err = anyhow::Error::new(io);
        assert!(is_storage_full_error(&err));
    }

    #[test]
    fn is_storage_full_error_matches_common_disk_full_messages() {
        let err = anyhow::anyhow!("write failed: No space left on device");
        assert!(is_storage_full_error(&err));
        let other = anyhow::anyhow!("permission denied");
        assert!(!is_storage_full_error(&other));
    }

    #[test]
    fn codegen_circuits_are_isolated_by_generator_and_mode() {
        let claude = codegen_circuit(&CodeGenerator::Claude, false).expect("claude circuit");
        let opencode = codegen_circuit(&CodeGenerator::Opencode, false).expect("opencode circuit");
        let opencode_interactive =
            codegen_circuit(&CodeGenerator::Opencode, true).expect("opencode interactive circuit");

        claude.record_success();
        opencode.record_success();
        opencode_interactive.record_success();

        for _ in 0..OPENCODE_CB_THRESHOLD {
            opencode.record_failure();
        }

        assert!(!opencode.allow_request());
        assert!(claude.allow_request());
        assert!(opencode_interactive.allow_request());

        claude.record_success();
        opencode.record_success();
        opencode_interactive.record_success();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn child_cleanup_drop_reaps_process() {
        let mut cmd = tokio::process::Command::new("sleep");
        cmd.arg("1000")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .stdin(std::process::Stdio::null());

        let child = cmd.spawn().expect("spawn sleep");
        let pid = child.id().expect("pid");
        let cleanup = ChildCleanup::new(child);
        drop(cleanup);

        tokio::time::timeout(StdDuration::from_secs(5), async move {
            loop {
                if !test_pid_alive(pid) {
                    break;
                }
                tokio::time::sleep(StdDuration::from_millis(50)).await;
            }
        })
        .await
        .expect("child process should be reaped within timeout");
    }

    #[test]
    fn child_cleanup_drop_reaps_without_runtime() {
        let (cleanup, pid) = {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("runtime");
            rt.block_on(async {
                let mut cmd = tokio::process::Command::new("sleep");
                cmd.arg("1000")
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .stdin(std::process::Stdio::null());
                let child = cmd.spawn().expect("spawn sleep");
                let pid = child.id().expect("pid");
                (ChildCleanup::new(child), pid)
            })
        };

        drop(cleanup);

        let deadline = StdInstant::now() + StdDuration::from_secs(5);
        while test_pid_alive(pid) && StdInstant::now() < deadline {
            std::thread::sleep(StdDuration::from_millis(50));
        }
        assert!(
            !test_pid_alive(pid),
            "child process should be reaped by fallback thread"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn child_cleanup_drop_is_bounded() {
        let mut cmd = tokio::process::Command::new("sleep");
        cmd.arg("1000")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .stdin(std::process::Stdio::null());

        let child = cmd.spawn().expect("spawn sleep");
        let cleanup = ChildCleanup::new(child);

        let start = StdInstant::now();
        drop(cleanup);
        assert!(
            start.elapsed() <= CHILD_DROP_REAP_WAIT_TIMEOUT + StdDuration::from_secs(1),
            "drop should not exceed bounded reap wait"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn interactive_process_requires_startup_readiness() {
        let (_shutdown_tx, shutdown_rx) = watch::channel(false);
        let mut cmd = tokio::process::Command::new("sleep");
        cmd.arg("1")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .stdin(std::process::Stdio::null());

        run_generator_interactive_process(
            cmd,
            &CodeGenerator::Opencode,
            StdDuration::from_secs(5),
            StdDuration::from_secs(5),
            shutdown_rx,
        )
        .await
        .expect("interactive session should complete after startup probing");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn interactive_process_reaps_on_successful_exit() {
        let pid_file = unique_pid_file_path();
        let pid_file_str = pid_file.to_string_lossy().into_owned();
        let shell = format!("echo $$ > \"{pid_file_str}\"; sleep 0.1; exit 0");

        let (_shutdown_tx, shutdown_rx) = watch::channel(false);
        let mut cmd = tokio::process::Command::new("sh");
        cmd.arg("-c")
            .arg(shell)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .stdin(std::process::Stdio::null());

        run_generator_interactive_process(
            cmd,
            &CodeGenerator::Opencode,
            StdDuration::from_secs(2),
            StdDuration::from_secs(5),
            shutdown_rx,
        )
        .await
        .expect("interactive process should exit successfully");

        let pid = tokio::time::timeout(StdDuration::from_secs(3), async {
            loop {
                if let Ok(raw) = std::fs::read_to_string(&pid_file) {
                    if let Ok(pid) = raw.trim().parse::<u32>() {
                        return pid;
                    }
                }
                tokio::time::sleep(StdDuration::from_millis(20)).await;
            }
        })
        .await
        .expect("interactive process should write pid file");

        tokio::time::timeout(StdDuration::from_secs(5), async {
            while test_pid_alive(pid) {
                tokio::time::sleep(StdDuration::from_millis(50)).await;
            }
        })
        .await
        .expect("interactive process should be reaped on success");

        let _ = std::fs::remove_file(pid_file);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn interactive_process_reports_early_non_success_exit() {
        let (_shutdown_tx, shutdown_rx) = watch::channel(false);
        let mut cmd = tokio::process::Command::new("sh");
        cmd.arg("-c")
            .arg("exit 1")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .stdin(std::process::Stdio::null());

        let err = run_generator_interactive_process(
            cmd,
            &CodeGenerator::Opencode,
            StdDuration::from_millis(500),
            StdDuration::from_secs(5),
            shutdown_rx,
        )
        .await
        .expect_err("non-success exit should fail");
        assert!(error_has_context(&err, "codegen_retryable"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn interactive_process_spawn_failure_is_fatal() {
        let (_shutdown_tx, shutdown_rx) = watch::channel(false);
        let cmd = tokio::process::Command::new("binary-that-should-not-exist-lambda-rlm");
        let err = run_generator_interactive_process(
            cmd,
            &CodeGenerator::Opencode,
            StdDuration::from_millis(50),
            StdDuration::from_secs(5),
            shutdown_rx,
        )
        .await
        .expect_err("missing binary should fail to spawn");
        assert!(error_has_context(&err, "codegen_fatal"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn interactive_process_shutdown_signal_kills_child() {
        let pid_file = unique_pid_file_path();
        let pid_file_str = pid_file.to_string_lossy().into_owned();
        let shell = format!("echo $$ > \"{pid_file_str}\"; exec sleep 1000");

        let mut cmd = tokio::process::Command::new("sh");
        cmd.arg("-c")
            .arg(shell)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .stdin(std::process::Stdio::null());

        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let join = tokio::spawn(run_generator_interactive_process(
            cmd,
            &CodeGenerator::Opencode,
            StdDuration::from_millis(50),
            StdDuration::from_secs(30),
            shutdown_rx,
        ));

        let pid = tokio::time::timeout(StdDuration::from_secs(3), async {
            loop {
                if let Ok(raw) = std::fs::read_to_string(&pid_file) {
                    if let Ok(pid) = raw.trim().parse::<u32>() {
                        return pid;
                    }
                }
                tokio::time::sleep(StdDuration::from_millis(20)).await;
            }
        })
        .await
        .expect("interactive process should write pid file");

        shutdown_tx.send(true).expect("send shutdown");
        let shutdown_started = tokio::time::Instant::now();
        let err = tokio::time::timeout(StdDuration::from_millis(500), join)
            .await
            .expect("interactive task should finish quickly after shutdown")
            .expect("interactive task join should succeed")
            .expect_err("shutdown should interrupt interactive process");
        assert!(error_has_context(&err, "codegen_retryable"));
        assert!(
            shutdown_started.elapsed() <= StdDuration::from_millis(500),
            "interactive shutdown path exceeded 500ms"
        );

        tokio::time::timeout(StdDuration::from_secs(5), async {
            while test_pid_alive(pid) {
                tokio::time::sleep(StdDuration::from_millis(50)).await;
            }
        })
        .await
        .expect("child process should be reaped after shutdown");

        let _ = std::fs::remove_file(pid_file);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn interactive_process_reaps_on_parent_abort() {
        let pid_file = unique_pid_file_path();
        let pid_file_str = pid_file.to_string_lossy().into_owned();
        let shell = format!("echo $$ > \"{pid_file_str}\"; exec sleep 1000");

        let mut cmd = tokio::process::Command::new("sh");
        cmd.arg("-c")
            .arg(shell)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .stdin(std::process::Stdio::null());

        let (_shutdown_tx, shutdown_rx) = watch::channel(false);
        let join = tokio::spawn(run_generator_interactive_process(
            cmd,
            &CodeGenerator::Opencode,
            StdDuration::from_millis(50),
            StdDuration::from_secs(30),
            shutdown_rx,
        ));

        let pid = tokio::time::timeout(StdDuration::from_secs(3), async {
            loop {
                if let Ok(raw) = std::fs::read_to_string(&pid_file) {
                    if let Ok(pid) = raw.trim().parse::<u32>() {
                        return pid;
                    }
                }
                tokio::time::sleep(StdDuration::from_millis(20)).await;
            }
        })
        .await
        .expect("interactive process should write pid file");

        join.abort();
        let _ = join.await;

        tokio::time::timeout(StdDuration::from_secs(5), async {
            while test_pid_alive(pid) {
                tokio::time::sleep(StdDuration::from_millis(50)).await;
            }
        })
        .await
        .expect("aborting parent task should still reap interactive process");

        let _ = std::fs::remove_file(pid_file);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn interactive_process_session_timeout_kills_child() {
        let (_shutdown_tx, shutdown_rx) = watch::channel(false);
        let mut cmd = tokio::process::Command::new("sleep");
        cmd.arg("1000")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .stdin(std::process::Stdio::null());

        let err = run_generator_interactive_process(
            cmd,
            &CodeGenerator::Opencode,
            StdDuration::from_millis(250),
            StdDuration::from_millis(120),
            shutdown_rx,
        )
        .await
        .expect_err("session timeout should terminate hanging interactive process");
        assert!(error_has_context(&err, "codegen_retryable"));

        tokio::time::timeout(StdDuration::from_secs(2), async {
            loop {
                if codegen_guard_live_counts().child_cleanup == 0 {
                    break;
                }
                tokio::time::sleep(StdDuration::from_millis(20)).await;
            }
        })
        .await
        .expect("child cleanup guards should drain after timeout");
    }

    #[tokio::test(flavor = "current_thread")]
    #[cfg(unix)]
    async fn interactive_process_group_cleanup_on_timeout() {
        let pid_file = unique_pid_file_path();
        let pid_file_str = pid_file.to_string_lossy().into_owned();
        let shell = format!("sleep 1000 & echo $! > \"{pid_file_str}\"; exec sleep 1000");

        let (_shutdown_tx, shutdown_rx) = watch::channel(false);
        let mut cmd = tokio::process::Command::new("sh");
        cmd.arg("-c")
            .arg(shell)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .stdin(std::process::Stdio::null());
        cmd.process_group(0);

        let err = run_generator_interactive_process(
            cmd,
            &CodeGenerator::Opencode,
            StdDuration::from_millis(250),
            StdDuration::from_millis(120),
            shutdown_rx,
        )
        .await
        .expect_err("session timeout should terminate interactive process group");
        assert!(error_has_context(&err, "codegen_retryable"));

        let child_pid = tokio::time::timeout(StdDuration::from_secs(3), async {
            loop {
                if let Ok(raw) = std::fs::read_to_string(&pid_file) {
                    if let Ok(pid) = raw.trim().parse::<u32>() {
                        return pid;
                    }
                }
                tokio::time::sleep(StdDuration::from_millis(20)).await;
            }
        })
        .await
        .expect("process group child pid should be written");

        tokio::time::timeout(StdDuration::from_secs(5), async {
            while test_pid_alive(child_pid) {
                tokio::time::sleep(StdDuration::from_millis(50)).await;
            }
        })
        .await
        .expect("session timeout should reap background process group child");

        let _ = std::fs::remove_file(pid_file);
    }

    #[tokio::test(flavor = "current_thread")]
    #[cfg(unix)]
    async fn interactive_timeout_sends_sigterm_then_sigkill() {
        let pid_file = unique_pid_file_path();
        let term_file = unique_pid_file_path();
        let pid_file_str = pid_file.to_string_lossy().into_owned();
        let term_file_str = term_file.to_string_lossy().into_owned();
        let shell = format!(
            "trap 'echo term > \"{term_file_str}\"' TERM; echo $$ > \"{pid_file_str}\"; while true; do sleep 1; done"
        );

        let (_shutdown_tx, shutdown_rx) = watch::channel(false);
        let mut cmd = tokio::process::Command::new("sh");
        cmd.arg("-c")
            .arg(shell)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .stdin(std::process::Stdio::null());
        cmd.process_group(0);

        let err = run_generator_interactive_process(
            cmd,
            &CodeGenerator::Opencode,
            StdDuration::from_millis(50),
            StdDuration::from_millis(120),
            shutdown_rx,
        )
        .await
        .expect_err("session timeout should terminate interactive process group");
        assert!(error_has_context(&err, "codegen_retryable"));

        let pid = tokio::time::timeout(StdDuration::from_secs(3), async {
            loop {
                if let Ok(raw) = std::fs::read_to_string(&pid_file) {
                    if let Ok(pid) = raw.trim().parse::<u32>() {
                        return pid;
                    }
                }
                tokio::time::sleep(StdDuration::from_millis(20)).await;
            }
        })
        .await
        .expect("interactive process should publish pid");

        tokio::time::timeout(StdDuration::from_secs(3), async {
            loop {
                if term_file.exists() {
                    break;
                }
                tokio::time::sleep(StdDuration::from_millis(20)).await;
            }
        })
        .await
        .expect("interactive timeout should send SIGTERM before escalation");

        tokio::time::timeout(StdDuration::from_secs(8), async {
            while test_pid_alive(pid) {
                tokio::time::sleep(StdDuration::from_millis(50)).await;
            }
        })
        .await
        .expect("interactive timeout should escalate to SIGKILL for TERM-ignoring child");

        let _ = std::fs::remove_file(pid_file);
        let _ = std::fs::remove_file(term_file);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn interactive_cleanup_has_no_guard_leak_over_many_iterations() {
        for _ in 0..200 {
            let (_shutdown_tx, shutdown_rx) = watch::channel(false);
            let mut cmd = tokio::process::Command::new("sh");
            cmd.arg("-c")
                .arg("sleep 0.1; exit 0")
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .stdin(std::process::Stdio::null());

            run_generator_interactive_process(
                cmd,
                &CodeGenerator::Opencode,
                StdDuration::from_millis(50),
                StdDuration::from_secs(1),
                shutdown_rx,
            )
            .await
            .expect("interactive process exits cleanly");
        }

        tokio::time::timeout(StdDuration::from_secs(5), async {
            loop {
                if codegen_guard_live_counts().child_cleanup == 0 {
                    break;
                }
                tokio::time::sleep(StdDuration::from_millis(20)).await;
            }
        })
        .await
        .expect("child cleanup guards should return to zero");
    }

    #[test]
    fn interactive_startup_timeout_is_clamped() {
        assert_eq!(
            clamp_interactive_startup_timeout(StdDuration::from_secs(1)),
            INTERACTIVE_STARTUP_TIMEOUT_MIN
        );
        assert_eq!(
            clamp_interactive_startup_timeout(StdDuration::from_secs(3)),
            INTERACTIVE_STARTUP_TIMEOUT_MIN
        );
        assert_eq!(
            clamp_interactive_startup_timeout(StdDuration::from_secs(60)),
            INTERACTIVE_STARTUP_TIMEOUT_MAX
        );
        assert_eq!(
            clamp_interactive_startup_timeout(StdDuration::from_secs(120)),
            INTERACTIVE_STARTUP_TIMEOUT_MAX
        );
    }

    #[test]
    #[ignore = "loom model test; run explicitly when validating lock-free invariants"]
    fn loom_child_cleanup_no_double_reap() {
        loom::model(|| {
            let state = LoomArc::new(LoomMutex::new(Some(())));
            let reap_count = LoomArc::new(LoomAtomicUsize::new(0));

            let state_a = LoomArc::clone(&state);
            let count_a = LoomArc::clone(&reap_count);
            let t1 = loom_thread::spawn(move || {
                let mut lock = state_a.lock().expect("state lock");
                if lock.take().is_some() {
                    count_a.fetch_add(1, LoomOrdering::AcqRel);
                }
            });

            let state_b = LoomArc::clone(&state);
            let count_b = LoomArc::clone(&reap_count);
            let t2 = loom_thread::spawn(move || {
                let mut lock = state_b.lock().expect("state lock");
                if lock.take().is_some() {
                    count_b.fetch_add(1, LoomOrdering::AcqRel);
                }
            });

            t1.join().expect("thread1 joined");
            t2.join().expect("thread2 joined");

            let total = reap_count.load(LoomOrdering::Acquire);
            assert!(total <= 1, "reap action must happen at most once");
        });
    }

    #[test]
    #[ignore = "loom model test; run explicitly when validating lock-free invariants"]
    fn loom_codegen_budget_guard_restores_budget_under_race() {
        struct LoomBudget {
            remaining: LoomAtomicUsize,
        }

        impl LoomBudget {
            fn new(units: usize) -> Self {
                Self {
                    remaining: LoomAtomicUsize::new(units),
                }
            }

            fn try_reserve(&self, units: usize) -> bool {
                let current = self.remaining.load(LoomOrdering::Acquire);
                if current < units {
                    return false;
                }
                self.remaining
                    .compare_exchange(
                        current,
                        current - units,
                        LoomOrdering::AcqRel,
                        LoomOrdering::Acquire,
                    )
                    .is_ok()
            }

            fn unreserve(&self, units: usize) {
                let _ = self.remaining.fetch_update(
                    LoomOrdering::AcqRel,
                    LoomOrdering::Acquire,
                    |current| Some(current.saturating_add(units)),
                );
            }
        }

        struct LoomCodegenGuard {
            budget: LoomArc<LoomBudget>,
            units: usize,
            committed: bool,
        }

        impl LoomCodegenGuard {
            fn commit(&mut self) {
                self.committed = true;
            }
        }

        impl Drop for LoomCodegenGuard {
            fn drop(&mut self) {
                if !self.committed {
                    self.budget.unreserve(self.units);
                }
            }
        }

        loom::model(|| {
            let budget = LoomArc::new(LoomBudget::new(50));

            let b1 = LoomArc::clone(&budget);
            let t1 = loom_thread::spawn(move || {
                if b1.try_reserve(50) {
                    let mut guard = LoomCodegenGuard {
                        budget: LoomArc::clone(&b1),
                        units: 50,
                        committed: false,
                    };
                    guard.commit();
                }
            });

            let b2 = LoomArc::clone(&budget);
            let t2 = loom_thread::spawn(move || {
                if b2.try_reserve(50) {
                    let _guard = LoomCodegenGuard {
                        budget: LoomArc::clone(&b2),
                        units: 50,
                        committed: false,
                    };
                    loom_thread::yield_now();
                }
            });

            t1.join().unwrap();
            t2.join().unwrap();

            let remaining = budget.remaining.load(LoomOrdering::Acquire);
            assert!(
                remaining == 0 || remaining == 50,
                "remaining must be either fully committed (0) or fully restored (50), got {remaining}"
            );
        });
    }
}
