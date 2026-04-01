//! Post-RLM code generation orchestrator.
//!
//! Dispatches to Claude CLI or OpenCode CLI after the RLM analysis
//! produces `.lambda-rlm-result.md`. Both generators use the same
//! prompt format and output contract: read findings, fix code, commit,
//! output a summary line (or "CLEAN" if nothing actionable).

use anyhow::{Context, Result};
use std::io::Write;
use std::time::Instant;
use std::{
    path::Path,
    process::Stdio,
    sync::atomic::{AtomicU64, Ordering},
    sync::Arc,
    sync::OnceLock,
    time::SystemTime,
};
use tokio::io::{AsyncRead, AsyncReadExt};
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
const CLAUDE_TIMEOUT: Duration = Duration::from_secs(30);
const OPENCODE_TIMEOUT: Duration = Duration::from_secs(45);
static CODEGEN_GUARD_LIVE_COUNT: AtomicU64 = AtomicU64::new(0);

pub fn codegen_guard_live_count() -> u64 {
    CODEGEN_GUARD_LIVE_COUNT.load(Ordering::Acquire)
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
        CODEGEN_GUARD_LIVE_COUNT.fetch_add(1, Ordering::AcqRel);
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
        CODEGEN_GUARD_LIVE_COUNT.fetch_sub(1, Ordering::AcqRel);
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

fn codegen_circuit(generator: &CodeGenerator) -> Result<&'static CircuitBreaker> {
    static CLAUDE_CIRCUIT: OnceLock<CircuitBreaker> = OnceLock::new();
    // OpenCode is still less stable under sustained load in our testing, so it
    // uses a stricter threshold/longer cooldown to isolate faults faster.
    static OPENCODE_CIRCUIT: OnceLock<CircuitBreaker> = OnceLock::new();
    let circuit = match generator {
        CodeGenerator::Claude => CLAUDE_CIRCUIT
            .get_or_init(|| CircuitBreaker::new(CLAUDE_CB_THRESHOLD, CLAUDE_CB_COOLDOWN)),
        CodeGenerator::Opencode => OPENCODE_CIRCUIT
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
         4. Do a proper git commit(non-signed)\n\
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
         4. Do a proper git commit(non-signed)\n\
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
) -> Result<Arc<str>> {
    let mut budget_guard = CodegenBudgetGuard::acquire(oracle, CODEGEN_BUDGET_UNITS)?;
    let circuit = codegen_circuit(generator)?;
    anyhow::ensure!(
        circuit.allow_request(),
        "code generation temporarily unavailable"
    );

    let started = Instant::now();
    let run = match generator {
        CodeGenerator::Claude => run_claude(result, work_dir, question, iteration).await,
        CodeGenerator::Opencode => run_opencode(result, work_dir, question, iteration).await,
    };
    oracle.record_codegen_call(generator, started.elapsed());

    match run {
        Ok(summary) => {
            circuit.record_success();
            budget_guard.commit();
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

fn error_has_context(error: &anyhow::Error, marker: &str) -> bool {
    error
        .chain()
        .any(|cause| cause.to_string().contains(marker))
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
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)
        .with_context(|| format!("refusing unsafe result path: {}", path.display()))?;
    let meta = file.metadata()?;
    Ok((file, meta.len(), meta.dev(), meta.ino()))
}

#[cfg(not(unix))]
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
    if result_file.exists() {
        let (_existing, size, _dev, _ino) = open_file_no_follow(&result_file)?;
        anyhow::ensure!(
            size <= MAX_RESULT_BYTES as u64,
            "existing result file exceeds safety bound"
        );
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
            if let Ok(dir) = std::fs::File::open(&work_dir) {
                let _ = dir.sync_all();
            }
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
        .or_else(|| first_non_empty_line(text))
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
    anyhow::ensure!(
        version.trim_start().starts_with("opencode 1."),
        "OpenCode v1.x required, found: {}",
        version.trim()
    );
    Ok(())
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

        summary.map(Arc::from).ok_or_else(|| {
            anyhow::anyhow!("{self} output did not contain a non-empty summary line")
        })
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

struct ChildCleanup {
    child: Option<tokio::process::Child>,
}

impl ChildCleanup {
    fn new(child: tokio::process::Child) -> Self {
        Self { child: Some(child) }
    }

    fn child_mut(&mut self) -> Result<&mut tokio::process::Child> {
        self.child
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("child process handle missing"))
    }

    async fn kill_and_reap(&mut self) {
        if let Some(child) = self.child.as_mut() {
            let _ = child.start_kill();
            let _ = child.wait().await;
        }
        self.child = None;
    }

    fn disarm(&mut self) {
        self.child = None;
    }
}

impl Drop for ChildCleanup {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            if let Ok(handle) = tokio::runtime::Handle::try_current() {
                handle.spawn(async move {
                    let _ = child.start_kill();
                    let _ = child.wait().await;
                });
                return;
            }

            let _ = child.start_kill();
            let deadline = std::time::Instant::now() + Duration::from_secs(5);
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

fn classify_non_success_exit(
    generator_name: &str,
    status: std::process::ExitStatus,
) -> anyhow::Error {
    match status.code() {
        Some(127) => anyhow::anyhow!("{generator_name} command not found (exit 127)")
            .context("codegen_fatal"),
        Some(1) => anyhow::anyhow!("{generator_name} exited with retryable CLI error (exit 1)")
            .context("codegen_retryable"),
        Some(code) => {
            anyhow::anyhow!("{generator_name} exited with code {code}").context("codegen_retryable")
        }
        None => {
            anyhow::anyhow!("{generator_name} terminated by signal").context("codegen_retryable")
        }
    }
}

fn sanitize_generator_stderr(input: &str) -> String {
    let mut out = input.to_owned();
    for (name, value) in std::env::vars() {
        let looks_sensitive = name == "FIREWORKS_API" || name.starts_with("OPENCODE_");
        if looks_sensitive && !value.is_empty() {
            out = out.replace(&value, "[REDACTED]");
        }
    }
    out
}

async fn run_generator_process(
    mut cmd: tokio::process::Command,
    generator: &CodeGenerator,
) -> Result<GeneratorOutput> {
    let generator_name = match generator {
        CodeGenerator::Claude => "claude",
        CodeGenerator::Opencode => "opencode",
    };
    let timeout = match generator {
        CodeGenerator::Claude => CLAUDE_TIMEOUT,
        CodeGenerator::Opencode => OPENCODE_TIMEOUT,
    };
    let child = cmd.spawn().with_context(|| {
        format!("Failed to spawn `{generator_name}` — is it installed and on PATH?")
    })?;
    let mut child = ChildCleanup::new(child);

    let stdout = match child.child_mut()?.stdout.take() {
        Some(stdout) => stdout,
        None => {
            child.kill_and_reap().await;
            anyhow::bail!("failed to capture generator stdout");
        }
    };
    let stderr = match child.child_mut()?.stderr.take() {
        Some(stderr) => stderr,
        None => {
            child.kill_and_reap().await;
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

    let status = match tokio::time::timeout(timeout, child.child_mut()?.wait()).await {
        Ok(Ok(status)) => status,
        Ok(Err(error)) => {
            child.kill_and_reap().await;
            return Err(error).with_context(|| format!("Failed to run {generator_name}"));
        }
        Err(_) => {
            child.kill_and_reap().await;
            return Err(anyhow::anyhow!(
                "{generator_name} timed out after {}s",
                timeout.as_secs()
            ))
            .context("codegen_retryable");
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
            return Err(classify_non_success_exit("claude", output.status));
        }

        let summary = CodeGenerator::Claude.extract_result(&output.stdout)?;

        Ok(summary)
    }
    .await;

    remove_result_file(work_dir).await;
    cleanup_old_logs(work_dir, &CodeGenerator::Claude);
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
            return Err(classify_non_success_exit("opencode", output.status));
        }

        let summary = CodeGenerator::Opencode.extract_result(&output.stdout)?;

        Ok(summary)
    }
    .await;

    remove_result_file(work_dir).await;
    cleanup_old_logs(work_dir, &CodeGenerator::Opencode);
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
    use std::time::Duration as StdDuration;

    fn pid_alive(pid: u32) -> bool {
        std::process::Command::new("kill")
            .arg("-0")
            .arg(pid.to_string())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|status| status.success())
            .unwrap_or(false)
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
        let err = CodeGenerator::Opencode.extract_result(output).unwrap_err();
        assert!(
            err.to_string()
                .contains("did not contain a non-empty summary line"),
            "unexpected error: {err}"
        );
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
            std::process::Command::new("sh")
                .arg("-c")
                .arg("exit 1")
                .status()
                .expect("status"),
        );
        let fatal = classify_non_success_exit(
            "opencode",
            std::process::Command::new("sh")
                .arg("-c")
                .arg("exit 127")
                .status()
                .expect("status"),
        );

        assert!(error_has_context(&retryable, "codegen_retryable"));
        assert!(error_has_context(&fatal, "codegen_fatal"));
    }

    #[test]
    fn codegen_circuits_are_isolated_by_generator() {
        let claude = codegen_circuit(&CodeGenerator::Claude).expect("claude circuit");
        let opencode = codegen_circuit(&CodeGenerator::Opencode).expect("opencode circuit");

        claude.record_success();
        opencode.record_success();

        for _ in 0..OPENCODE_CB_THRESHOLD {
            opencode.record_failure();
        }

        assert!(!opencode.allow_request());
        assert!(claude.allow_request());

        claude.record_success();
        opencode.record_success();
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
                if !pid_alive(pid) {
                    break;
                }
                tokio::time::sleep(StdDuration::from_millis(50)).await;
            }
        })
        .await
        .expect("child process should be reaped within timeout");
    }
}
