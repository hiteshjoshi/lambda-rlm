//! Post-RLM code generation orchestrator.
//!
//! Dispatches to Claude CLI or OpenCode CLI after the RLM analysis
//! produces `.lambda-rlm-result.md`. Both generators use the same
//! prompt format and output contract: read findings, fix code, commit,
//! output a summary line (or "CLEAN" if nothing actionable).

use anyhow::{Context, Result};
use std::io::Write;
use std::time::Instant;
use std::{path::Path, process::Stdio, sync::OnceLock, time::SystemTime};
use tokio::io::AsyncReadExt;
use tokio::time::{timeout, Duration};

use crate::oracle::Oracle;
use crate::resilience::CircuitBreaker;
use crate::types::CodeGenerator;

const RESULT_FILE_NAME: &str = ".lambda-rlm-result.md";
const CLAUDE_TIMEOUT: Duration = Duration::from_secs(15 * 60);
const OPENCODE_TIMEOUT: Duration = Duration::from_secs(15 * 60);
const MAX_RESULT_BYTES: usize = 8 * 1024 * 1024;
const MAX_LOG_FILES_PER_GENERATOR: usize = 10;
const CODEGEN_BUDGET_UNITS: usize = 50;
const CLAUDE_CB_THRESHOLD: usize = 3;
const CLAUDE_CB_COOLDOWN: Duration = Duration::from_secs(60);
const OPENCODE_CB_THRESHOLD: usize = 2;
const OPENCODE_CB_COOLDOWN: Duration = Duration::from_secs(120);

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
        if !self.committed {
            if !std::thread::panicking() {
                tracing::error!(
                    units = self.units,
                    "CodegenBudgetGuard dropped without commit() — restoring reserved budget"
                );
            }
            self.oracle.budget_unreserve(self.units);
        }
    }
}

fn codegen_circuit(generator: &CodeGenerator) -> Result<&'static CircuitBreaker> {
    static CLAUDE_CIRCUIT: OnceLock<CircuitBreaker> = OnceLock::new();
    static OPENCODE_CIRCUIT: OnceLock<CircuitBreaker> = OnceLock::new();
    let circuit = match generator {
        CodeGenerator::Claude => {
            CLAUDE_CIRCUIT.get_or_init(|| CircuitBreaker::new(CLAUDE_CB_THRESHOLD, CLAUDE_CB_COOLDOWN))
        }
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
) -> Result<String> {
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
            circuit.record_failure();
            tracing::error!(generator = %generator, iteration, error = ?error, "code generation execution failed");
            Err(
                anyhow::anyhow!("code generation unavailable").context(match generator {
                    CodeGenerator::Claude => "claude_service_unavailable",
                    CodeGenerator::Opencode => "opencode_service_unavailable",
                }),
            )
        }
    }
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

async fn write_result_file_atomically(
    work_dir: &Path,
    result: &str,
    _iteration: usize,
) -> Result<()> {
    validate_analysis_result_content(result)?;

    let result_file = work_dir.join(RESULT_FILE_NAME);
    if let Ok(meta) = tokio::fs::symlink_metadata(&result_file).await {
        anyhow::ensure!(
            !meta.file_type().is_symlink(),
            "refusing to write through symlink result file"
        );
    }

    let work_dir = work_dir.to_path_buf();
    let result_file_clone = result_file.clone();
    let result_owned = result.to_owned();
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
        tmp.persist(&result_file_clone).map_err(|e| {
            anyhow::anyhow!(
                "Failed to move {}: {}",
                result_file_clone.display(),
                e.error
            )
        })?;
        Ok(())
    })
    .await
    .context("Result file writer task failed")??;

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

fn first_non_empty_line(text: &str) -> String {
    text.lines()
        .find(|line| !line.trim().is_empty())
        .unwrap_or("(no output)")
        .to_string()
}

fn last_non_empty_line(text: &str) -> String {
    text.lines()
        .rev()
        .find(|line| !line.trim().is_empty())
        .unwrap_or("(no output)")
        .to_string()
}

impl CodeGenerator {
    fn extract_result(&self, output: &[u8]) -> Result<String> {
        let stdout = std::str::from_utf8(output)
            .with_context(|| format!("{self} subprocess produced invalid UTF-8 on stdout"))?;
        validate_generator_output(self, &stdout)?;

        // Output-format invariant by generator:
        // - Claude streams progress/thinking first, then prints the actionable result last.
        // - OpenCode emits the concise actionable result first, then diagnostics.
        let summary = match self {
            CodeGenerator::Claude => last_non_empty_line(&stdout),
            CodeGenerator::Opencode => first_non_empty_line(&stdout),
        };

        Ok(summary)
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

async fn run_generator_process_with_timeout(
    mut cmd: tokio::process::Command,
    generator_name: &str,
    timeout_duration: Duration,
) -> Result<GeneratorOutput> {
    let mut child = cmd.spawn().with_context(|| {
        format!("Failed to spawn `{generator_name}` — is it installed and on PATH?")
    })?;

    let stdout = match child.stdout.take() {
        Some(stdout) => stdout,
        None => {
            let _ = child.start_kill();
            let _ = child.wait().await;
            anyhow::bail!("failed to capture generator stdout");
        }
    };
    let stderr = match child.stderr.take() {
        Some(stderr) => stderr,
        None => {
            let _ = child.start_kill();
            let _ = child.wait().await;
            anyhow::bail!("failed to capture generator stderr");
        }
    };

    let stdout_task = tokio::spawn(async move {
        let mut reader = stdout;
        let mut buf = Vec::new();
        reader.read_to_end(&mut buf).await?;
        Ok::<Vec<u8>, std::io::Error>(buf)
    });
    let stderr_task = tokio::spawn(async move {
        let mut reader = stderr;
        let mut buf = Vec::new();
        reader.read_to_end(&mut buf).await?;
        Ok::<Vec<u8>, std::io::Error>(buf)
    });

    let status = match timeout(timeout_duration, child.wait()).await {
        Ok(wait_result) => match wait_result {
            Ok(status) => status,
            Err(error) => {
                let _ = child.start_kill();
                let _ = child.wait().await;
                return Err(error).with_context(|| format!("Failed to run {generator_name}"));
            }
        },
        Err(_) => {
            let _ = child.start_kill();
            let _ = child.wait().await;
            let _ = stdout_task.await;
            let _ = stderr_task.await;
            anyhow::bail!("{generator_name} timed out after {timeout_duration:?}");
        }
    };

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
) -> Result<String> {
    write_result_file_atomically(work_dir, result, iteration).await?;

    let run = async {
        let log_file = work_dir.join(format!(".lambda-rlm-claude-{iteration}.log"));

        eprintln!(
            ">>> Iteration {iteration}: launching claude in {} ...",
            work_dir.display()
        );
        let prompt = build_prompt(question, iteration);

        let mut cmd = tokio::process::Command::new("claude");
        cmd.arg("--dangerously-skip-permissions")
            .arg("-p")
            .arg(&prompt);
        configure_generator_command(&mut cmd, work_dir);

        let output = run_generator_process_with_timeout(cmd, "claude", CLAUDE_TIMEOUT).await?;

        anyhow::ensure!(
            output.stdout.len() <= MAX_RESULT_BYTES,
            "claude output exceeds {} bytes",
            MAX_RESULT_BYTES
        );

        let summary = CodeGenerator::Claude.extract_result(&output.stdout)?;
        let stdout = std::str::from_utf8(&output.stdout)
            .context("claude subprocess produced invalid UTF-8 on stdout")?;

        // Save full output to log — warn on failure rather than swallowing
        if let Err(e) = tokio::fs::write(&log_file, stdout).await {
            tracing::warn!(path = %log_file.display(), error = %e, "failed to write iteration log");
        }

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            tracing::error!(status = %output.status, stderr = %stderr, "claude exited with failure status");
            anyhow::bail!("claude execution failed");
        }

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
) -> Result<String> {
    write_result_file_atomically(work_dir, result, iteration).await?;

    let run = async {
        let log_file = work_dir.join(format!(".lambda-rlm-opencode-{iteration}.log"));

        eprintln!(
            ">>> Iteration {iteration}: launching opencode in {} ...",
            work_dir.display()
        );
        let prompt = build_prompt(question, iteration);

        let mut cmd = tokio::process::Command::new("opencode");
        cmd.arg("run")
            .arg("--file")
            .arg(".lambda-rlm-result.md")
            .arg("--")
            .arg(&prompt);
        configure_generator_command(&mut cmd, work_dir);

        let output = run_generator_process_with_timeout(cmd, "opencode", OPENCODE_TIMEOUT).await?;

        anyhow::ensure!(
            output.stdout.len() <= MAX_RESULT_BYTES,
            "opencode output exceeds {} bytes",
            MAX_RESULT_BYTES
        );

        let summary = CodeGenerator::Opencode.extract_result(&output.stdout)?;
        let stdout = std::str::from_utf8(&output.stdout)
            .context("opencode subprocess produced invalid UTF-8 on stdout")?;

        if let Err(e) = tokio::fs::write(&log_file, stdout).await {
            tracing::warn!(path = %log_file.display(), error = %e, "failed to write iteration log");
        }

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            tracing::error!(status = %output.status, stderr = %stderr, "opencode exited with failure status");
            anyhow::bail!("opencode execution failed");
        }

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
    fn first_non_empty_line_extracts_first_signal() {
        let output = "\n\nFix applied\nMore details\n";
        assert_eq!(first_non_empty_line(output), "Fix applied");
    }

    #[test]
    fn last_non_empty_line_extracts_final_signal() {
        let output = "line one\n\nline two\n\n";
        assert_eq!(last_non_empty_line(output), "line two");
    }

    #[test]
    fn extract_result_uses_last_line_for_claude() {
        let output = b"thinking...\n\nfix done\n";
        let summary = CodeGenerator::Claude
            .extract_result(output)
            .expect("summary");
        assert_eq!(summary, "fix done");
    }

    #[test]
    fn extract_result_uses_first_line_for_opencode() {
        let output = b"fix done\nmetadata\n";
        let summary = CodeGenerator::Opencode
            .extract_result(output)
            .expect("summary");
        assert_eq!(summary, "fix done");
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
}
