//! Post-RLM code generation orchestrator.
//!
//! Dispatches to Claude CLI or OpenCode CLI after the RLM analysis
//! produces `.lambda-rlm-result.md`. Both generators use the same
//! prompt format and output contract: read findings, fix code, commit,
//! output a summary line (or "CLEAN" if nothing actionable).

use anyhow::{Context, Result};
use std::time::Instant;
use std::{path::Path, process::Stdio, sync::OnceLock, time::SystemTime};
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
const CODEGEN_CB_THRESHOLD: usize = 3;
const CODEGEN_CB_COOLDOWN: Duration = Duration::from_secs(60);

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
            self.oracle.budget_unreserve(self.units);
        }
    }
}

fn codegen_circuit() -> &'static CircuitBreaker {
    static CODEGEN_CIRCUIT: OnceLock<CircuitBreaker> = OnceLock::new();
    CODEGEN_CIRCUIT.get_or_init(|| CircuitBreaker::new(CODEGEN_CB_THRESHOLD, CODEGEN_CB_COOLDOWN))
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
    let circuit = codegen_circuit();
    anyhow::ensure!(
        circuit.allow_request(),
        "code generation circuit open (3 consecutive failures; cooling off 60s)"
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
            Err(anyhow::anyhow!("Code generation unavailable"))
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
    iteration: usize,
) -> Result<()> {
    validate_analysis_result_content(result)?;

    let result_file = work_dir.join(RESULT_FILE_NAME);
    if let Ok(meta) = tokio::fs::symlink_metadata(&result_file).await {
        anyhow::ensure!(
            !meta.file_type().is_symlink(),
            "refusing to write through symlink result file"
        );
    }

    let tmp_file = work_dir.join(format!(
        "{RESULT_FILE_NAME}.tmp.{}.{}",
        std::process::id(),
        iteration
    ));
    tokio::fs::write(&tmp_file, result)
        .await
        .with_context(|| format!("Failed to write {}", tmp_file.display()))?;

    tokio::fs::rename(&tmp_file, &result_file)
        .await
        .with_context(|| format!("Failed to move {}", result_file.display()))?;
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

/// Spawn claude in print mode and wait for it to finish (no timeout).
/// Returns Claude's output summary (first line).
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
            .arg(&prompt)
            .current_dir(work_dir)
            .stdin(Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true);

        #[cfg(unix)]
        {
            cmd.process_group(0);
        }

        let child = cmd
            .spawn()
            .context("Failed to spawn `claude` — is it installed and on PATH?")?;

        let output = match timeout(CLAUDE_TIMEOUT, child.wait_with_output()).await {
            Ok(output) => output.context("Failed to run claude")?,
            Err(_) => anyhow::bail!("claude timed out"),
        };

        anyhow::ensure!(
            output.stdout.len() <= MAX_RESULT_BYTES,
            "claude output exceeds {} bytes",
            MAX_RESULT_BYTES
        );

        // Strict UTF-8: lossy conversion silently replaces invalid bytes with U+FFFD,
        // corrupting data that feeds downstream hashing and content analysis.
        let stdout = String::from_utf8(output.stdout)
            .context("claude subprocess produced invalid UTF-8 on stdout")?;
        validate_generator_output(&CodeGenerator::Claude, &stdout)?;

        // Save full output to log — warn on failure rather than swallowing
        if let Err(e) = tokio::fs::write(&log_file, &stdout).await {
            tracing::warn!(path = %log_file.display(), error = %e, "failed to write iteration log");
        }

        if !output.status.success() {
            tracing::error!(status = %output.status, "claude exited with failure status");
            anyhow::bail!("claude execution failed");
        }

        let summary = first_non_empty_line(&stdout);

        Ok(summary)
    }
    .await;

    remove_result_file(work_dir).await;
    cleanup_old_logs(work_dir, &CodeGenerator::Claude);
    run
}

/// Spawn opencode in non-interactive mode and wait for it to finish.
/// Returns OpenCode's output summary (last non-empty line).
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
            .arg(&prompt)
            .current_dir(work_dir)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        #[cfg(unix)]
        {
            cmd.process_group(0);
        }

        let child = cmd
            .spawn()
            .context("Failed to spawn `opencode` — is it installed and on PATH?")?;

        let output = match timeout(OPENCODE_TIMEOUT, child.wait_with_output()).await {
            Ok(output) => output.context("Failed to run opencode")?,
            Err(_) => anyhow::bail!("opencode timed out"),
        };

        anyhow::ensure!(
            output.stdout.len() <= MAX_RESULT_BYTES,
            "opencode output exceeds {} bytes",
            MAX_RESULT_BYTES
        );

        let stdout = String::from_utf8(output.stdout)
            .context("opencode subprocess produced invalid UTF-8 on stdout")?;
        validate_generator_output(&CodeGenerator::Opencode, &stdout)?;

        if let Err(e) = tokio::fs::write(&log_file, &stdout).await {
            tracing::warn!(path = %log_file.display(), error = %e, "failed to write iteration log");
        }

        if !output.status.success() {
            tracing::error!(status = %output.status, "opencode exited with failure status");
            anyhow::bail!("opencode execution failed");
        }

        let summary = last_non_empty_line(&stdout);

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
}
