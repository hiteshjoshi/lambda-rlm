//! Post-RLM code generation orchestrator.
//!
//! Dispatches to Claude CLI or OpenCode CLI after the RLM analysis
//! produces `.lambda-rlm-result.md`. Both generators use the same
//! prompt format and output contract: read findings, fix code, commit,
//! output a summary line (or "CLEAN" if nothing actionable).

use anyhow::{Context, Result};
use std::path::Path;

use crate::types::CodeGenerator;

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
    generator: &CodeGenerator,
    result: &str,
    work_dir: &Path,
    question: &str,
    iteration: usize,
) -> Result<String> {
    match generator {
        CodeGenerator::Claude => run_claude(result, work_dir, question, iteration).await,
        CodeGenerator::Opencode => run_opencode(result, work_dir, question, iteration).await,
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
    let result_file = work_dir.join(".lambda-rlm-result.md");
    let log_file = work_dir.join(format!(".lambda-rlm-claude-{iteration}.log"));
    std::fs::write(&result_file, result)
        .with_context(|| format!("Failed to write {}", result_file.display()))?;

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
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());

    let child = cmd
        .spawn()
        .context("Failed to spawn `claude` — is it installed and on PATH?")?;

    let output = child
        .wait_with_output()
        .await
        .context("Failed to run claude")?;

    let _ = std::fs::remove_file(&result_file);

    // Strict UTF-8: lossy conversion silently replaces invalid bytes with U+FFFD,
    // corrupting data that feeds downstream hashing and content analysis.
    let stdout = String::from_utf8(output.stdout)
        .context("claude subprocess produced invalid UTF-8 on stdout")?;

    // Save full output to log — warn on failure rather than swallowing
    if let Err(e) = std::fs::write(&log_file, &stdout) {
        tracing::warn!(path = %log_file.display(), error = %e, "failed to write iteration log");
    }

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        eprintln!("    claude stderr: {stderr}");
        anyhow::bail!("claude exited with {}", output.status);
    }

    // Print first meaningful line as confirmation
    let summary = stdout
        .lines()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("(no output)")
        .to_string();

    Ok(summary)
}

/// Spawn opencode and wait for it to finish (no timeout).
/// Returns OpenCode's output summary (first line).
///
/// PRE: work_dir exists and is writable
/// POST: opencode process has exited
async fn run_opencode(
    result: &str,
    work_dir: &Path,
    question: &str,
    iteration: usize,
) -> Result<String> {
    let result_file = work_dir.join(".lambda-rlm-result.md");
    let log_file = work_dir.join(format!(".lambda-rlm-opencode-{iteration}.log"));
    std::fs::write(&result_file, result)
        .with_context(|| format!("Failed to write {}", result_file.display()))?;

    eprintln!(
        ">>> Iteration {iteration}: launching opencode in {} ...",
        work_dir.display()
    );
    let prompt = build_prompt(question, iteration);

    let mut cmd = tokio::process::Command::new("opencode");
    cmd.arg("-p")
        .arg(&prompt)
        .current_dir(work_dir)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());

    let child = cmd
        .spawn()
        .context("Failed to spawn `opencode` — is it installed and on PATH?")?;

    let output = child
        .wait_with_output()
        .await
        .context("Failed to run opencode")?;

    let _ = std::fs::remove_file(&result_file);

    let stdout = String::from_utf8(output.stdout)
        .context("opencode subprocess produced invalid UTF-8 on stdout")?;

    if let Err(e) = std::fs::write(&log_file, &stdout) {
        tracing::warn!(path = %log_file.display(), error = %e, "failed to write iteration log");
    }

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        eprintln!("    opencode stderr: {stderr}");
        anyhow::bail!("opencode exited with {}", output.status);
    }

    let summary = stdout
        .lines()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("(no output)")
        .to_string();

    Ok(summary)
}

/// Log file name for the given generator and iteration.
pub fn log_file_name(generator: &CodeGenerator, iteration: usize) -> String {
    match generator {
        CodeGenerator::Claude => format!(".lambda-rlm-claude-{iteration}.log"),
        CodeGenerator::Opencode => format!(".lambda-rlm-opencode-{iteration}.log"),
    }
}
