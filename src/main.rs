//! λ-RLM v3: Hardened, modular implementation of the λ-RLM framework.
//! Paper: "The Y-Combinator for LLMs: Solving Long-Context Rot with λ-Calculus"
//! (Roy et al., arXiv:2603.20105, March 2026)
//!
//! v3 over v2:
//!   - Verifier trait: task-specific validation at every leaf (replaces filter_useful)
//!   - Structural chunking: splits at definition boundaries, not raw line counts
//!   - RAII budget guards: panic-safe budget management via Drop
//!   - Improved quorum: normalized trigram similarity (structural, not word Jaccard)
//!   - Deterministic plan cache: compute_plan results are content-addressed
//!   - Module split: types, resilience, oracle, cost, combinator, verify, reduce, phi
//!   - Contract comments: PRE/POST invariants on critical functions
//!
//! Usage:
//!   lambda_rlm -p ./src -q "Find security vulnerabilities" -t aggregate
//!   lambda_rlm -p ./src -q "Where is auth handled?" -t search
//!   lambda_rlm -p ./src -q "Summarize this codebase" -t summarise
//!   lambda_rlm -p ./src -q "Classify each module" -t classify
//!   lambda_rlm -p ./src -q "Find duplicate logic" -t pairwise
//!   lambda_rlm -p ./src -q "Trace auth flow end to end" -t multi-hop
//!   lambda_rlm -p ./src -q "Any question"  # auto-detect task type
//!
//! Set FIREWORKS_API in your environment.

mod combinator;
mod cost;
mod oracle;
mod phi;
mod reduce;
mod resilience;
mod types;
mod verify;

use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing_subscriber::EnvFilter;
use walkdir::WalkDir;

use combinator::{comb_peek, extract_keywords};
use cost::{composition_desc, compute_plan, pipeline_desc, CostModel};
use oracle::Oracle;
use phi::{auto_detect_task, phi, PhiConfig};
use types::TaskType;
use verify::Verifier;

// ── CLI ──────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(
    name = "lambda_rlm",
    about = "lambda-RLM v3: Hardened functional runtime for long-context reasoning (arXiv:2603.20105)"
)]
struct Cli {
    /// Path to scan (file or directory)
    #[arg(short, long)]
    path: PathBuf,

    /// Question or task description
    #[arg(short, long)]
    question: String,

    /// Task type (auto-detected if omitted)
    #[arg(short, long, default_value = "auto")]
    task: TaskType,

    /// Leaf chunk threshold τ in chars (0 = auto from cost model)
    #[arg(short, long, default_value = "6000")]
    window: usize,

    /// Split factor k (0 = auto-compute optimal k* via Theorem 4)
    #[arg(short, long, default_value = "0")]
    k: usize,

    /// Max parallel LLM calls
    #[arg(long, default_value = "8")]
    concurrency: usize,

    /// Context-rot decay factor ρ ∈ (0, 1] (Definition 4)
    #[arg(long, default_value = "0.85")]
    rho: f64,

    /// Accuracy target α ∈ (0, 1] for optimal planner
    #[arg(long, default_value = "0.8")]
    alpha: f64,

    /// Overlap chars δ for multi-hop splitting
    #[arg(long, default_value = "200")]
    overlap: usize,

    /// Model context window K in chars (~4 chars/token)
    #[arg(long, default_value = "32000")]
    context_window: usize,

    /// Peek size for auto-detection (Phase 2)
    #[arg(long, default_value = "500")]
    peek_size: usize,

    /// Max output tokens for the final (depth=0) answer
    #[arg(long, default_value = "8192")]
    max_tokens: u32,

    /// Fireworks model ID
    #[arg(long, default_value = "accounts/fireworks/routers/kimi-k2p5-turbo")]
    model: String,

    /// Dry run — no API calls
    #[arg(long, default_value = "false")]
    dry_run: bool,

    /// Per-call timeout in seconds
    #[arg(long, default_value = "120")]
    timeout: u64,

    /// Max total LLM calls (0 = unlimited)
    #[arg(long, default_value = "0")]
    max_calls: usize,

    /// Cache directory for deterministic replay
    #[arg(long, default_value = ".lambda-rlm-cache")]
    cache_dir: PathBuf,

    /// Disable replay cache
    #[arg(long, default_value = "false")]
    no_cache: bool,

    /// Max cache entries before eviction (0 = unlimited)
    #[arg(long, default_value = "10000")]
    max_cache_entries: usize,

    /// Max retries per API call
    #[arg(long, default_value = "2")]
    max_retries: usize,

    /// Enable quorum consensus for critical reduce paths (3x calls)
    #[arg(long, default_value = "false")]
    quorum: bool,

    /// Pipe final output into a new `claude` CLI session in the target directory
    #[arg(long, default_value = "false")]
    claude: bool,

    /// Max fix iterations when using --claude (0 = unlimited)
    #[arg(long, default_value = "10")]
    max_iterations: usize,
}

impl Cli {
    /// Fail fast on invalid parameter combinations before spending API budget.
    fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.k == 0 || self.k >= 2,
            "k must be 0 (auto) or >= 2; k=1 causes infinite recursion"
        );
        anyhow::ensure!(
            self.alpha > 0.0 && self.alpha <= 1.0,
            "alpha must be in (0, 1]; got {}",
            self.alpha
        );
        anyhow::ensure!(
            self.rho > 0.0 && self.rho <= 1.0,
            "rho must be in (0, 1]; got {}",
            self.rho
        );
        anyhow::ensure!(self.concurrency >= 1, "concurrency must be >= 1");
        anyhow::ensure!(self.timeout > 0, "timeout must be > 0");
        Ok(())
    }
}

// ── File Collector ───────────────────────────────────────────────

/// PRE: path exists and is readable
/// POST: all returned content is from files strictly under the canonical root
///
/// Security: canonicalizes the root path and verifies every entry stays
/// within it. Symlinks that escape the root are skipped.
fn collect_source_files(path: &PathBuf) -> Result<String> {
    let extensions = [
        "rs", "ts", "js", "py", "svelte", "toml", "yaml", "yml", "json", "go", "java", "c", "cpp",
        "h", "hpp", "rb", "php", "swift", "kt", "sh", "sql", "tf", "hcl",
    ];

    let canonical_root = path
        .canonicalize()
        .with_context(|| format!("Cannot resolve path: {}", path.display()))?;

    if canonical_root.is_file() {
        let content = std::fs::read_to_string(&canonical_root)
            .with_context(|| format!("Failed to read {}", canonical_root.display()))?;
        return Ok(format!("// === {} ===\n{}", path.display(), content));
    }

    let mut all_code = String::new();
    let mut file_count = 0usize;
    let mut skipped = 0usize;

    for entry in WalkDir::new(&canonical_root)
        .into_iter()
        .filter_entry(|e| {
            let name = e.file_name().to_string_lossy();
            !name.starts_with('.')
                && name != "node_modules"
                && name != "target"
                && name != "dist"
                && name != "build"
                && name != "__pycache__"
                && name != ".git"
                && name != "vendor"
        })
        .filter_map(|e| e.ok())
    {
        let file_path = entry.path();

        // Skip symlinks — prevents traversal attacks
        if entry.path_is_symlink() {
            skipped += 1;
            continue;
        }

        // Path traversal protection: verify resolved path stays under root
        if let Ok(canonical_file) = file_path.canonicalize() {
            if !canonical_file.starts_with(&canonical_root) {
                tracing::warn!(
                    path = %file_path.display(),
                    resolved = %canonical_file.display(),
                    "skipping path that escapes root"
                );
                skipped += 1;
                continue;
            }
        }

        if !file_path.is_file() {
            continue;
        }
        let ext = file_path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if !extensions.contains(&ext) {
            continue;
        }
        if let Ok(content) = std::fs::read_to_string(file_path) {
            all_code.push_str(&format!("\n// === {} ===\n", file_path.display()));
            all_code.push_str(&content);
            all_code.push('\n');
            file_count += 1;
        }
    }

    if skipped > 0 {
        tracing::info!(
            count = skipped,
            "skipped symlinks/traversals during collection"
        );
    }
    eprintln!("Collected {file_count} files ({} chars)", all_code.len());
    Ok(all_code)
}

// ── Output dispatch ─────────────────────────────────────────────

/// Spawn claude in print mode — it applies fixes and exits.
/// Returns Claude's output summary (first line).
fn run_claude(
    result: &str,
    work_dir: &std::path::Path,
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
    let prompt = format!(
        "Read the analysis in .lambda-rlm-result.md — it's iteration {iteration} of a λ-RLM \
         fix loop for the question: \"{question}\".\n\n\
         Your job:\n\
         1. Read the findings carefully.\n\
         2. Act on every actionable item — fix bugs, refactor code, add missing pieces.\n\
         3. When done, output a single line summary of what you changed.\n\
         4. Do a proper git commit(non-signed)
         5. Update readme with commit id and change-log
         5. If there is nothing actionable (only informational notes or the analysis is clean), \
            output exactly: CLEAN\n\n\
         Start by reading .lambda-rlm-result.md now.",
    );
    let output = Command::new("claude")
        .arg("--dangerously-skip-permissions")
        .arg("-p")
        .arg(&prompt)
        .current_dir(work_dir)
        .output()
        .context("Failed to spawn `claude` — is it installed and on PATH?")?;

    let _ = std::fs::remove_file(&result_file);

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();

    // Save full output to log
    let _ = std::fs::write(&log_file, &stdout);

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

// ── Analysis runner (reusable per iteration) ────────────────────

async fn run_analysis(cli: &Cli) -> Result<String> {
    let start = Instant::now();

    let api_key = if cli.dry_run {
        "dry-run".into()
    } else {
        std::env::var("FIREWORKS_API").context("Set FIREWORKS_API env var")?
    };

    // ── Phase 1: REPL Initialization ──
    let prompt = collect_source_files(&cli.path)?;
    if prompt.is_empty() {
        anyhow::bail!("No source files found in {}", cli.path.display());
    }
    let oracle = Oracle::new(
        api_key,
        cli.model.clone(),
        cli.dry_run,
        cli.concurrency,
        Duration::from_secs(cli.timeout),
        cli.max_retries,
        cli.max_calls,
        cli.cache_dir.clone(),
        !cli.no_cache,
        cli.max_cache_entries,
    );
    eprintln!(
        "Phase 1: P stored externally ({} chars), library L loaded",
        prompt.len()
    );

    // ── Phase 2: Task Detection ──
    let task = if cli.task == TaskType::Auto {
        eprintln!("Phase 2: Auto-detecting task type...");
        let preview = comb_peek(&prompt, 0, cli.peek_size);
        auto_detect_task(preview, prompt.len(), &cli.question, &oracle).await?
    } else {
        eprintln!("Phase 2: Task type = {} (user-specified)", cli.task);
        cli.task.clone()
    };

    // ── Phase 3: Dispatch (direct if |P| ≤ K) ──
    if prompt.len() <= cli.window {
        eprintln!(
            "Phase 3: |P| = {} <= tau = {}, direct dispatch (no recursion)\n",
            prompt.len(),
            cli.window
        );
        let keywords = extract_keywords(&cli.question);
        let verifier = Verifier::new(task.clone(), keywords);
        let (sys, user) = phi::leaf_prompt_public(&prompt, &cli.question, &task);
        let raw = oracle.call(&sys, &user, cli.max_tokens).await?;
        let result = match verifier.check(&raw, prompt.len()) {
            verify::VerifyResult::Accept(v) | verify::VerifyResult::Degraded(v, _) => v,
            verify::VerifyResult::Reject(reason) => format!("No useful output ({reason})"),
        };

        let elapsed = start.elapsed();
        eprintln!(
            "  DONE -- {} M call(s) in {:.1}s",
            oracle.calls(),
            elapsed.as_secs_f64()
        );
        oracle.print_telemetry();
        return Ok(result);
    }

    // ── Phase 4: Planning (Theorem 4) ──
    let cost_model = CostModel::new(cli.rho, cli.context_window);
    let plan_cache = if cli.no_cache {
        None
    } else {
        Some(oracle.cache())
    };
    let p = compute_plan(
        prompt.len(),
        &task,
        &cost_model,
        cli.alpha,
        cli.k,
        cli.window,
        plan_cache,
    );

    eprintln!("Phase 4: Execution Plan");
    eprintln!("  Task:        {task}");
    eprintln!("  Pipeline:    {}", pipeline_desc(&task));
    eprintln!("  Compose:     {}", composition_desc(&task));
    eprintln!(
        "  Input:       {} chars, tau={}  k={}  depth={}",
        prompt.len(),
        p.tau,
        p.k,
        p.depth
    );
    eprintln!(
        "  Calls:       {} leaf + {} reduce = {} total",
        p.leaf_calls, p.reduce_calls, p.total_calls
    );
    eprintln!("  Concurrency: {} max parallel\n", cli.concurrency);

    let keywords = extract_keywords(&cli.question);
    let verifier = Verifier::new(task.clone(), keywords.clone());

    // Shutdown channel for cooperative cancellation
    let (_shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);

    let cfg = Arc::new(PhiConfig {
        question: cli.question.clone(),
        task,
        tau: p.tau,
        k: p.k,
        max_depth: p.depth,
        overlap: cli.overlap,
        max_tokens: cli.max_tokens,
        keywords,
        oracle: oracle.clone(),
        use_quorum: cli.quorum,
        verifier,
        shutdown: shutdown_rx,
    });

    // ── Phase 5: Execute Φ ──
    eprintln!("Phase 5: Executing Phi...\n");
    let result = phi(cfg, prompt, 0).await?;

    let elapsed = start.elapsed();
    eprintln!(
        "  DONE -- {} M call(s) in {:.1}s ({:.1} calls/sec)",
        oracle.calls(),
        elapsed.as_secs_f64(),
        oracle.calls() as f64 / elapsed.as_secs_f64().max(0.001)
    );
    oracle.print_telemetry();
    Ok(result)
}

// ── Main — Algorithm 1: Complete λ-RLM System ───────────────────

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize structured logging (controlled by RUST_LOG env var)
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn")),
        )
        .with_target(false)
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();
    cli.validate()?;

    eprintln!("\n================================================================");
    eprintln!("  lambda-RLM v3: Hardened Functional Runtime for Long-Context Reasoning");
    eprintln!("  (arXiv:2603.20105 -- Roy et al., 2026)");
    eprintln!("================================================================\n");

    if !cli.claude {
        // Single-shot mode: analyze and print
        let result = run_analysis(&cli).await?;
        println!("{result}");
        return Ok(());
    }

    // ── Claude loop mode ──
    let work_dir = if cli.path.is_file() {
        std::fs::canonicalize(cli.path.parent().unwrap_or(&cli.path))?
    } else {
        std::fs::canonicalize(&cli.path)?
    };

    let max_iter = if cli.max_iterations == 0 {
        usize::MAX
    } else {
        cli.max_iterations
    };

    for iteration in 1..=max_iter {
        eprintln!("\n================================================================");
        if cli.max_iterations == 0 {
            eprintln!("  ITERATION {iteration} (unlimited)");
        } else {
            eprintln!("  ITERATION {iteration}/{}", cli.max_iterations);
        }
        eprintln!("================================================================\n");

        // 1. Analyze
        let result = run_analysis(&cli).await?;

        // 2. Check if clean (heuristic: very short output or known "no issues" patterns)
        let lower = result.trim().to_lowercase();
        let is_clean = lower == "no_match"
            || lower == "no_data"
            || lower.starts_with("no relevant")
            || lower.starts_with("no items found")
            || lower.starts_with("no useful output");

        if is_clean {
            eprintln!("\n>>> Analysis came back clean. Nothing to fix.");
            println!("{result}");
            return Ok(());
        }

        // 3. Hand to claude
        let summary = run_claude(&result, &work_dir, &cli.question, iteration)?;

        let trimmed: String = summary.chars().take(200).collect();
        eprintln!(">>> Iteration {iteration} done: {trimmed}");
        eprintln!("    (full log: .lambda-rlm-claude-{iteration}.log)");
    }

    eprintln!(
        "\n>>> Reached max iterations ({}). Stopping loop.",
        cli.max_iterations
    );
    eprintln!(">>> Run again to continue if needed.");
    Ok(())
}
