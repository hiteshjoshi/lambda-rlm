#![forbid(unsafe_code)]
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

mod codegen;
mod combinator;
mod cost;
mod oracle;
mod phi;
mod reduce;
mod resilience;
mod types;
mod verify;

use anyhow::{Context, Result};
use clap::{ArgGroup, Parser};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing_subscriber::EnvFilter;

/// Open a file with atomic symlink protection and return (File, size_bytes, inode).
/// Returns None if the file is a symlink or cannot be opened.
/// On Unix, uses O_NOFOLLOW to atomically reject symlinks.
/// Size is obtained via fstat on the open fd, eliminating the TOCTOU gap
/// between size check and read (an attacker cannot swap a small file for
/// a large one between check and open).
/// Returns (dev, ino) from fstat for caller-side inode verification.
#[cfg(unix)]
fn open_file_no_follow(path: &std::path::Path) -> Option<(std::fs::File, u64, u64, u64)> {
    use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
    let file = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)
        .ok()?;
    let meta = file.metadata().ok()?;
    Some((file, meta.len(), meta.dev(), meta.ino()))
}

/// On non-Unix (Windows), check symlink_metadata before opening to reject
/// symlinks/junctions. Not fully atomic like O_NOFOLLOW, but closes the
/// common case and prevents directory traversal via reparse points.
/// dev/ino returned as 0 — inode verification not available on Windows.
#[cfg(not(unix))]
fn open_file_no_follow(path: &std::path::Path) -> Option<(std::fs::File, u64, u64, u64)> {
    let meta = std::fs::symlink_metadata(path).ok()?;
    if meta.file_type().is_symlink() {
        return None;
    }
    let file = std::fs::File::open(path).ok()?;
    let size = file.metadata().ok()?.len();
    Some((file, size, 0, 0))
}

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
    about = "lambda-RLM v3: Hardened functional runtime for long-context reasoning (arXiv:2603.20105)",
    group = ArgGroup::new("code_generator").args(["claude", "opencode"]).multiple(false)
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

    /// Pipe final output into a new `claude` CLI session in the target directory
    #[arg(long, default_value = "false")]
    claude: bool,

    /// Pipe final output into a new `opencode` session in the target directory
    #[arg(long, default_value = "false")]
    opencode: bool,

    /// Max fix iterations when using --claude or --opencode (0 = unlimited)
    #[arg(long, default_value = "10")]
    max_iterations: usize,
}

/// Schema version for CLI configuration. Bump when adding/removing/renaming
/// parameters that change runtime behavior. Logged at startup alongside a
/// blake3 hash of effective values — any config drift across deployments
/// is observable in logs without requiring code changes.
const CONFIG_SCHEMA_VERSION: u32 = 1;

impl Cli {
    /// Fail fast on invalid parameter combinations before spending API budget.
    fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.k == 0 || self.k >= 2,
            "k must be 0 (auto) or >= 2; k=1 causes infinite recursion"
        );
        anyhow::ensure!(
            self.k <= 16,
            "k must be <= 16 to prevent 2^64 expansion; got {}",
            self.k
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
        anyhow::ensure!(
            !(self.claude && self.opencode),
            "--claude and --opencode are mutually exclusive; pick one code generator"
        );
        // Prevent pathological expansion: k^depth must stay under 100K total calls
        if self.k >= 2 {
            let max_safe_depth = ((100_000f64).ln() / (self.k as f64).ln()).floor() as usize;
            if max_safe_depth == 0 {
                anyhow::bail!("k={} is too large for any recursion depth", self.k);
            }
        }
        Ok(())
    }

    /// Compute a blake3 fingerprint of the effective configuration.
    /// Logged at startup so operators can detect config drift across
    /// deployments without comparing individual env var values.
    fn config_fingerprint(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&CONFIG_SCHEMA_VERSION.to_le_bytes());
        hasher.update(self.model.as_bytes());
        hasher.update(&(self.window as u64).to_le_bytes());
        hasher.update(&(self.k as u64).to_le_bytes());
        hasher.update(&(self.concurrency as u64).to_le_bytes());
        hasher.update(&self.rho.to_le_bytes());
        hasher.update(&self.alpha.to_le_bytes());
        hasher.update(&(self.overlap as u64).to_le_bytes());
        hasher.update(&(self.context_window as u64).to_le_bytes());
        hasher.update(&self.max_tokens.to_le_bytes());
        hasher.update(&self.timeout.to_le_bytes());
        hasher.update(&(self.max_calls as u64).to_le_bytes());
        hasher.update(&(self.max_retries as u64).to_le_bytes());
        hasher.update(&(self.max_cache_entries as u64).to_le_bytes());
        hasher.update(&[self.no_cache as u8, self.dry_run as u8, self.opencode as u8]);
        let hash = hasher.finalize();
        format!("v{}:{}", CONFIG_SCHEMA_VERSION, &hash.to_hex()[..16])
    }
}

// ── File Collector ───────────────────────────────────────────────

/// PRE: path exists and is readable
/// POST: all returned content is from files strictly under the canonical root
///
/// Security: canonicalizes the root path and verifies every entry stays
/// within it. Symlinks that escape the root are skipped.
/// Maximum aggregate bytes to collect. Prevents heap exhaustion from
/// scanning enormous repositories or zip bombs before chunking begins.
/// 1GB accommodates large monorepos while protecting against pathological inputs.
const MAX_AGGREGATE_BYTES: u64 = 1024 * 1024 * 1024;

fn collect_source_files(path: &PathBuf) -> Result<String> {
    let extensions = [
        "rs", "ts", "js", "py", "svelte", "toml", "yaml", "yml", "json", "go", "java", "c", "cpp",
        "h", "hpp", "rb", "php", "swift", "kt", "sh", "sql", "tf", "hcl",
    ];

    let canonical_root = path
        .canonicalize()
        .with_context(|| format!("Cannot resolve path: {}", path.display()))?;

    if canonical_root.is_file() {
        let meta = std::fs::metadata(&canonical_root)
            .with_context(|| format!("Cannot stat {}", canonical_root.display()))?;
        if meta.len() > MAX_AGGREGATE_BYTES {
            anyhow::bail!(
                "Single file {} bytes exceeds {} byte aggregate limit",
                meta.len(),
                MAX_AGGREGATE_BYTES
            );
        }
        let content = std::fs::read_to_string(&canonical_root)
            .with_context(|| format!("Failed to read {}", canonical_root.display()))?;
        return Ok(format!("// === {} ===\n{}", path.display(), content));
    }

    let mut all_code = String::new();
    let mut file_count = 0usize;
    let mut skipped = 0usize;
    let mut accumulated_bytes: u64 = 0;

    // Manual stack-based directory walk (replaces walkdir crate, -10 transitive deps).
    let skip_dirs: &[&str] = &[
        "node_modules",
        "target",
        "dist",
        "build",
        "__pycache__",
        ".git",
        "vendor",
    ];
    let mut stack: Vec<PathBuf> = vec![canonical_root.clone()];
    while let Some(dir) = stack.pop() {
        let read_dir = match std::fs::read_dir(&dir) {
            Ok(rd) => rd,
            Err(_) => continue,
        };
        for entry in read_dir.filter_map(|e| e.ok()) {
            let file_path = entry.path();
            let name = entry.file_name();
            let name_str = name.to_string_lossy();

            // Skip hidden dirs/files and known non-source directories
            if name_str.starts_with('.') || skip_dirs.contains(&name_str.as_ref()) {
                continue;
            }

            // Skip symlinks — prevents traversal attacks
            let meta = match std::fs::symlink_metadata(&file_path) {
                Ok(m) => m,
                Err(_) => continue,
            };
            if meta.file_type().is_symlink() {
                skipped += 1;
                continue;
            }

            if meta.is_dir() {
                // Path traversal protection: verify resolved path stays under root
                if let Ok(canonical_dir) = file_path.canonicalize() {
                    if canonical_dir.starts_with(&canonical_root) {
                        stack.push(file_path);
                    } else {
                        tracing::warn!(
                            path = %file_path.display(),
                            resolved = %canonical_dir.display(),
                            "skipping directory that escapes root"
                        );
                        skipped += 1;
                    }
                }
                continue;
            }

            if !meta.is_file() {
                continue;
            }
            let ext = file_path.extension().and_then(|e| e.to_str()).unwrap_or("");
            if !extensions.contains(&ext) {
                continue;
            }

            // Path traversal protection for files
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

            // Open with O_NOFOLLOW, fstat the open fd, check size, then read.
            if let Some((mut file, size, open_dev, open_ino)) = open_file_no_follow(&file_path) {
                // TOCTOU inode check: verify the opened file is the same inode
                // the directory walker saw.
                #[cfg(unix)]
                {
                    use std::os::unix::fs::MetadataExt;
                    if let Ok(walk_meta) = std::fs::symlink_metadata(&file_path) {
                        if walk_meta.dev() != open_dev || walk_meta.ino() != open_ino {
                            tracing::warn!(
                                path = %file_path.display(),
                                walk_ino = walk_meta.ino(),
                                open_ino,
                                "inode mismatch: file swapped between readdir and open, skipping"
                            );
                            skipped += 1;
                            continue;
                        }
                    }
                }
                #[cfg(not(unix))]
                let _ = (open_dev, open_ino);

                accumulated_bytes = accumulated_bytes.checked_add(size).with_context(|| {
                    format!(
                        "Aggregate size overflow while scanning {}",
                        file_path.display()
                    )
                })?;
                if accumulated_bytes > MAX_AGGREGATE_BYTES {
                    anyhow::bail!(
                        "Aggregate source size {} bytes exceeds {} byte limit after {} files",
                        accumulated_bytes,
                        MAX_AGGREGATE_BYTES,
                        file_count
                    );
                }
                use std::io::Read;
                let mut content = String::new();
                if file.read_to_string(&mut content).is_ok() {
                    all_code.push_str(&format!("\n// === {} ===\n", file_path.display()));
                    all_code.push_str(&content);
                    all_code.push('\n');
                    file_count += 1;
                }
            }
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

// ── Analysis runner (reusable per iteration) ────────────────────

async fn run_analysis(cli: &Cli, oracle: &Arc<Oracle>) -> Result<String> {
    let start = Instant::now();

    // ── Phase 1: REPL Initialization ──
    let prompt = collect_source_files(&cli.path)?;
    if prompt.is_empty() {
        anyhow::bail!("No source files found in {}", cli.path.display());
    }
    eprintln!(
        "Phase 1: P stored externally ({} chars), library L loaded",
        prompt.len()
    );

    // ── Phase 2: Task Detection ──
    let task = if cli.task == TaskType::Auto {
        eprintln!("Phase 2: Auto-detecting task type...");
        let preview = comb_peek(&prompt, 0, cli.peek_size);
        auto_detect_task(preview, prompt.len(), &cli.question, oracle).await?
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
        &cli.model,
        cli.context_window,
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

    // Shutdown channel for cooperative cancellation.
    // shutdown_tx is held by the signal handler installed in main().
    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
    // Leak-safe: clone into a background task that listens for SIGINT/SIGTERM.
    // When signalled, it broadcasts shutdown to all phi() recursions.
    {
        let tx = shutdown_tx.clone();
        let oracle_shutdown = Arc::clone(&oracle);
        tokio::spawn(async move {
            let ctrl_c = tokio::signal::ctrl_c();
            #[cfg(unix)]
            {
                use tokio::signal::unix::{signal, SignalKind};
                let mut sigterm =
                    signal(SignalKind::terminate()).expect("failed to install SIGTERM handler");
                // SIGHUP: allows systemd/kubernetes to trigger clean restart
                // with new env vars (config reload) without SIGKILL. phi()
                // drains in-flight work via the shutdown channel before exit.
                let mut sighup =
                    signal(SignalKind::hangup()).expect("failed to install SIGHUP handler");
                tokio::select! {
                    _ = ctrl_c => {},
                    _ = sigterm.recv() => {},
                    _ = sighup.recv() => {
                        eprintln!("\n>>> SIGHUP received — reloading requires restart. Draining in-flight work...");
                    },
                }
            }
            #[cfg(not(unix))]
            {
                let _ = ctrl_c.await;
            }
            eprintln!("\n>>> Signal received, initiating graceful shutdown (30s drain)...");
            // Signal Oracle to stop accepting new API calls (prevents billing waste)
            oracle_shutdown.trigger_shutdown();
            let _ = tx.send(true);
        });
    }
    // Drop our copy of shutdown_tx — only the signal task holds it now.
    // When all receivers see `true`, phi() trees will abort cooperatively.
    drop(shutdown_tx);

    // Bounded concurrency: prevents OOM on wide/deep recursion trees.
    // Default 100 permits limits total in-flight phi() calls across all depths.
    let concurrency_semaphore = Arc::new(tokio::sync::Semaphore::new(100));

    // Generate trace_id for correlating logs across the entire recursion tree.
    // blake3(question + timestamp) truncated to 16 bytes — unique per invocation.
    let trace_id: [u8; 16] = {
        let mut hasher = blake3::Hasher::new();
        hasher.update(cli.question.as_bytes());
        hasher.update(
            &std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
                .to_le_bytes(),
        );
        hasher.update(&(std::process::id() as u64).to_le_bytes());
        let hash = hasher.finalize();
        let mut id = [0u8; 16];
        id.copy_from_slice(&hash.as_bytes()[..16]);
        id
    };
    eprintln!("  Trace ID: {}", hex::encode(trace_id));

    let cfg = Arc::new(PhiConfig {
        question: cli.question.clone(),
        task,
        tau: p.tau,
        k: p.k,
        max_depth: p.depth,
        overlap: cli.overlap,
        max_tokens: cli.max_tokens,
        keywords,
        oracle: Arc::clone(oracle),
        verifier,
        shutdown: shutdown_rx,
        concurrency: concurrency_semaphore,
        trace_id,
    });

    // ── Phase 5: Execute Φ ──
    // Admission control: reject work that cannot complete given available budget.
    // Calculate worst-case cost (all leaves at max depth) and compare to budget.
    // This prevents spawning thousands of tasks that exhaust budget mid-tree,
    // wasting CPU and leaving zombie tasks.
    if !oracle.budget_unlimited() {
        let worst_case = p.total_calls;
        let available = oracle.budget_remaining();
        if worst_case > available {
            anyhow::bail!(
                "Admission declined: worst-case execution needs {} calls but only {} budget remaining. \
                 Reduce input size, increase --max-calls, or use a smaller k/depth.",
                worst_case, available
            );
        }
    }

    eprintln!("Phase 5: Executing Phi...\n");
    let result = phi(cfg, prompt, 0, None).await?;

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

fn build_oracle(cli: &Cli) -> Result<Arc<Oracle>> {
    let api_key = if cli.dry_run {
        "dry-run".into()
    } else {
        std::env::var("FIREWORKS_API").context("Set FIREWORKS_API env var")?
    };

    Ok(Oracle::new(
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
    ))
}

fn validate_codegen_result_target(work_dir: &Path) -> Result<()> {
    let target = work_dir.join(".lambda-rlm-result.md");
    if target.exists() {
        anyhow::ensure!(
            open_file_no_follow(&target).is_some(),
            "refusing codegen result target symlink: {}",
            target.display()
        );
    }
    Ok(())
}

// ── Main — Algorithm 1: Complete λ-RLM System ───────────────────

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize structured logging with non-blocking writer.
    // Non-blocking prevents disk-full or slow disks from stalling the async
    // runtime — logs are dropped (with a counter) rather than blocking tasks.
    // The _guard must live for the duration of main() to flush on exit.
    let (non_blocking, _guard) = tracing_appender::non_blocking(std::io::stderr());
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn")),
        )
        .with_target(false)
        .with_writer(non_blocking)
        .init();

    let cli = Cli::parse();
    cli.validate()?;

    eprintln!("\n================================================================");
    eprintln!("  lambda-RLM v3: Hardened Functional Runtime for Long-Context Reasoning");
    eprintln!("  (arXiv:2603.20105 -- Roy et al., 2026)");
    eprintln!("  config fingerprint: {}", cli.config_fingerprint());
    eprintln!("================================================================\n");

    // Resolve effective code generator from --claude / --opencode flags
    let generator = if cli.claude {
        Some(types::CodeGenerator::Claude)
    } else if cli.opencode {
        Some(types::CodeGenerator::Opencode)
    } else {
        None
    };

    if generator.is_none() {
        // Single-shot mode: analyze and print
        let oracle = build_oracle(&cli)?;
        let result = run_analysis(&cli, &oracle).await?;
        println!("{result}");
        return Ok(());
    }
    let generator = generator.unwrap();

    // ── Fix loop mode ──
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

        let oracle = build_oracle(&cli)?;

        // 1. Analyze
        let result = run_analysis(&cli, &oracle).await?;

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

        // 3. Hand to code generator
        validate_codegen_result_target(&work_dir)?;
        let summary = codegen::run_code_generator(
            oracle.as_ref(),
            &generator,
            &result,
            &work_dir,
            &cli.question,
            iteration,
        )
        .await?;

        oracle.print_telemetry();

        let trimmed: String = summary.chars().take(200).collect();
        eprintln!(">>> Iteration {iteration} done: {trimmed}");
        eprintln!(
            "    (full log: {})",
            codegen::log_file_name(&generator, iteration)
        );
    }

    eprintln!(
        "\n>>> Reached max iterations ({}) with {}. Stopping loop.",
        cli.max_iterations, generator
    );
    eprintln!(">>> Run again to continue if needed.");
    Ok(())
}
