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

#[cfg(not(windows))]
#[global_allocator]
static GLOBAL_ALLOCATOR: jemallocator::Jemalloc = jemallocator::Jemalloc;

use anyhow::{Context, Result};
use clap::{ArgGroup, Parser};
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};
use tokio::sync::{watch, OwnedSemaphorePermit};
use tracing_subscriber::EnvFilter;

static INTERACTIVE_SESSION_SEMAPHORE: OnceLock<tokio::sync::Semaphore> = OnceLock::new();

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

fn validate_generator_access(generator: &types::CodeGenerator) -> Result<()> {
    let binary = match generator {
        types::CodeGenerator::Claude => "claude",
        types::CodeGenerator::Opencode => "opencode",
    };
    which::which(binary).with_context(|| format!("{binary} binary not in PATH"))?;
    if matches!(generator, types::CodeGenerator::Opencode) {
        codegen::validate_opencode_binary_version()?;
    }
    Ok(())
}

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

    /// Question or task prompt description
    #[arg(short, long, visible_alias = "prompt")]
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

    /// Run code generator in interactive TUI mode (requires --claude or --opencode)
    #[arg(long, default_value = "false", requires = "code_generator")]
    interactive: bool,

    /// Interactive startup timeout in minutes (1..=120)
    #[arg(
        long,
        default_value = "30",
        requires = "interactive",
        value_parser = parse_interactive_timeout_minutes
    )]
    interactive_timeout: u64,

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
        if self.interactive {
            anyhow::ensure!(
                self.claude || self.opencode,
                "--interactive requires one of --claude or --opencode"
            );
            anyhow::ensure!(
                std::io::stdin().is_terminal() && std::io::stdout().is_terminal(),
                "--interactive requires an attached TTY on stdin/stdout"
            );
        }
        if self.claude {
            validate_generator_access(&types::CodeGenerator::Claude)?;
        } else if self.opencode {
            validate_generator_access(&types::CodeGenerator::Opencode)?;
        }
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
        hasher.update(&[
            self.no_cache as u8,
            self.dry_run as u8,
            self.claude as u8,
            self.opencode as u8,
            self.interactive as u8,
        ]);
        hasher.update(&self.interactive_timeout.to_le_bytes());
        let hash = hasher.finalize();
        format!("v{}:{}", CONFIG_SCHEMA_VERSION, &hash.to_hex()[..16])
    }
}

fn parse_interactive_timeout_minutes(raw: &str) -> std::result::Result<u64, String> {
    let mins: u64 = raw
        .parse()
        .map_err(|_| "interactive startup timeout must be an integer".to_string())?;
    if (1..=120).contains(&mins) {
        Ok(mins)
    } else {
        Err("interactive startup timeout must be between 1 and 120 minutes".to_string())
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
/// 256MB aligns with Phi's hard input limit and prevents oversized ingest.
const MAX_AGGREGATE_BYTES: u64 = 256 * 1024 * 1024;
const MAX_OPEN_FILES: usize = 1024;
const CACHE_MAINTENANCE_INTERVAL_SECS: u64 = 15 * 60;
static FD_SEMAPHORE: OnceLock<Arc<tokio::sync::Semaphore>> = OnceLock::new();

#[must_use = "GuardedFile holds an FD permit until dropped"]
struct GuardedFile {
    file: Option<std::fs::File>,
    permit: Option<OwnedSemaphorePermit>,
}

impl GuardedFile {
    fn new(file: std::fs::File, permit: OwnedSemaphorePermit) -> Self {
        Self {
            file: Some(file),
            permit: Some(permit),
        }
    }

    fn file_mut(&mut self) -> Result<&mut std::fs::File> {
        self.file
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("guarded file already closed"))
    }
}

impl Drop for GuardedFile {
    fn drop(&mut self) {
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = self.permit.take();
            let _ = self.file.take();
        }));
    }
}

fn collect_source_files(path: &PathBuf) -> Result<String> {
    let extensions = [
        "rs", "ts", "js", "py", "svelte", "toml", "yaml", "yml", "json", "go", "java", "c", "cpp",
        "h", "hpp", "rb", "php", "swift", "kt", "sh", "sql", "tf", "hcl",
    ];

    let canonical_root = path
        .canonicalize()
        .with_context(|| format!("Cannot resolve path: {}", path.display()))?;

    if canonical_root.is_file() {
        let (_, file_size, _, _) = open_file_no_follow(&canonical_root)
            .ok_or_else(|| anyhow::anyhow!("Cannot safely open {}", canonical_root.display()))?;
        let oversized = file_size > MAX_AGGREGATE_BYTES;
        let mut out = format!("// === {} ===\n", path.display());
        let (mut file, _, _, _) = open_file_no_follow(&canonical_root)
            .ok_or_else(|| anyhow::anyhow!("Failed to open {}", canonical_root.display()))?;
        use std::io::Read;
        file.read_to_string(&mut out)
            .with_context(|| format!("Failed to read {}", canonical_root.display()))?;
        if oversized {
            let mut end = MAX_AGGREGATE_BYTES as usize;
            while end > 0 && !out.is_char_boundary(end) {
                end -= 1;
            }
            let omitted = out.len().saturating_sub(end);
            out.truncate(end);
            out.push_str(&format!("\n// [TRUNCATED: {omitted} bytes omitted]\n"));
            tracing::warn!(
                path = %canonical_root.display(),
                kept = end,
                omitted,
                "truncated oversized single-file input"
            );
        }
        return Ok(out);
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
            let fd_semaphore = Arc::clone(
                FD_SEMAPHORE.get_or_init(|| Arc::new(tokio::sync::Semaphore::new(MAX_OPEN_FILES))),
            );
            let fd_permit = match fd_semaphore.try_acquire_owned() {
                Ok(permit) => permit,
                Err(_) => {
                    tracing::warn!(
                        path = %file_path.display(),
                        "fd limit reached while collecting; skipping file"
                    );
                    skipped += 1;
                    continue;
                }
            };
            if let Some((file, size, open_dev, open_ino)) = open_file_no_follow(&file_path) {
                let mut guarded = GuardedFile::new(file, fd_permit);
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

                let next_total = accumulated_bytes.saturating_add(size);
                if next_total > MAX_AGGREGATE_BYTES {
                    let omitted = next_total - MAX_AGGREGATE_BYTES;
                    tracing::warn!(
                        bytes = accumulated_bytes,
                        omitted,
                        "truncating source collection at aggregate byte cap"
                    );
                    all_code.push_str(&format!("\n// [TRUNCATED: {omitted} bytes omitted]\n"));
                    break;
                }
                accumulated_bytes = next_total;
                use std::io::Read;
                let checkpoint = all_code.len();
                all_code.push_str(&format!("\n// === {} ===\n", file_path.display()));
                match guarded.file_mut()?.read_to_string(&mut all_code) {
                    Ok(_) => {
                        if !all_code.ends_with('\n') {
                            all_code.push('\n');
                        }
                        file_count += 1;
                    }
                    Err(error) => {
                        tracing::warn!(path = %file_path.display(), error = %error, "failed to read source file");
                        all_code.truncate(checkpoint);
                        skipped += 1;
                    }
                }
            }
        }
        if accumulated_bytes >= MAX_AGGREGATE_BYTES {
            break;
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

async fn run_analysis(
    cli: &Cli,
    oracle: &Arc<Oracle>,
    shutdown_rx: watch::Receiver<bool>,
) -> Result<String> {
    let start = Instant::now();

    // ── Phase 1: REPL Initialization ──
    let collect_path = cli.path.clone();
    let prompt = tokio::task::spawn_blocking(move || collect_source_files(&collect_path))
        .await
        .context("source collection task panicked")??;
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
        match auto_detect_task(preview, prompt.len(), &cli.question, oracle).await {
            Ok(task) => task,
            Err(error) => {
                if is_auto_detect_truncation_error(&error) {
                    tracing::warn!(
                        error = %error,
                        "auto-detect truncated; defaulting to summarise"
                    );
                    eprintln!(
                        "  Phase 2: auto-detect truncated at classifier budget; defaulting to summarise"
                    );
                } else {
                    tracing::warn!(error = %error, "auto-detect failed; defaulting to summarise");
                    eprintln!("  Phase 2: auto-detect failed; defaulting to summarise");
                }
                TaskType::Summarise
            }
        }
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
        joinset_pool: Arc::new(std::sync::Mutex::new(Vec::new())),
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
    match std::fs::symlink_metadata(&target) {
        Ok(meta) => {
            anyhow::ensure!(
                !meta.file_type().is_symlink(),
                "refusing codegen result target symlink: {}",
                target.display()
            );
            anyhow::ensure!(
                open_file_no_follow(&target).is_some(),
                "refusing unsafe codegen result target: {}",
                target.display()
            );
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(error)
                .with_context(|| format!("failed to inspect result target {}", target.display()));
        }
    }
    Ok(())
}

fn sanitize_error(error: &anyhow::Error) -> String {
    let mut detail = format!("{error:?}");
    for (name, value) in std::env::vars() {
        let looks_sensitive =
            name == "FIREWORKS_API" || name.starts_with("OPENCODE_") || name == "CLAUDE_API_KEY";
        if looks_sensitive && !value.is_empty() {
            detail = detail.replace(&value, "[REDACTED]");
        }
    }
    tracing::error!(error = %detail, "request failed");
    "Not Found".to_string()
}

fn is_auto_detect_truncation_error(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        cause
            .to_string()
            .contains("response truncated at max_tokens=")
    })
}

fn ensure_no_live_guards(oracle: &Arc<Oracle>) -> Result<()> {
    let metrics = oracle.metrics();
    let codegen_guards = codegen::codegen_guard_live_counts();
    debug_assert_eq!(metrics.budget_guards_live, 0, "BudgetGuard leak detected");
    debug_assert_eq!(
        metrics.inflight_guards_live, 0,
        "InflightGuard leak detected"
    );
    debug_assert_eq!(codegen_guards.budget, 0, "CodegenBudgetGuard leak detected");
    debug_assert_eq!(codegen_guards.flight, 0, "CodegenFlightGuard leak detected");
    debug_assert_eq!(
        codegen_guards.child_cleanup, 0,
        "ChildCleanup leak detected"
    );
    if metrics.budget_guards_live > 0
        || metrics.inflight_guards_live > 0
        || codegen_guards.budget > 0
        || codegen_guards.flight > 0
        || codegen_guards.child_cleanup > 0
    {
        anyhow::bail!(
            "resource leak detected at shutdown (budget_guards_live={}, inflight_guards_live={}, codegen_budget_guards_live={}, codegen_flight_guards_live={}, child_cleanup_guards_live={})",
            metrics.budget_guards_live,
            metrics.inflight_guards_live,
            codegen_guards.budget,
            codegen_guards.flight,
            codegen_guards.child_cleanup
        );
    }
    Ok(())
}

// ── Main — Algorithm 1: Complete λ-RLM System ───────────────────

#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("{}", sanitize_error(&error));
        std::process::exit(1);
    }
}

async fn run() -> Result<()> {
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

    let oracle = build_oracle(&cli)?;
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
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
                let mut sighup =
                    signal(SignalKind::hangup()).expect("failed to install SIGHUP handler");
                tokio::select! {
                    _ = ctrl_c => {},
                    _ = sigterm.recv() => {},
                    _ = sighup.recv() => {
                        eprintln!("\n>>> SIGHUP received — draining in-flight work before restart...");
                    },
                }
            }
            #[cfg(not(unix))]
            {
                let _ = ctrl_c.await;
            }
            eprintln!("\n>>> Signal received, initiating graceful shutdown...");
            oracle_shutdown.trigger_shutdown();
            let _ = tx.send(true);
        });
    }
    {
        let oracle_cache = Arc::clone(&oracle);
        let mut maintenance_shutdown_rx = shutdown_rx.clone();
        tokio::spawn(async move {
            let mut interval =
                tokio::time::interval(Duration::from_secs(CACHE_MAINTENANCE_INTERVAL_SECS));
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        oracle_cache.run_cache_maintenance();
                    }
                    changed = maintenance_shutdown_rx.changed() => {
                        if changed.is_err() || *maintenance_shutdown_rx.borrow() {
                            break;
                        }
                    }
                }
            }
        });
    }
    drop(shutdown_tx);

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
        let result = run_analysis(&cli, &oracle, shutdown_rx.clone()).await?;
        println!("{result}");
        ensure_no_live_guards(&oracle)?;
        return Ok(());
    }
    let generator = generator.unwrap();

    // ── Fix loop mode ──
    let work_dir = if cli.path.is_file() {
        std::fs::canonicalize(cli.path.parent().unwrap_or(&cli.path))?
    } else {
        std::fs::canonicalize(&cli.path)?
    };

    let unlimited_iterations = cli.max_iterations == 0;
    if unlimited_iterations {
        eprintln!(
            ">>> --max-iterations=0 requested; running until clean, shutdown, or budget exhaustion"
        );
    }

    let mut iteration: usize = 1;
    let mut reached_iteration_limit = false;
    loop {
        if !unlimited_iterations && iteration > cli.max_iterations {
            reached_iteration_limit = true;
            break;
        }

        if *shutdown_rx.borrow() {
            eprintln!("\n>>> Shutdown requested; stopping fix loop.");
            break;
        }

        eprintln!("\n================================================================");
        if unlimited_iterations {
            eprintln!("  ITERATION {iteration} (unbounded)");
        } else {
            eprintln!("  ITERATION {iteration}/{}", cli.max_iterations);
        }
        eprintln!("================================================================\n");

        // 1. Analyze
        let result = run_analysis(&cli, &oracle, shutdown_rx.clone()).await?;

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
            ensure_no_live_guards(&oracle)?;
            return Ok(());
        }

        // 3. Hand to code generator
        validate_codegen_result_target(&work_dir)?;
        let _interactive_permit = if cli.interactive {
            Some(
                INTERACTIVE_SESSION_SEMAPHORE
                    .get_or_init(|| tokio::sync::Semaphore::new(1))
                    .acquire()
                    .await
                    .context("interactive session already in progress")?,
            )
        } else {
            None
        };
        let summary = codegen::run_code_generator(
            oracle.as_ref(),
            &generator,
            &result,
            &work_dir,
            &cli.question,
            iteration,
            cli.interactive,
            Duration::from_secs(cli.interactive_timeout.saturating_mul(60)),
            shutdown_rx.clone(),
        )
        .await?;

        ensure_no_live_guards(&oracle)?;

        oracle.print_telemetry();

        let trimmed: String = summary.chars().take(200).collect();
        eprintln!(">>> Iteration {iteration} done: {trimmed}");
        eprintln!(
            "    (full log: {})",
            codegen::log_file_name(&generator, iteration)
        );

        if !oracle.budget_unlimited() && oracle.budget_remaining() == 0 {
            eprintln!(">>> Budget exhausted. Terminating fix loop.");
            break;
        }

        iteration = iteration
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("iteration counter overflow"))?;
    }

    if reached_iteration_limit {
        eprintln!(
            "\n>>> Reached max iterations ({}) with {}. Stopping loop.",
            cli.max_iterations, generator
        );
        eprintln!(">>> Run again to continue if needed.");
    }
    ensure_no_live_guards(&oracle)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    fn env_lock() -> &'static Mutex<()> {
        static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        ENV_LOCK.get_or_init(|| Mutex::new(()))
    }

    struct ScopedEnvVar {
        key: &'static str,
        prior: Option<std::ffi::OsString>,
    }

    impl ScopedEnvVar {
        fn set(key: &'static str, value: &str) -> Self {
            let prior = std::env::var_os(key);
            std::env::set_var(key, value);
            Self { key, prior }
        }
    }

    impl Drop for ScopedEnvVar {
        fn drop(&mut self) {
            if let Some(value) = self.prior.take() {
                std::env::set_var(self.key, value);
            } else {
                std::env::remove_var(self.key);
            }
        }
    }

    #[test]
    fn validate_generator_access_fails_fast_when_opencode_missing_from_path() {
        let _guard = env_lock().lock().expect("env lock");
        let temp = tempfile::tempdir().expect("tempdir");
        let empty_path = temp.path().to_string_lossy().to_string();
        let _path = ScopedEnvVar::set("PATH", &empty_path);

        let err = validate_generator_access(&types::CodeGenerator::Opencode)
            .expect_err("missing opencode should fail preflight");
        assert!(
            format!("{err:#}").contains("opencode binary not in PATH"),
            "unexpected preflight error: {err:#}"
        );
    }

    #[test]
    fn cli_accepts_prompt_alias_for_question() {
        let cli = Cli::parse_from(["lambda_rlm", "--path", ".", "--prompt", "smoke"]);
        assert_eq!(cli.question, "smoke");
    }

    #[test]
    fn truncation_error_detection_matches_oracle_message() {
        let err = anyhow::anyhow!(
            "response truncated at max_tokens=64 after 5 segment(s); increase token budget"
        );
        assert!(is_auto_detect_truncation_error(&err));
    }

    #[test]
    fn truncation_error_detection_ignores_other_errors() {
        let err = anyhow::anyhow!("network timeout");
        assert!(!is_auto_detect_truncation_error(&err));
    }

    #[test]
    fn sanitize_error_redacts_configured_secrets() {
        let _guard = env_lock().lock().expect("env lock");
        let _fireworks = ScopedEnvVar::set("FIREWORKS_API", "fw-test-secret");
        let _claude = ScopedEnvVar::set("CLAUDE_API_KEY", "claude-test-secret");
        let _opencode = ScopedEnvVar::set("OPENCODE_TOKEN", "opencode-test-secret");

        let err = anyhow::anyhow!("failure fw-test-secret claude-test-secret opencode-test-secret");
        let exposed = sanitize_error(&err);
        assert_eq!(exposed, "Not Found");

        let debug = format!("{err:?}");
        assert!(debug.contains("fw-test-secret"));
        assert!(debug.contains("claude-test-secret"));
        assert!(debug.contains("opencode-test-secret"));
    }

    #[test]
    fn interactive_timeout_parser_enforces_bounds() {
        assert!(parse_interactive_timeout_minutes("0").is_err());
        assert!(parse_interactive_timeout_minutes("121").is_err());
        assert_eq!(parse_interactive_timeout_minutes("30").expect("valid"), 30);
    }

    #[test]
    fn cli_rejects_interactive_without_generator_flag() {
        let parsed = Cli::try_parse_from([
            "lambda_rlm",
            "--path",
            ".",
            "--question",
            "smoke",
            "--interactive",
        ]);
        assert!(parsed.is_err());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn guarded_file_drop_releases_permit_during_unwind() {
        let temp = tempfile::tempdir().expect("tempdir");
        let file_path = temp.path().join("guarded.txt");
        std::fs::write(&file_path, "ok").expect("seed file");

        let sem = Arc::new(tokio::sync::Semaphore::new(1));
        let permit = Arc::clone(&sem)
            .acquire_owned()
            .await
            .expect("acquire permit");
        let file = std::fs::File::open(&file_path).expect("open file");

        let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guarded = GuardedFile::new(file, permit);
            panic!("simulate panic after GuardedFile construction");
        }));
        assert!(unwind.is_err());

        let recovered = Arc::clone(&sem)
            .try_acquire_owned()
            .expect("fd permit should be returned by Drop");
        drop(recovered);
    }
}
