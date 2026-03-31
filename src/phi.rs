//! Recursive Executor Φ — Algorithm 2 from the paper.
//!
//! v3: JoinSet structured concurrency, Verifier at leaves,
//! structural chunking, ordered results, graceful degradation,
//! cooperative shutdown via watch channel.
//!
//! v3.1: Bounded concurrency via semaphore, budget pre-flight checks.
//!
//! Core equation (Equation 4):
//!   fix(λf. λP.
//!     if |P| ≤ τ  then  Verify(M(P))
//!     else  Reduce(⊕, Map(λpi. f(pi), Chunk(P, k)))
//!   )

use crate::combinator::{
    comb_split, comb_split_overlap, filter_by_keyword_predicate, structural_chunk,
};
use crate::oracle::Oracle;
use crate::reduce::reduce_for_task;
use crate::types::TaskType;
use crate::verify::{Verifier, VerifyResult};
use anyhow::Result;
use futures::future::BoxFuture;
use std::sync::Mutex;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{watch, OwnedSemaphorePermit, Semaphore};
use tokio::task::JoinSet;

/// Configuration for the recursive executor. Shared via Arc across all
/// recursive calls — zero cloning cost per recursion level.
pub struct PhiConfig {
    pub question: String,
    pub task: TaskType,
    pub tau: usize,
    pub k: usize,
    pub max_depth: usize,
    pub overlap: usize,
    pub max_tokens: u32,
    pub keywords: Vec<String>,
    // Keep this a one-way reference (PhiConfig -> Oracle). Oracle must not
    // store Arc<PhiConfig> or callbacks that capture PhiConfig to avoid cycles.
    pub oracle: Arc<Oracle>,
    pub verifier: Verifier,
    /// Cooperative shutdown signal. When the sender drops or sends true,
    /// in-flight phi recursions abort gracefully, preserving budget.
    pub shutdown: watch::Receiver<bool>,
    /// Bounds total concurrent recursive tasks to prevent OOM on wide trees.
    /// Default: 100 permits. Each phi() call holds one permit for its duration.
    pub concurrency: Arc<Semaphore>,
    /// Reuses JoinSet allocations across recursion levels.
    pub joinset_pool: Arc<Mutex<Vec<JoinSet<(usize, Result<String>)>>>>,
    /// Trace ID for correlating logs across recursive phi calls.
    /// Survives the entire recursion tree — all children share the same trace_id.
    /// 16 bytes (blake3 truncated) — negligible allocation cost per request.
    pub trace_id: [u8; 16],
}

/// Maximum input bytes phi() will process. Inputs beyond this are truncated
/// with a warning. Prevents OOM when a single massive file (e.g., 1GB minified
/// JS) generates millions of chunks that exhaust memory before semaphore
/// backpressure kicks in. 256MB allows ~42k chunks at tau=6000.
const MAX_PHI_INPUT_BYTES: usize = 256 * 1024 * 1024;
const JOINSET_DRAIN_TIMEOUT_SECS: u64 = 5;
const MAX_JOINSET_POOL: usize = 32;

fn checkout_joinset(cfg: &PhiConfig) -> JoinSet<(usize, Result<String>)> {
    cfg.joinset_pool
        .lock()
        .ok()
        .and_then(|mut pool| pool.pop())
        .unwrap_or_else(JoinSet::new)
}

fn return_joinset(cfg: &PhiConfig, mut set: JoinSet<(usize, Result<String>)>) {
    set.detach_all();
    if let Ok(mut pool) = cfg.joinset_pool.lock() {
        if pool.len() < MAX_JOINSET_POOL {
            pool.push(set);
        }
    }
}

async fn abort_and_drain(set: &mut JoinSet<(usize, Result<String>)>, depth: usize) {
    set.abort_all();
    let drain = async {
        while let Some(res) = set.join_next().await {
            drop(res);
        }
    };
    if tokio::time::timeout(Duration::from_secs(JOINSET_DRAIN_TIMEOUT_SECS), drain)
        .await
        .is_err()
    {
        tracing::error!(
            depth,
            timeout_secs = JOINSET_DRAIN_TIMEOUT_SECS,
            "joinset drain timed out after abort_all"
        );
        set.detach_all();
    }
}

/// PRE: cfg.task != Auto (resolved in Phase 2)
/// PRE: text.len() > 0
/// POST: Ok(result) where result is verified at leaf level
///
/// GLOBAL RESOURCE ORDERING (must be maintained across all refactors):
///   Semaphore (concurrency permit) → Budget (CallBudget) → Cache (ReplayCache)
///
/// phi() acquires a Semaphore permit, then calls Oracle::call() which acquires
/// Budget then checks Cache. Parent phi() drops its permit (line ~203) before
/// spawning children, preventing hierarchical deadlock when k^depth exceeds
/// available permits. This ordering guarantees liveness: no cycle exists in the
/// resource acquisition graph. A future refactor that inverts this order (e.g.,
/// acquiring Budget before Semaphore) risks deadlock under deep recursion.
pub fn phi(
    cfg: Arc<PhiConfig>,
    text: String,
    depth: usize,
    permit: Option<OwnedSemaphorePermit>,
) -> BoxFuture<'static, Result<String>> {
    Box::pin(async move {
        // Check for shutdown before doing any work
        if *cfg.shutdown.borrow() {
            return Err(anyhow::anyhow!("Graceful shutdown requested"));
        }

        let indent = "| ".repeat(depth);

        // ── INPUT SIZE GUARD ──
        // Prevent OOM from pathologically large inputs generating unbounded chunks.
        let max_input = MAX_PHI_INPUT_BYTES;
        let text = if text.len() > max_input {
            tracing::warn!(
                depth,
                input_bytes = text.len(),
                limit = max_input,
                "input exceeds memory safety limit, truncating"
            );
            eprintln!(
                "{indent}|  WARNING: input {} bytes exceeds {} limit, truncating",
                text.len(),
                max_input
            );
            let mut byte_end = max_input;
            while byte_end > 0 && !text.is_char_boundary(byte_end) {
                byte_end -= 1;
            }
            text[..byte_end].to_string()
        } else {
            text
        };

        // ── BASE CASE: |P| ≤ τ → Verify(M(P)) ──
        if text.len() <= cfg.tau {
            eprintln!(
                "{indent}+- LEAF ({} chars) depth={depth} trace={}",
                text.len(),
                hex::encode(&cfg.trace_id[..4])
            );
            let (sys, prompt) = leaf_prompt(&text, &cfg.question, &cfg.task);

            match cfg.oracle.call(&sys, &prompt, 2048).await {
                Ok(raw) => {
                    // Verify the leaf output before returning
                    match cfg.verifier.check(&raw, text.len()) {
                        VerifyResult::Accept(verified) => return Ok(verified),
                        VerifyResult::Degraded(output, reason) => {
                            tracing::debug!(depth, reason, "leaf verified (degraded)");
                            eprintln!("{indent}|  VERIFIED (degraded): {reason}");
                            return Ok(output);
                        }
                        VerifyResult::Reject(reason) => {
                            tracing::debug!(depth, reason, "leaf verified (rejected)");
                            eprintln!("{indent}|  VERIFIED (rejected): {reason}");
                            if depth > 0 {
                                return Ok(String::new()); // filtered out by reduce
                            }
                            return Ok(format!("No useful output ({reason})"));
                        }
                    }
                }
                Err(e) if depth > 0 => {
                    tracing::warn!(depth, error = %e, "leaf call failed, degrading");
                    eprintln!("{indent}|  DEGRADED (leaf failed): {e}");
                    return Ok(format!("[degraded: leaf call failed at depth {depth}]"));
                }
                Err(e) => return Err(e),
            }
        }

        // ── DEPTH GUARD: prevent unbounded recursion on adversarial input ──
        if depth >= cfg.max_depth {
            tracing::warn!(
                depth,
                max_depth = cfg.max_depth,
                chars = text.len(),
                "max recursion depth reached, treating as leaf"
            );
            eprintln!(
                "{indent}+- DEPTH LIMIT ({} chars) depth={depth}/{}",
                text.len(),
                cfg.max_depth
            );
            // Truncate to tau to stay within model context window
            let truncated = if text.len() > cfg.tau {
                let mut end = cfg.tau;
                while end > 0 && !text.is_char_boundary(end) {
                    end -= 1;
                }
                &text[..end]
            } else {
                &text
            };
            let (sys, prompt) = leaf_prompt(truncated, &cfg.question, &cfg.task);
            match cfg.oracle.call(&sys, &prompt, 2048).await {
                Ok(raw) => match cfg.verifier.check(&raw, text.len()) {
                    VerifyResult::Accept(v) | VerifyResult::Degraded(v, _) => return Ok(v),
                    VerifyResult::Reject(reason) => {
                        return Ok(format!(
                            "[degraded: depth limit, verification rejected: {reason}]"
                        ));
                    }
                },
                Err(_) if depth > 0 => {
                    return Ok(format!(
                        "[degraded: depth limit leaf failed at depth {depth}]"
                    ));
                }
                Err(e) => return Err(e),
            }
        }

        // ── RECURSIVE CASE ──

        // Release parent's concurrency permit before spawning children.
        // Prevents hierarchical deadlock: parents holding permits while
        // awaiting children who need permits starves leaves when
        // k^depth exceeds available permits (e.g., k=2, depth=7, permits=100).
        drop(permit);

        // 1. CHUNK — structural chunking for code, overlap for multi-hop
        let chunks = match cfg.task {
            TaskType::MultiHop => comb_split_overlap(&text, cfg.k, cfg.overlap),
            TaskType::Search
            | TaskType::Classify
            | TaskType::Aggregate
            | TaskType::Pairwise
            | TaskType::Summarise => structural_chunk(&text, cfg.tau),
            TaskType::Auto => unreachable!("Auto resolved in Phase 2"),
        };

        // Fall back to comb_split if structural chunking produced too few chunks
        let chunks = if chunks.len() < 2 && text.len() > cfg.tau {
            comb_split(&text, cfg.k)
        } else {
            chunks
        };

        eprintln!(
            "{indent}+- SPLIT ({} chars) -> {} children, depth={depth} trace={}",
            text.len(),
            chunks.len(),
            hex::encode(&cfg.trace_id[..4]),
        );

        // 2. KEYWORD PRE-FILTER (Algorithm 2, line 9)
        let chunks = filter_by_keyword_predicate(chunks, &cfg.task, &cfg.keywords);
        if chunks.is_empty() {
            return Ok(match cfg.task {
                TaskType::Search => "No relevant information found.".into(),
                TaskType::MultiHop => "No relevant evidence in this section.".into(),
                TaskType::Classify
                | TaskType::Aggregate
                | TaskType::Pairwise
                | TaskType::Summarise
                | TaskType::Auto => "No data in this section.".into(),
            });
        }

        // 3. BUDGET PRE-FLIGHT CHECK (tree-aware, pessimistic reservation)
        // Uses atomic try_acquire_n + release_n to eliminate TOCTOU race.
        // Under high concurrency, a read-then-check allows multiple branches to
        // simultaneously see sufficient budget and proceed, causing partial
        // orphan exhaustion. The atomic reserve serializes concurrent checks:
        // only branches that atomically acquire the estimated budget proceed.
        let num_children = chunks.len();
        if !cfg.oracle.budget_unlimited() {
            let remaining_depth = cfg.max_depth.saturating_sub(depth);
            let estimated_subtree = if remaining_depth <= 1 {
                // Leaf level: each child is one call
                num_children
            } else {
                // Interior: estimate k^remaining_depth leaves, capped to avoid overflow
                let leaves = (cfg.k as u64)
                    .checked_pow(remaining_depth as u32)
                    .unwrap_or(u64::MAX)
                    .min(100_000) as usize;
                leaves
            };
            // Atomic reservation: if two branches race, only one succeeds.
            // Immediately release after the gate — children acquire individually
            // via oracle.call(). The atomic reserve-release prevents phantom
            // budget availability without changing per-call accounting.
            if !cfg.oracle.budget_try_reserve(estimated_subtree) {
                let remaining = cfg.oracle.budget_remaining();
                tracing::warn!(
                    depth,
                    num_children,
                    estimated_subtree,
                    remaining,
                    "budget insufficient for subtree, degrading"
                );
                eprintln!(
                    "{indent}|  DEGRADED (budget: subtree needs ~{estimated_subtree}, have {remaining})"
                );
                return Ok(format!(
                    "[degraded: budget_exhausted, subtree needs ~{} but {} remaining]",
                    estimated_subtree, remaining
                ));
            }
            cfg.oracle.budget_unreserve(estimated_subtree);
        }

        // 4. MAP — JoinSet structured concurrency, index-tagged for ordering
        // Each spawn acquires a concurrency permit to bound total recursive tasks.
        let mut set = checkout_joinset(&cfg);
        for (i, chunk) in chunks.into_iter().enumerate() {
            let cfg = Arc::clone(&cfg);
            let sem = Arc::clone(&cfg.concurrency);
            eprintln!("{indent}|  child {}/{num_children}", i + 1);
            set.spawn(async move {
                // Acquire concurrency permit — blocks if too many tasks active
                let permit = sem.acquire_owned().await.unwrap();
                let result = phi(cfg, chunk, depth + 1, Some(permit)).await;
                (i, result)
            });
        }

        // Collect results with cooperative shutdown check.
        // On depth-0 fatal errors, drain remaining tasks before returning
        // to ensure RAII guards (SemaphorePermit, BudgetGuard) fire cleanly.
        let mut indexed_results: Vec<(usize, String)> = Vec::with_capacity(num_children);
        let mut shutdown_rx = cfg.shutdown.clone();
        let mut fatal_error: Option<anyhow::Error> = None;

        loop {
            tokio::select! {
                biased;
                // Shutdown takes priority — drain with timeout instead of
                // hard abort to ensure RAII guards (BudgetGuard, SemaphorePermit)
                // complete their Drop before we return.
                _ = shutdown_rx.changed() => {
                    tracing::info!(depth, collected = indexed_results.len(), total = num_children, "shutdown: draining children");
                    abort_and_drain(&mut set, depth).await;
                    return_joinset(&cfg, set);
                    return Err(anyhow::anyhow!("Graceful shutdown requested at depth {depth}"));
                }
                join_result = set.join_next() => {
                    match join_result {
                        None => break, // All tasks completed
                        Some(Ok((idx, Ok(text)))) => indexed_results.push((idx, text)),
                        Some(Ok((idx, Err(e)))) if depth > 0 => {
                            tracing::warn!(depth, child = idx, error = %e, "child failed, degrading");
                            eprintln!("{indent}|  child {idx} failed, degrading: {e}");
                        }
                        Some(Ok((_idx, Err(e)))) => {
                            // Depth 0: record error but drain remaining tasks
                            // so their RAII guards drop cleanly.
                            fatal_error.get_or_insert(e);
                            abort_and_drain(&mut set, depth).await;
                            break;
                        }
                        Some(Err(join_err)) if join_err.is_cancelled() => {
                            tracing::debug!(depth, "child task cancelled");
                        }
                        Some(Err(join_err)) if depth > 0 => {
                            tracing::error!(depth, error = %join_err, "child task panicked");
                            eprintln!("{indent}|  child task panicked, degrading: {join_err}");
                        }
                        Some(Err(join_err)) => {
                            fatal_error.get_or_insert_with(|| anyhow::anyhow!("Child task failed: {join_err}"));
                            abort_and_drain(&mut set, depth).await;
                            break;
                        }
                    }
                }
            }
        }

        if let Some(e) = fatal_error {
            return_joinset(&cfg, set);
            return Err(e);
        }

        indexed_results.sort_by_key(|(idx, _)| *idx);
        let child_results: Vec<String> =
            indexed_results.into_iter().map(|(_, text)| text).collect();

        if child_results.is_empty() {
            return_joinset(&cfg, set);
            return Ok(format!(
                "[degraded: all {} children failed at depth {depth}]",
                num_children
            ));
        }

        // 5. REDUCE ⊕ — task-specific composition
        eprintln!(
            "{indent}+- REDUCE depth={depth} ({} children -> 1 result)",
            child_results.len(),
        );

        if matches!(cfg.task, TaskType::Summarise | TaskType::MultiHop) {
            let chars: usize = child_results.iter().map(|r| r.len()).sum();
            eprintln!("{indent}|  (synthesis: {chars} chars)");
        }

        let reduce_result = reduce_for_task(
            &cfg.task,
            child_results,
            &cfg.question,
            depth,
            cfg.max_depth,
            Arc::clone(&cfg.oracle),
            cfg.max_tokens,
        )
        .await;

        return_joinset(&cfg, set);

        // Graceful degradation for neural reduce failures
        match reduce_result {
            Ok(text) => Ok(text),
            Err(e) if depth > 0 => {
                tracing::warn!(depth, error = %e, "reduce failed, degrading");
                eprintln!("{indent}|  DEGRADED (reduce failed): {e}");
                Ok(format!("[degraded: reduce failed at depth {depth}]"))
            }
            Err(e) => Err(e),
        }
    })
}

// ── Task Auto-Detection — Phase 2 ───────────────────────────────

/// Sanitize document preview for LLM classification to prevent prompt injection.
/// Strips control characters (except \n, \t) and truncates to a fixed byte length
/// to prevent suffix attacks and injection via delimiter keywords.
fn sanitize_for_classification(preview: &str) -> String {
    const MAX_PREVIEW_BYTES: usize = 2000;
    let truncated = if preview.len() > MAX_PREVIEW_BYTES {
        let mut end = MAX_PREVIEW_BYTES;
        while end > 0 && !preview.is_char_boundary(end) {
            end -= 1;
        }
        &preview[..end]
    } else {
        preview
    };
    truncated
        .chars()
        .filter(|c| *c == '\n' || *c == '\t' || !c.is_control())
        .collect()
}

pub async fn auto_detect_task(
    preview: &str,
    total_len: usize,
    question: &str,
    oracle: &Oracle,
) -> Result<TaskType> {
    // Sanitize preview to prevent prompt injection steering classification
    // toward expensive task types (e.g., MultiHop, Pairwise).
    let preview = sanitize_for_classification(preview);

    let system = "You are a task classifier. Given a document preview and a question, \
                  classify the required task type as EXACTLY one of these:\n\
                  - search: find specific information or code\n\
                  - classify: label, categorize, or identify types of each section\n\
                  - aggregate: extract, count, collect, or list items across the document\n\
                  - pairwise: compare, find duplicates, or cross-reference items\n\
                  - summarise: summarize, explain, or understand the document\n\
                  - multi_hop: trace a flow, chain reasoning across sections, follow dependencies\n\n\
                  Respond with ONLY the task type name, nothing else.";

    let user = format!(
        "Document preview (first {} chars of {} total):\n{}\n\nQuestion: {}",
        preview.len(),
        total_len,
        preview,
        question
    );

    let response = oracle.call(system, &user, 64).await?;
    let normalized = response.trim().to_lowercase().replace(['-', '_', ' '], "");

    let task = match normalized.as_str() {
        "search" => TaskType::Search,
        "classify" => TaskType::Classify,
        "aggregate" => TaskType::Aggregate,
        "pairwise" => TaskType::Pairwise,
        "summarise" | "summarize" | "summary" => TaskType::Summarise,
        "multihop" => TaskType::MultiHop,
        other => {
            tracing::warn!(
                raw = other,
                "auto-detect returned unknown task, defaulting to summarise"
            );
            eprintln!(
                "    Auto-detect returned '{}', defaulting to summarise",
                other
            );
            TaskType::Summarise
        }
    };

    eprintln!("  Phase 2: auto-detected task = {task}");
    Ok(task)
}

// ── Leaf Prompts ─────────────────────────────────────────────────
// Grounding anchors: every prompt instructs M to cite exact file
// paths and line numbers from the source input.

/// Public wrapper for direct dispatch in main (Phase 3, |P| ≤ K).
pub fn leaf_prompt_public(chunk: &str, question: &str, task: &TaskType) -> (String, String) {
    leaf_prompt(chunk, question, task)
}

fn leaf_prompt(chunk: &str, question: &str, task: &TaskType) -> (String, String) {
    match task {
        TaskType::Search => (
            "You are a code search engine. Find the specific code relevant to the question. \
             Quote the relevant lines with exact file paths and line numbers. \
             Cite as `path/file.rs:42`. If nothing relevant exists, respond: NO_MATCH"
                .into(),
            format!("Question: {question}\n\nCode:\n```\n{chunk}\n```"),
        ),

        TaskType::Classify => (
            "Analyze this code section. Classify its purpose, module type, and key patterns. \
             Return a concise bullet list with file path references: what it is, what it does, \
             key functions/types. Cite as `path/file.rs:FunctionName`."
                .into(),
            format!("Task: {question}\n\nCode:\n```\n{chunk}\n```"),
        ),

        TaskType::Aggregate => (
            "Extract all items matching the task description from this code section. \
             Return each item as a bullet point with file path and function context. \
             Cite as `path/file.rs:42`. If nothing matches, respond: NO_DATA"
                .into(),
            format!("Task: {question}\n\nCode:\n```\n{chunk}\n```"),
        ),

        TaskType::Pairwise => (
            "Extract distinct entities, patterns, or code structures from this section. \
             Return each as a bullet point with enough context to compare later. \
             Include source location as `path/file.rs:FunctionName`."
                .into(),
            format!("Task: {question}\n\nCode:\n```\n{chunk}\n```"),
        ),

        TaskType::Summarise => (
            "Summarize this code section in 3-5 concise bullets: \
             functions/structs defined, what they do, dependencies, patterns used. \
             Reference file paths where relevant."
                .into(),
            format!("```\n{chunk}\n```"),
        ),

        TaskType::MultiHop => (
            "Extract any facts, relationships, or evidence chains from this section \
             relevant to the question. Be specific about entity names, function calls, \
             data flows. Cite exact file paths and line numbers as `path/file.rs:42`. \
             If nothing relevant, respond: NO_DATA"
                .into(),
            format!("Question: {question}\n\nCode:\n```\n{chunk}\n```"),
        ),

        TaskType::Auto => unreachable!("Auto should be resolved before leaf calls"),
    }
}
