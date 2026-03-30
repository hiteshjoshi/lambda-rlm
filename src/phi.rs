//! Recursive Executor Φ — Algorithm 2 from the paper.
//!
//! v3: JoinSet structured concurrency, Verifier at leaves,
//! structural chunking, ordered results, graceful degradation,
//! cooperative shutdown via watch channel.
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
use std::sync::Arc;
use tokio::sync::watch;
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
    pub oracle: Arc<Oracle>,
    pub use_quorum: bool,
    pub verifier: Verifier,
    /// Cooperative shutdown signal. When the sender drops or sends true,
    /// in-flight phi recursions abort gracefully, preserving budget.
    pub shutdown: watch::Receiver<bool>,
}

/// PRE: cfg.task != Auto (resolved in Phase 2)
/// PRE: text.len() > 0
/// POST: Ok(result) where result is verified at leaf level
pub fn phi(cfg: Arc<PhiConfig>, text: String, depth: usize) -> BoxFuture<'static, Result<String>> {
    Box::pin(async move {
        // Check for shutdown before doing any work
        if *cfg.shutdown.borrow() {
            return Err(anyhow::anyhow!("Graceful shutdown requested"));
        }

        let indent = "| ".repeat(depth);

        // ── BASE CASE: |P| ≤ τ → Verify(M(P)) ──
        if text.len() <= cfg.tau {
            eprintln!("{indent}+- LEAF ({} chars) depth={depth}", text.len());
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

        // ── RECURSIVE CASE ──

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
            "{indent}+- SPLIT ({} chars) -> {} children, depth={depth}",
            text.len(),
            chunks.len(),
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

        // 3. MAP — JoinSet structured concurrency, index-tagged for ordering
        let num_children = chunks.len();
        let mut set = JoinSet::new();
        for (i, chunk) in chunks.into_iter().enumerate() {
            let cfg = Arc::clone(&cfg);
            eprintln!("{indent}|  child {}/{num_children}", i + 1);
            set.spawn(async move {
                let result = phi(cfg, chunk, depth + 1).await;
                (i, result)
            });
        }

        // Collect results with cooperative shutdown check
        let mut indexed_results: Vec<(usize, String)> = Vec::with_capacity(num_children);
        let mut shutdown_rx = cfg.shutdown.clone();

        loop {
            tokio::select! {
                biased;
                // Shutdown takes priority
                _ = shutdown_rx.changed() => {
                    tracing::info!(depth, collected = indexed_results.len(), total = num_children, "shutdown: aborting children");
                    set.abort_all();
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
                        Some(Ok((_idx, Err(e)))) => return Err(e),
                        Some(Err(join_err)) if join_err.is_cancelled() => {
                            // Task was cancelled by shutdown — not an error
                            tracing::debug!(depth, "child task cancelled");
                        }
                        Some(Err(join_err)) if depth > 0 => {
                            tracing::error!(depth, error = %join_err, "child task panicked");
                            eprintln!("{indent}|  child task panicked, degrading: {join_err}");
                        }
                        Some(Err(join_err)) => anyhow::bail!("Child task failed: {join_err}"),
                    }
                }
            }
        }

        indexed_results.sort_by_key(|(idx, _)| *idx);
        let child_results: Vec<String> = indexed_results
            .into_iter()
            .map(|(_, text)| text)
            .collect();

        if child_results.is_empty() {
            return Ok(format!(
                "[degraded: all {} children failed at depth {depth}]",
                num_children
            ));
        }

        // 4. REDUCE ⊕ — task-specific composition
        eprintln!(
            "{indent}+- REDUCE depth={depth} ({} children -> 1 result)",
            child_results.len(),
        );

        if matches!(
            cfg.task,
            TaskType::Summarise | TaskType::MultiHop
        ) {
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
            cfg.use_quorum,
        )
        .await;

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

pub async fn auto_detect_task(
    preview: &str,
    total_len: usize,
    question: &str,
    oracle: &Oracle,
) -> Result<TaskType> {
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
    let normalized = response
        .trim()
        .to_lowercase()
        .replace(['-', '_', ' '], "");

    let task = match normalized.as_str() {
        "search" => TaskType::Search,
        "classify" => TaskType::Classify,
        "aggregate" => TaskType::Aggregate,
        "pairwise" => TaskType::Pairwise,
        "summarise" | "summarize" | "summary" => TaskType::Summarise,
        "multihop" => TaskType::MultiHop,
        other => {
            tracing::warn!(raw = other, "auto-detect returned unknown task, defaulting to summarise");
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
