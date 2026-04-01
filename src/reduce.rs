//! Reduce operators ⊕ — Table 1, Panel B.
//!
//! Symbolic ⊕: deterministic, no M calls (search, classify, aggregate)
//! Neural ⊕: calls M to synthesize (summarise, multi_hop)
//! Hybrid ⊕: symbolic + final M call (pairwise)
//!
//! All reduce functions take verified inputs — refusal detection is
//! handled upstream by the Verifier at leaf level.

use crate::combinator::{comb_cross, merge_dedup, merge_dedup_arc, parse_items};
use crate::oracle::Oracle;
use crate::types::TaskType;
use anyhow::Result;
use std::sync::Arc;

/// Maximum cross-product pairs sent to the neural comparison at depth 0.
/// Limits prompt size to prevent context-window overflow on large aggregate results.
/// The paper does not specify a limit; this is a practical guard.
const MAX_PAIRWISE_COMPARISONS: usize = 50;

/// Filter empty and degraded-marker results before reduction.
/// Light-weight: refusal detection is done upstream by Verifier.
fn filter_nonempty(results: Vec<String>) -> Vec<String> {
    results
        .into_iter()
        .filter(|r| {
            let t = r.trim();
            !t.is_empty() && !t.starts_with("[degraded:")
        })
        .collect()
}

/// Dispatch to the correct reducer for the task type.
/// Exhaustive match — every task type has a reduce path, no wildcards.
///
/// PRE: child_results are verified leaf/recursive outputs
/// POST: single combined result string
pub async fn reduce_for_task(
    task: &TaskType,
    child_results: Vec<String>,
    question: &str,
    depth: usize,
    max_depth: usize,
    oracle: Arc<Oracle>,
    max_tokens: u32,
) -> Result<String> {
    debug_assert!(
        !child_results.is_empty(),
        "PRE: child_results must be non-empty"
    );
    debug_assert!(depth <= max_depth, "PRE: depth must not exceed max_depth");

    let result = match task {
        TaskType::Search => {
            if depth == 0 && child_results.len() > 1 {
                reduce_search_top(child_results, question, oracle, max_tokens).await
            } else {
                Ok(reduce_search(child_results))
            }
        }
        TaskType::Classify => Ok(reduce_classify(child_results)),
        TaskType::Aggregate => Ok(reduce_aggregate(child_results)),
        TaskType::Pairwise => {
            if depth == 0 {
                reduce_pairwise_top(child_results, question, oracle, max_tokens).await
            } else {
                Ok(reduce_pairwise_intermediate(child_results))
            }
        }
        TaskType::Summarise => {
            reduce_summarise(
                child_results,
                question,
                depth,
                max_depth,
                &oracle,
                max_tokens,
            )
            .await
        }
        TaskType::MultiHop => {
            reduce_multi_hop(
                child_results,
                question,
                depth,
                max_depth,
                oracle,
                max_tokens,
            )
            .await
        }
        TaskType::Auto => unreachable!("Auto resolved in Phase 2"),
    };

    // POST: successful reduction must produce non-empty output
    if let Ok(ref s) = result {
        debug_assert!(!s.is_empty(), "POST: reduce output must be non-empty");
    }

    result
}

// ── Symbolic reducers (zero neural cost) ─────────────────────────

fn reduce_search(child_results: Vec<String>) -> String {
    let useful = filter_nonempty(child_results);
    if useful.is_empty() {
        "No relevant information found.".into()
    } else {
        useful.join("\n\n---\n\n")
    }
}

/// FilterBest for search at depth 0: LLM ranks and filters results by relevance.
async fn reduce_search_top(
    child_results: Vec<String>,
    question: &str,
    oracle: Arc<Oracle>,
    max_tokens: u32,
) -> Result<String> {
    let useful = filter_nonempty(child_results);
    if useful.is_empty() {
        return Ok("No relevant information found.".into());
    }
    if useful.len() == 1 {
        return Ok(useful.into_iter().next().unwrap());
    }
    let combined = useful
        .iter()
        .enumerate()
        .map(|(i, r)| format!("--- Result {} ---\n{}", i + 1, r))
        .collect::<Vec<_>>()
        .join("\n\n");
    let system = "You are ranking search results for relevance. Given search results \
                  from different code sections, select and present ONLY the results that \
                  directly answer the question. Discard irrelevant results. Preserve \
                  exact file paths and line citations from the original results.";
    let user = format!("Question: {question}\n\n{combined}");
    oracle.call(system, &user, max_tokens).await
}

fn reduce_classify(child_results: Vec<String>) -> String {
    let useful = filter_nonempty(child_results);
    if useful.is_empty() {
        "No classifications produced.".into()
    } else {
        useful.join("\n\n")
    }
}

fn reduce_aggregate(child_results: Vec<String>) -> String {
    let useful = filter_nonempty(child_results);
    if useful.is_empty() {
        "No items found.".into()
    } else {
        let all_items: Vec<Arc<str>> = useful
            .iter()
            .flat_map(|r| parse_items(r).into_iter().map(Arc::<str>::from))
            .collect();
        let deduped = merge_dedup_arc(all_items);
        if deduped.is_empty() {
            "No items found.".into()
        } else {
            // Pre-allocate: "- " (2) + item + "\n" (1) per item
            let mut out = String::with_capacity(deduped.iter().map(|s| s.len() + 3).sum());
            for (i, item) in deduped.iter().enumerate() {
                if i > 0 {
                    out.push('\n');
                }
                out.push_str("- ");
                out.push_str(item.as_ref());
            }
            out
        }
    }
}

fn reduce_pairwise_intermediate(child_results: Vec<String>) -> String {
    let useful = filter_nonempty(child_results);
    let all_items: Vec<Arc<str>> = useful
        .iter()
        .flat_map(|r| parse_items(r).into_iter().map(Arc::<str>::from))
        .collect();
    let deduped = merge_dedup_arc(all_items);
    // Pre-allocate: "- " (2) + item + "\n" (1) per item
    let mut out = String::with_capacity(deduped.iter().map(|s| s.len() + 3).sum());
    for (i, item) in deduped.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        out.push_str("- ");
        out.push_str(item.as_ref());
    }
    out
}

// ── Neural/hybrid reducers (call M) ─────────────────────────────

async fn reduce_pairwise_top(
    child_results: Vec<String>,
    question: &str,
    oracle: Arc<Oracle>,
    max_tokens: u32,
) -> Result<String> {
    let useful = filter_nonempty(child_results);
    if useful.is_empty() {
        return Ok("No entities found for comparison.".into());
    }

    let groups: Vec<Vec<String>> = useful.iter().map(|r| merge_dedup(parse_items(r))).collect();
    let pairs = comb_cross(&groups);
    verbose!(
        "    CROSS: {} groups -> {} pairs (symbolic)",
        groups.len(),
        pairs.len()
    );

    if pairs.is_empty() {
        return Ok("No cross-group pairs to compare.".into());
    }

    let pair_text: String = pairs
        .iter()
        .take(MAX_PAIRWISE_COMPARISONS)
        .enumerate()
        .map(|(i, (a, b))| format!("Pair {}:\n  A: {}\n  B: {}", i + 1, a, b))
        .collect::<Vec<_>>()
        .join("\n\n");

    let system = "You are analyzing pairs of code entities for the given task. \
                  For each pair, identify if they are related (duplicates, overlapping logic, \
                  similar patterns, inconsistencies). Only report meaningful findings.";
    let user = format!(
        "Task: {question}\n\n{pair_text}\n\n\
         Total pairs: {} (showing first {}). \
         Report findings as bullet points.",
        pairs.len(),
        pairs.len().min(MAX_PAIRWISE_COMPARISONS)
    );

    oracle.call(system, &user, max_tokens).await
}

async fn reduce_summarise(
    child_results: Vec<String>,
    question: &str,
    depth: usize,
    max_depth: usize,
    oracle: &Oracle,
    max_tokens: u32,
) -> Result<String> {
    let useful = filter_nonempty(child_results);
    if useful.is_empty() {
        return Ok("No summaries produced.".into());
    }

    let combined = useful
        .iter()
        .enumerate()
        .map(|(i, s)| format!("--- Section {} ---\n{}", i + 1, s))
        .collect::<Vec<_>>()
        .join("\n\n");

    if depth == 0 {
        let system = "You are a senior engineer. You have summaries of the entire codebase below. \
                      Answer the question with specific references to functions, files, patterns. \
                      Be thorough and actionable.";
        let user =
            format!("=== Codebase summaries ===\n\n{combined}\n\n=== Question ===\n{question}");
        oracle.call(system, &user, max_tokens).await
    } else {
        let system = format!(
            "You have summaries of {} adjacent code sections (depth {}/{} in recursion). \
             Combine them into ONE concise summary focused on the question below. \
             Preserve details relevant to the question: modules, purpose, \
             important functions, dependencies.",
            useful.len(),
            max_depth - depth,
            max_depth
        );
        let user = format!("Question: {question}\n\n{combined}");
        oracle.call(&system, &user, max_tokens).await
    }
}

async fn reduce_multi_hop(
    child_results: Vec<String>,
    question: &str,
    depth: usize,
    max_depth: usize,
    oracle: Arc<Oracle>,
    max_tokens: u32,
) -> Result<String> {
    let useful = filter_nonempty(child_results);
    if useful.is_empty() {
        return Ok("No relevant evidence found in this section.".into());
    }

    let combined = useful
        .iter()
        .enumerate()
        .map(|(i, s)| format!("--- Evidence {} ---\n{}", i + 1, s))
        .collect::<Vec<_>>()
        .join("\n\n");

    if depth == 0 {
        let system = "You are tracing a multi-hop reasoning chain across a codebase. \
                      Below are evidence fragments gathered from different sections. \
                      Chain them together to answer the question. Show the full path: \
                      which function calls which, how data flows, what depends on what.";
        let user =
            format!("=== Gathered evidence ===\n\n{combined}\n\n=== Question ===\n{question}");
        oracle.call(system, &user, max_tokens).await
    } else {
        let system = format!(
            "Synthesize these {} evidence fragments (depth {}/{}) relevant to the question. \
             Preserve specifics: function names, data flows, call chains.",
            useful.len(),
            max_depth - depth,
            max_depth
        );
        let user = format!("Question: {question}\n\n{combined}");
        oracle.call(&system, &user, max_tokens).await
    }
}
