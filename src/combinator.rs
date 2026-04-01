//! Combinator Library L — Table 1 from the paper.
//!
//! Every combinator except M is total and deterministic (Assumption A2).
//! Plus: structural chunker, keyword utilities, text processing.

use crate::types::TaskType;
use std::sync::Arc;

/// Split: Σ* × N → [Σ*]
/// PRE: k > 0
/// POST: each chunk is non-empty, chunks cover all input lines
pub fn comb_split(text: &str, k: usize) -> Vec<String> {
    if text.is_empty() || k == 0 {
        return vec![];
    }
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return vec![text.to_string()];
    }
    let per_chunk = (lines.len() + k - 1) / k;
    lines
        .chunks(per_chunk)
        .map(|ls| ls.join("\n"))
        .filter(|s| !s.trim().is_empty())
        .collect()
}

/// Split_δ: Σ* × N × N → [Σ*]
/// Overlap δ lines at boundaries for multi-hop continuity.
/// PRE: k > 0, delta_chars >= 0
/// POST: adjacent chunks share ~delta_chars of overlap
pub fn comb_split_overlap(text: &str, k: usize, delta_chars: usize) -> Vec<String> {
    if text.is_empty() || k == 0 {
        return vec![];
    }
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return vec![text.to_string()];
    }
    let total = lines.len();
    let per_chunk = (total + k - 1) / k;
    let avg_line_len = text.len() / total.max(1);
    let delta_lines = delta_chars / avg_line_len.max(1);

    let mut chunks = Vec::with_capacity(k);
    for i in 0..k {
        let raw_start = i * per_chunk;
        let raw_end = ((i + 1) * per_chunk).min(total);
        let start = raw_start.saturating_sub(delta_lines);
        let end = (raw_end + delta_lines).min(total);
        let slice = &lines[start..end];
        let estimated_len: usize = slice
            .iter()
            .map(|line| line.len())
            .sum::<usize>()
            .saturating_add(slice.len().saturating_sub(1));
        let mut chunk = String::with_capacity(estimated_len);
        for (idx, line) in slice.iter().enumerate() {
            if idx > 0 {
                chunk.push('\n');
            }
            chunk.push_str(line);
        }
        if !chunk.trim().is_empty() {
            chunks.push(chunk);
        }
    }
    chunks
}

/// Peek: Σ* × N² → Σ*
/// Safe substring on char boundaries.
pub fn comb_peek(text: &str, start: usize, end: usize) -> &str {
    let mut s = start.min(text.len());
    while s < text.len() && !text.is_char_boundary(s) {
        s += 1;
    }
    let mut e = end.min(text.len());
    while e > 0 && !text.is_char_boundary(e) {
        e -= 1;
    }
    if s > e {
        s = e;
    }
    debug_assert!(text.is_char_boundary(s));
    debug_assert!(text.is_char_boundary(e));
    &text[s..e]
}

/// Max combinatorial product to prevent OOM on large aggregate results.
/// Pre-calculated before allocation to fail fast on pathological input.
const MAX_COMBINATORIAL_PRODUCT: usize = 1_000_000;
const MAX_CROSS_PRODUCT_BYTES: usize = 64 * 1024 * 1024;

// Compile-time validation: zero causes logic errors in comb_cross,
// overflow causes silent truncation in pre-calculation.
const _: () = assert!(MAX_COMBINATORIAL_PRODUCT > 0);
const _: () = assert!(MAX_COMBINATORIAL_PRODUCT < usize::MAX / 2);

/// Cross: [α] × [α] → [(α, α)]
/// Pairs from DIFFERENT groups only. Purely symbolic, zero neural cost.
/// Returns empty Vec if the product would exceed MAX_COMBINATORIAL_PRODUCT.
pub fn comb_cross(groups: &[Vec<String>]) -> Vec<(String, String)> {
    // Pre-calculate total pairs to fail fast on combinatorial explosion
    let totals: Option<(usize, usize)> = {
        let mut count: usize = 0;
        let mut bytes: usize = 0;
        let group_lens: Vec<(usize, usize)> = groups
            .iter()
            .map(|g| (g.len(), g.iter().map(|s| s.len()).sum::<usize>()))
            .collect();
        for i in 0..groups.len() {
            for j in (i + 1)..groups.len() {
                let product = group_lens[i].0.checked_mul(group_lens[j].0);
                match product {
                    Some(p) => {
                        count = match count.checked_add(p) {
                            Some(c) => c,
                            None => return Vec::new(),
                        };
                        let ij_bytes = group_lens[i].1.checked_mul(group_lens[j].0).and_then(|a| {
                            group_lens[j]
                                .1
                                .checked_mul(group_lens[i].0)
                                .and_then(|b| a.checked_add(b))
                        });
                        bytes = match ij_bytes.and_then(|pair_bytes| bytes.checked_add(pair_bytes))
                        {
                            Some(v) => v,
                            None => return Vec::new(),
                        };
                    }
                    None => return Vec::new(),
                }
            }
        }
        Some((count, bytes))
    };
    let total = match totals {
        Some((t, b)) if t <= MAX_COMBINATORIAL_PRODUCT && b <= MAX_CROSS_PRODUCT_BYTES => t,
        _ => {
            tracing::warn!(
                limit = MAX_COMBINATORIAL_PRODUCT,
                byte_limit = MAX_CROSS_PRODUCT_BYTES,
                "combinatorial product too large, returning empty"
            );
            return Vec::new();
        }
    };

    let mut pairs = Vec::new();
    if pairs.try_reserve(total).is_err() {
        tracing::warn!(total, "failed to reserve memory for combinatorial product");
        return Vec::new();
    }
    for i in 0..groups.len() {
        for j in (i + 1)..groups.len() {
            for a in &groups[i] {
                for b in &groups[j] {
                    pairs.push((a.clone(), b.clone()));
                }
            }
        }
    }
    pairs
}

#[cfg(test)]
mod tests {
    use super::{comb_cross, comb_peek};

    #[test]
    fn comb_peek_never_panics_on_mid_codepoint_bounds() {
        let text = "a😀z";
        let got = comb_peek(text, 2, 2);
        assert_eq!(got, "");
    }

    #[test]
    fn comb_cross_applies_byte_guard() {
        let huge = "x".repeat(70 * 1024 * 1024);
        let groups = vec![vec![huge], vec!["b".to_string()]];
        let got = comb_cross(&groups);
        assert!(got.is_empty());
    }
}

pub fn extract_keywords(question: &str) -> Vec<String> {
    let stop = [
        "where", "is", "how", "does", "the", "a", "an", "in", "on", "at", "to", "for", "of",
        "with", "and", "or", "this", "that", "what", "are", "can", "do", "find", "show", "me",
        "explain", "which", "code", "function", "all", "each", "every", "any", "some", "between",
        "from", "into", "about",
    ];
    question
        .to_lowercase()
        .split_whitespace()
        .filter(|w| w.len() > 2 && !stop.contains(&w.to_lowercase().as_str()))
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
        .filter(|w| !w.is_empty())
        .collect()
}

pub fn keyword_matches(text: &str, keywords: &[String]) -> bool {
    let lower = text.to_lowercase();
    keywords.iter().any(|kw| lower.contains(kw.as_str()))
}

/// Max input size for parse_items to prevent pathological processing time.
const MAX_PARSE_INPUT_BYTES: usize = 100 * 1024 * 1024; // 100MB

/// Max items returned from parse_items to prevent unbounded Vec growth.
const MAX_ITEMS_COUNT: usize = 100_000;

/// Parse markdown-style list items from LLM output.
/// Handles: `- item`, `* item`, `• item`, `1. item`, `  - nested`, and plain lines.
/// Strips leading list markers and numbering, then filters short/empty lines.
///
/// PRE: text is UTF-8 LLM output (may contain any Unicode), len <= MAX_PARSE_INPUT_BYTES
/// POST: each item is trimmed, non-empty, len > 3, count <= MAX_ITEMS_COUNT
pub fn parse_items(text: &str) -> Vec<String> {
    if text.len() > MAX_PARSE_INPUT_BYTES {
        tracing::warn!(
            len = text.len(),
            limit = MAX_PARSE_INPUT_BYTES,
            "parse_items input exceeds size limit, returning empty"
        );
        return vec![];
    }
    let estimated_items = (text.len() / 48).max(1).min(MAX_ITEMS_COUNT);
    let mut items = Vec::with_capacity(estimated_items);

    for raw_line in text.lines() {
        if items.len() >= MAX_ITEMS_COUNT {
            break;
        }

        let trimmed = raw_line.trim();
        let trimmed = trimmed
            .strip_prefix("- ")
            .or_else(|| trimmed.strip_prefix("* "))
            .or_else(|| trimmed.strip_prefix("• "))
            .or_else(|| trimmed.strip_prefix("▪ "))
            .or_else(|| trimmed.strip_prefix("▸ "))
            .or_else(|| trimmed.strip_prefix("► "))
            .unwrap_or(trimmed);

        let trimmed = if trimmed.len() > 2 {
            let bytes = trimmed.as_bytes();
            if bytes[0].is_ascii_digit() {
                let digit_end = trimmed.bytes().take_while(|b| b.is_ascii_digit()).count();
                if digit_end < trimmed.len() {
                    let after_digits = &trimmed[digit_end..];
                    if let Some(rest) = after_digits.strip_prefix(". ") {
                        rest
                    } else if let Some(rest) = after_digits.strip_prefix(") ") {
                        rest
                    } else {
                        trimmed
                    }
                } else {
                    trimmed
                }
            } else {
                trimmed
            }
        } else {
            trimmed
        };

        let item = trimmed.trim();
        if !item.is_empty() && item.len() > 3 {
            items.push(item.to_string());
        }
    }

    items
}

pub fn merge_dedup(items: Vec<String>) -> Vec<String> {
    let mut seen = std::collections::BTreeSet::new();
    items
        .into_iter()
        .filter(|item| {
            let key: String = item
                .chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>()
                .to_lowercase();
            seen.insert(key)
        })
        .collect()
}

#[must_use]
pub fn merge_dedup_arc(items: Vec<Arc<str>>) -> Vec<Arc<str>> {
    let mut seen = std::collections::BTreeSet::new();
    items
        .into_iter()
        .filter(|item| {
            let key: String = item
                .chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>()
                .to_lowercase();
            seen.insert(key)
        })
        .collect()
}

// ── Keyword Pre-filter — Algorithm 2, line 9 ────────────────────
// Between Split and Map. Purely symbolic (zero neural cost).

/// PRE: chunks from comb_split or structural_chunk
/// POST: for Search/MultiHop, only chunks containing keywords survive
pub fn filter_by_keyword_predicate(
    chunks: Vec<String>,
    task: &TaskType,
    keywords: &[String],
) -> Vec<String> {
    match task {
        TaskType::Search | TaskType::MultiHop => {
            if keywords.is_empty() {
                return chunks;
            }
            let before = chunks.len();
            let filtered: Vec<String> = chunks
                .into_iter()
                .filter(|chunk| keyword_matches(chunk, keywords))
                .collect();
            tracing::debug!(
                survived = filtered.len(),
                total = before,
                "keyword pre-filter"
            );
            verbose!(
                "    PRUNE: {}/{} chunks survived keyword filter",
                filtered.len(),
                before
            );
            filtered
        }
        TaskType::Classify
        | TaskType::Aggregate
        | TaskType::Pairwise
        | TaskType::Summarise
        | TaskType::Auto => chunks,
    }
}

// ── Structural Chunker ──────────────────────────────────────────
// Replaces raw line-count splitting with definition-boundary splitting.
// Preserves semantic integrity: never cuts a function/struct/class in half.

/// Max line length before falling back to size-based chunking.
/// Minified JS or single-line 100MB files would cause unbounded processing time
/// in structural detection. Fall back to safe O(n) character splitting instead.
const MAX_LINE_LENGTH: usize = 16384;

/// Max chunks to prevent Vec allocation OOM before semaphore backpressure kicks in.
const MAX_CHUNK_COUNT: usize = 10_000;

/// PRE: text contains source files with "// === path ===" markers
/// POST: each chunk contains complete definitions, approx ≤ tau chars
pub fn structural_chunk(text: &str, tau: usize) -> Vec<String> {
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return vec![];
    }

    // Pathological input guard: if any line exceeds MAX_LINE_LENGTH,
    // bypass structural detection (minified JS, single-line files, etc.)
    if lines.iter().any(|l| l.len() > MAX_LINE_LENGTH) {
        tracing::warn!(
            max_line = lines.iter().map(|l| l.len()).max().unwrap_or(0),
            threshold = MAX_LINE_LENGTH,
            "pathological input detected, falling back to size-based chunking"
        );
        return comb_split(text, (text.len() / tau).max(2));
    }

    // 1. Find structural boundaries
    let mut boundaries = Vec::with_capacity(lines.len() / 20 + 1);
    boundaries.push(0);
    for (i, line) in lines.iter().enumerate().skip(1) {
        if is_definition_boundary(line) {
            boundaries.push(i);
        }
    }
    boundaries.push(lines.len());

    // 2. Group blocks into chunks respecting tau
    let mut chunks = Vec::new();
    let mut current = String::with_capacity(tau);

    for window in boundaries.windows(2) {
        let block: String = lines[window[0]..window[1]].join("\n");

        // Flush current chunk if adding this block would exceed tau
        if current.len() + block.len() + 1 > tau && !current.is_empty() {
            chunks.push(std::mem::take(&mut current));
            current = String::with_capacity(tau);
        }

        if !current.is_empty() {
            current.push('\n');
        }
        current.push_str(&block);
    }

    if !current.trim().is_empty() {
        chunks.push(current);
    }

    // 3. Fall back to line-based split if structural detection found nothing useful
    if chunks.is_empty() || (chunks.len() == 1 && text.len() > tau * 2) {
        tracing::warn!(
            chunks = chunks.len(),
            text_len = text.len(),
            tau,
            "structural chunking produced insufficient boundaries, degrading to line-based split"
        );
        return comb_split(text, (text.len() / tau).max(2));
    }

    // 4. Cap chunk count to prevent OOM on pathologically fragmented input
    if chunks.len() > MAX_CHUNK_COUNT {
        tracing::warn!(
            chunks = chunks.len(),
            max = MAX_CHUNK_COUNT,
            "chunk count exceeds limit, truncating"
        );
        chunks.truncate(MAX_CHUNK_COUNT);
    }

    chunks
}

/// Detect top-level definition boundaries across languages.
/// Heuristic: matches opening lines of functions, classes, structs, impls, etc.
fn is_definition_boundary(line: &str) -> bool {
    let t = line.trim();
    // File markers from collect_source_files
    (t.starts_with("// === ") && t.ends_with(" ==="))
    // Rust
    || t.starts_with("pub fn ")
    || t.starts_with("fn ")
    || t.starts_with("pub struct ")
    || t.starts_with("struct ")
    || t.starts_with("pub enum ")
    || t.starts_with("enum ")
    || t.starts_with("impl ")
    || t.starts_with("pub trait ")
    || t.starts_with("trait ")
    || t.starts_with("pub mod ")
    || t.starts_with("mod ")
    || t.starts_with("pub const ")
    || t.starts_with("pub static ")
    // Python
    || t.starts_with("def ")
    || t.starts_with("async def ")
    || t.starts_with("class ")
    // JS/TS
    || t.starts_with("function ")
    || t.starts_with("async function ")
    || t.starts_with("export function ")
    || t.starts_with("export default ")
    || t.starts_with("export class ")
    || t.starts_with("export interface ")
    || t.starts_with("export type ")
    // Go
    || t.starts_with("func ")
    || (t.starts_with("type ") && (t.contains(" struct") || t.contains(" interface")))
    // Java/Kotlin
    || t.starts_with("public class ")
    || t.starts_with("private class ")
    || t.starts_with("public fun ")
    || t.starts_with("fun ")
}
