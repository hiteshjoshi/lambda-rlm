//! Combinator Library L — Table 1 from the paper.
//!
//! Every combinator except M is total and deterministic (Assumption A2).
//! Plus: structural chunker, keyword utilities, text processing.

use crate::types::TaskType;

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
        let chunk: String = lines[start..end].join("\n");
        if !chunk.trim().is_empty() {
            chunks.push(chunk);
        }
    }
    chunks
}

/// Peek: Σ* × N² → Σ*
/// Safe substring on char boundaries.
pub fn comb_peek(text: &str, start: usize, end: usize) -> &str {
    let s = start.min(text.len());
    let e = end.min(text.len());
    let s = if s == 0 {
        0
    } else {
        let mut i = s;
        while i < text.len() && !text.is_char_boundary(i) {
            i += 1;
        }
        i
    };
    let e = {
        let mut i = e;
        while i > 0 && !text.is_char_boundary(i) {
            i -= 1;
        }
        i
    };
    &text[s..e]
}

/// Cross: [α] × [α] → [(α, α)]
/// Pairs from DIFFERENT groups only. Purely symbolic, zero neural cost.
pub fn comb_cross(groups: &[Vec<String>]) -> Vec<(String, String)> {
    let mut pairs = Vec::new();
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
        .map(|w| {
            w.trim_matches(|c: char| !c.is_alphanumeric())
                .to_string()
        })
        .filter(|w| !w.is_empty())
        .collect()
}

pub fn keyword_matches(text: &str, keywords: &[String]) -> bool {
    let lower = text.to_lowercase();
    keywords.iter().any(|kw| lower.contains(kw.as_str()))
}

/// Parse markdown-style list items from LLM output.
/// Handles: `- item`, `* item`, `• item`, `1. item`, `  - nested`, and plain lines.
/// Strips leading list markers and numbering, then filters short/empty lines.
///
/// PRE: text is UTF-8 LLM output (may contain any Unicode)
/// POST: each item is trimmed, non-empty, len > 3
pub fn parse_items(text: &str) -> Vec<String> {
    text.lines()
        .map(|l| {
            let t = l.trim();
            // Strip bullet markers: -, *, •, ▪, ▸, ►
            let t = t.strip_prefix("- ")
                .or_else(|| t.strip_prefix("* "))
                .or_else(|| t.strip_prefix("• "))
                .or_else(|| t.strip_prefix("▪ "))
                .or_else(|| t.strip_prefix("▸ "))
                .or_else(|| t.strip_prefix("► "))
                .unwrap_or(t);
            // Strip numbered list markers: "1. ", "2) ", etc.
            let t = if t.len() > 2 {
                let bytes = t.as_bytes();
                if bytes[0].is_ascii_digit() {
                    // Find end of digits
                    let digit_end = t.bytes().take_while(|b| b.is_ascii_digit()).count();
                    if digit_end < t.len() {
                        let after_digits = &t[digit_end..];
                        if let Some(rest) = after_digits.strip_prefix(". ") {
                            rest
                        } else if let Some(rest) = after_digits.strip_prefix(") ") {
                            rest
                        } else {
                            t
                        }
                    } else {
                        t
                    }
                } else {
                    t
                }
            } else {
                t
            };
            t.trim().to_string()
        })
        .filter(|l| !l.is_empty() && l.len() > 3)
        .collect()
}

pub fn merge_dedup(items: Vec<String>) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    items
        .into_iter()
        .filter(|item| {
            let key: String = item
                .to_lowercase()
                .chars()
                .filter(|c| c.is_alphanumeric())
                .collect();
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
            tracing::debug!(survived = filtered.len(), total = before, "keyword pre-filter");
            eprintln!(
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

/// PRE: text contains source files with "// === path ===" markers
/// POST: each chunk contains complete definitions, approx ≤ tau chars
pub fn structural_chunk(text: &str, tau: usize) -> Vec<String> {
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return vec![];
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
        return comb_split(text, (text.len() / tau).max(2));
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
