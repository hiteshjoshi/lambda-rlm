//! Verifier — replaces filter_useful with task-specific validation.
//!
//! Applied to every leaf M output before it enters the reduce pipeline.
//! Catches refusals, empty outputs, and task-specific quality issues
//! that filter_useful missed (compression ratio, keyword presence, etc).

use crate::combinator::keyword_matches;
use crate::types::TaskType;

pub enum VerifyResult {
    /// Output passed all checks — use as-is.
    Accept(String),
    /// Output is a refusal or garbage — discard entirely.
    Reject(&'static str),
    /// Output is questionable but usable — include with warning.
    Degraded(String, &'static str),
}

pub struct Verifier {
    task: TaskType,
    keywords: Vec<String>,
}

impl Verifier {
    pub fn new(task: TaskType, keywords: Vec<String>) -> Self {
        Self { task, keywords }
    }

    /// PRE: output is raw text from an LLM leaf call
    /// POST: Accept iff output passes universal + task-specific validation
    pub fn check(&self, output: &str, input_len: usize) -> VerifyResult {
        let trimmed = output.trim();

        // Universal: empty output
        if trimmed.is_empty() {
            return VerifyResult::Reject("empty output");
        }

        // Universal: refusal detection (hard sentinels + soft refusals)
        if is_refusal(trimmed) {
            return VerifyResult::Reject("refusal detected");
        }

        // Task-specific validation — exhaustive match, no wildcards
        match self.task {
            TaskType::Search => {
                if !self.keywords.is_empty()
                    && !keyword_matches(trimmed, &self.keywords)
                {
                    return VerifyResult::Degraded(
                        trimmed.to_string(),
                        "output lacks search keywords",
                    );
                }
                VerifyResult::Accept(trimmed.to_string())
            }

            TaskType::Summarise => {
                // Compression ratio: summary should be < 50% of input.
                // If the LLM echoed most of the input, the summary is useless.
                if input_len > 100 && trimmed.len() > input_len / 2 {
                    return VerifyResult::Degraded(
                        trimmed.to_string(),
                        "summary not compressed (>50% of input)",
                    );
                }
                VerifyResult::Accept(trimmed.to_string())
            }

            TaskType::Classify => {
                if trimmed.len() < 10 {
                    return VerifyResult::Degraded(
                        trimmed.to_string(),
                        "classification too terse",
                    );
                }
                VerifyResult::Accept(trimmed.to_string())
            }

            TaskType::Aggregate => {
                VerifyResult::Accept(trimmed.to_string())
            }

            TaskType::Pairwise => {
                VerifyResult::Accept(trimmed.to_string())
            }

            TaskType::MultiHop => {
                if !self.keywords.is_empty()
                    && !keyword_matches(trimmed, &self.keywords)
                {
                    return VerifyResult::Degraded(
                        trimmed.to_string(),
                        "evidence lacks query keywords",
                    );
                }
                VerifyResult::Accept(trimmed.to_string())
            }

            TaskType::Auto => unreachable!("Auto resolved before verification"),
        }
    }
}

/// Detects both hard sentinels (NO_MATCH, NO_DATA) and soft refusals.
/// Consolidated from the original filter_useful patterns.
fn is_refusal(text: &str) -> bool {
    let l = text.to_lowercase();
    l.starts_with("no_match")
        || l.starts_with("no_data")
        || l.contains("no relevant information")
        || l.contains("nothing found")
        || l.contains("no issues found")
        || l.contains("not present in this section")
        || l.contains("does not contain")
        || l.contains("no security vulnerabilities")
        || l.contains("no vulnerabilities were")
        || l.contains("no items found")
        || l.contains("no entities found")
        || l.contains("insufficient context")
        || l.contains("cannot determine")
        || l.contains("unable to find")
        || l.contains("no evidence")
        || l.contains("not enough information")
        || l.contains("no relevant code")
        || l.contains("nothing relevant")
}
