use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Global error taxonomy for programmatic handling across 30-year maintenance.
/// Every error in the system maps to one of these kinds, enabling main.rs to
/// dispatch specific recovery actions (scrub cache, alert operator, retry, etc.)
/// instead of pattern-matching on anyhow string messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum ErrorKind {
    /// Cache integrity failure (bitrot, checksum mismatch). Recovery: scrub cache.
    Corruption,
    /// Disk full, OOM, file descriptor exhaustion. Recovery: alert operator.
    ResourceExhausted,
    /// Call budget depleted. Recovery: graceful shutdown, return partial results.
    BudgetDepleted,
    /// Transient network error (timeout, rate limit, 5xx). Recovery: retry/backoff.
    TransientNetwork,
    /// Permanent LLM failure (auth, 4xx). Recovery: alert operator, stop pipeline.
    PermanentLLMFailure,
    /// Cache schema mismatch. Recovery: migrate or recompute.
    SchemaIncompatible,
    /// Cooperative shutdown requested. Recovery: drain and exit.
    Shutdown,
}

#[derive(Clone, ValueEnum, Debug, PartialEq, Serialize, Deserialize)]
pub enum TaskType {
    Auto,
    Search,
    Classify,
    Aggregate,
    Pairwise,
    Summarise,
    #[value(name = "multi-hop")]
    MultiHop,
}

impl fmt::Display for TaskType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => write!(f, "auto"),
            Self::Search => write!(f, "search"),
            Self::Classify => write!(f, "classify"),
            Self::Aggregate => write!(f, "aggregate"),
            Self::Pairwise => write!(f, "pairwise"),
            Self::Summarise => write!(f, "summarise"),
            Self::MultiHop => write!(f, "multi_hop"),
        }
    }
}
