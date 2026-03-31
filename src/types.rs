use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use std::fmt;

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
