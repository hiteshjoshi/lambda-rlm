//! Cost model (Definitions 3-4, Theorems 2-4) and optimal planner.
//!
//! compute_plan finds (k*, τ*, d) that minimizes cost while meeting
//! accuracy target α, per Theorem 4.

use crate::resilience::ReplayCache;
use crate::types::TaskType;
use serde::{Deserialize, Serialize};

pub struct CostModel {
    c_in: f64,
    c_out: f64,
    n_out_avg: f64,
    a0: f64,
    rho: f64,
    context_window: usize,
}

impl CostModel {
    pub fn new(rho: f64, context_window: usize) -> Self {
        Self {
            c_in: 1.0,
            c_out: 3.0,
            n_out_avg: 500.0,
            a0: 1.0,
            rho: rho.clamp(0.01, 1.0),
            context_window,
        }
    }

    /// C(n) = c_in · n + c_out · n̄_out  (Definition 3)
    pub fn cost_leaf(&self, n: usize) -> f64 {
        self.c_in * n as f64 + self.c_out * self.n_out_avg
    }

    /// A(n) = A₀ · ρ^(n/K)  (Definition 4)
    pub fn accuracy_at(&self, n: usize) -> f64 {
        self.a0 * self.rho.powf(n as f64 / self.context_window as f64)
    }

    fn cost_reduce(&self, k: usize, neural: bool) -> f64 {
        if neural {
            self.c_in * (k as f64 * self.n_out_avg) + self.c_out * self.n_out_avg
        } else {
            0.0
        }
    }

    /// Total cost T(n) — Theorem 2, Equation 7
    pub fn total_cost(&self, n: usize, k: usize, tau: usize, neural_reduce: bool) -> f64 {
        if n <= tau {
            return self.cost_leaf(n);
        }
        let nf = n as f64;
        let kf = k as f64;
        let tf = tau as f64;
        let leaf_cost = (nf / tf) * self.cost_leaf(tau);
        if !neural_reduce || k <= 1 {
            return leaf_cost;
        }
        let c_reduce = self.cost_reduce(k, true);
        let reduce_cost = c_reduce * (nf * kf - tf) / (tf * (kf - 1.0));
        leaf_cost + reduce_cost
    }

    /// End-to-end accuracy — Theorem 3
    pub fn total_accuracy(&self, n: usize, k: usize, tau: usize, neural_reduce: bool) -> f64 {
        if n <= tau {
            return self.accuracy_at(n);
        }
        let d = ((n as f64 / tau as f64).ln() / (k as f64).ln()).ceil();
        let a_reduce: f64 = if neural_reduce { 0.95 } else { 1.0 };
        let a_leaf = self.accuracy_at(tau);
        let k_pow_d = (k as f64).powf(d);
        let leaf_acc = if k_pow_d > 1000.0 {
            a_leaf.powf(k_pow_d).max(1e-10)
        } else {
            a_leaf.powf(k_pow_d)
        };
        a_reduce.powf(d) * leaf_acc
    }
}

#[derive(Serialize, Deserialize)]
pub struct Plan {
    pub k: usize,
    pub tau: usize,
    pub depth: usize,
    pub leaf_calls: usize,
    pub reduce_calls: usize,
    pub total_calls: usize,
    pub estimated_cost: f64,
    pub estimated_accuracy: f64,
    pub neural_reduce: bool,
}

pub fn is_neural_reduce(task: &TaskType) -> bool {
    matches!(task, TaskType::Summarise | TaskType::MultiHop)
}

pub fn composition_desc(task: &TaskType) -> &'static str {
    match task {
        TaskType::Search => "FilterBest (deterministic)",
        TaskType::Classify => "Concat (deterministic)",
        TaskType::Aggregate => "Merge+Dedup (deterministic)",
        TaskType::Pairwise => "Cross+Filter -> M(analyze)",
        TaskType::Summarise => "M(Concat(children)) (neural at every level)",
        TaskType::MultiHop => "M(Concat(evidence)) (neural synthesis)",
        TaskType::Auto => "-",
    }
}

pub fn pipeline_desc(task: &TaskType) -> &'static str {
    match task {
        TaskType::Search => "Split -> Peek -> Filter(keyword) -> Map(M) -> FilterBest",
        TaskType::Classify => "Split -> Map(M:classify) -> Concat",
        TaskType::Aggregate => "Split -> Map(M:extract) -> Merge(dedup)",
        TaskType::Pairwise => "Split -> Map(M:extract) -> Cross -> M(compare)",
        TaskType::Summarise => "Split -> Map(M:summarize) -> M(synthesize)  [recursive]",
        TaskType::MultiHop => "Split_d -> Peek -> Filter -> Map(M:evidence) -> M(chain)",
        TaskType::Auto => "-",
    }
}

/// Phase 4: Optimal (k*, τ*, d) via Theorem 4.
///
/// PRE: input_size > 0, cost_model initialized
/// POST: Plan with k >= 2, tau >= 1000, depth >= 0
///
/// Uses deterministic parameters only (no LLM sampling). Results are
/// cached via ReplayCache keyed by (input_size, task_type, alpha, k, tau).
pub fn compute_plan(
    input_size: usize,
    task: &TaskType,
    cost_model: &CostModel,
    alpha: f64,
    user_k: usize,
    user_tau: usize,
    cache: Option<&ReplayCache>,
) -> Plan {
    let neural = is_neural_reduce(task);

    // Check plan cache
    let cache_key = ReplayCache::plan_key(
        input_size,
        &task.to_string(),
        alpha.to_bits(),
        user_k,
        user_tau,
    );
    if let Some(cache) = cache {
        if let Some(cached) = cache.get(&cache_key) {
            if let Ok(plan) = serde_json::from_str::<Plan>(&cached) {
                eprintln!("  Plan loaded from cache (deterministic replay)");
                return plan;
            }
        }
    }

    let tau_default = if user_tau > 0 { user_tau } else { 6000 };
    if input_size <= tau_default {
        let plan = Plan {
            k: 2,
            tau: input_size,
            depth: 0,
            leaf_calls: 1,
            reduce_calls: 0,
            total_calls: 1,
            estimated_cost: cost_model.cost_leaf(input_size),
            estimated_accuracy: cost_model.accuracy_at(input_size),
            neural_reduce: neural,
        };
        if let Some(cache) = cache {
            if let Ok(json) = serde_json::to_string(&plan) {
                if let Err(e) = cache.put(&cache_key, &json) {
                    tracing::warn!(error = %e, "plan cache write failed (non-fatal)");
                }
            }
        }
        return plan;
    }

    let tau = if user_tau > 0 {
        user_tau
    } else {
        let target = alpha.sqrt();
        let mut t = cost_model.context_window;
        while t > 1000 && cost_model.accuracy_at(t) < target {
            t -= 500;
        }
        t.max(1000).min(cost_model.context_window)
    };

    let k = if user_k > 0 {
        user_k
    } else {
        let mut best_k = 2usize;
        let mut best_cost = f64::MAX;
        for kc in 2..=16 {
            let cost = cost_model.total_cost(input_size, kc, tau, neural);
            let acc = cost_model.total_accuracy(input_size, kc, tau, neural);
            if acc >= alpha && cost < best_cost {
                best_cost = cost;
                best_k = kc;
            }
        }
        if best_cost == f64::MAX {
            2
        } else {
            best_k
        }
    };

    let depth = ((input_size as f64 / tau as f64).ln() / (k as f64).ln()).ceil() as usize;
    let leaf_calls = (k as f64).powi(depth as i32) as usize;

    let reduce_calls = if neural && depth > 0 {
        let mut internal = 0usize;
        for d in 0..depth {
            internal += (k as f64).powi(d as i32) as usize;
        }
        internal
    } else if matches!(task, TaskType::Pairwise) {
        1
    } else {
        0
    };

    let total_calls = leaf_calls + reduce_calls;
    let estimated_cost = cost_model.total_cost(input_size, k, tau, neural);
    let estimated_accuracy = cost_model.total_accuracy(input_size, k, tau, neural);

    let plan = Plan {
        k,
        tau,
        depth,
        leaf_calls,
        reduce_calls,
        total_calls,
        estimated_cost,
        estimated_accuracy,
        neural_reduce: neural,
    };

    // Cache the plan for deterministic replay
    if let Some(cache) = cache {
        if let Ok(json) = serde_json::to_string(&plan) {
            if let Err(e) = cache.put(&cache_key, &json) {
                tracing::warn!(error = %e, "plan cache write failed (non-fatal)");
            }
        }
    }

    plan
}
