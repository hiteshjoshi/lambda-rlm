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
    /// Returns finite cost; saturates to f64::MAX / 2.0 on overflow/NaN.
    ///
    /// Reduce cost depends on neural type:
    /// - Symbolic: 0
    /// - Depth0Only: 1 × c_reduce (single neural call at root)
    /// - AllDepths: c_reduce × (n/τ - 1)/(k - 1) (every internal node)
    pub fn total_cost(&self, n: usize, k: usize, tau: usize, rnt: ReduceNeuralType) -> f64 {
        if n <= tau {
            return self.cost_leaf(n);
        }
        let nf = n as f64;
        let kf = k as f64;
        let tf = tau as f64;
        let leaf_cost = (nf / tf) * self.cost_leaf(tau);
        let reduce_cost = match rnt {
            ReduceNeuralType::Symbolic => 0.0,
            ReduceNeuralType::Depth0Only => self.cost_reduce(k, true),
            ReduceNeuralType::AllDepths => {
                if k <= 1 {
                    0.0
                } else {
                    let c_reduce = self.cost_reduce(k, true);
                    // Paper recurrence: (n/τ - 1)/(k - 1) internal nodes in a k-ary tree
                    c_reduce * (nf / tf - 1.0) / (kf - 1.0)
                }
            }
        };
        let total = leaf_cost + reduce_cost;
        if total.is_finite() {
            total
        } else {
            f64::MAX / 2.0
        }
    }

    /// End-to-end accuracy — Theorem 3
    /// Returns value in [0, 1]; clamps non-finite intermediates to prevent NaN propagation.
    ///
    /// Accuracy penalty per neural reduce:
    /// - Symbolic: 1.0 (no degradation)
    /// - Depth0Only: 0.95^1 (one neural step)
    /// - AllDepths: 0.95^d (neural at each depth level)
    pub fn total_accuracy(&self, n: usize, k: usize, tau: usize, rnt: ReduceNeuralType) -> f64 {
        if n <= tau {
            return self.accuracy_at(n);
        }
        let d = ((n as f64 / tau as f64).ln() / (k as f64).ln()).ceil();
        if !d.is_finite() || d < 0.0 {
            return 0.0;
        }
        let a_reduce_total: f64 = match rnt {
            ReduceNeuralType::Symbolic => 1.0,
            ReduceNeuralType::Depth0Only => 0.95,
            ReduceNeuralType::AllDepths => 0.95_f64.powf(d),
        };
        let a_leaf = self.accuracy_at(tau);
        let k_pow_d = (k as f64).powf(d);
        let leaf_acc = if !k_pow_d.is_finite() || k_pow_d > 1000.0 {
            a_leaf.powf(k_pow_d.min(1e6)).max(1e-10)
        } else {
            a_leaf.powf(k_pow_d)
        };
        let result = a_reduce_total * leaf_acc;
        if result.is_finite() {
            result
        } else {
            0.0
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct Plan {
    #[serde(default = "default_k")]
    pub k: usize,
    #[serde(default)]
    pub tau: usize,
    #[serde(default)]
    pub depth: usize,
    #[serde(default)]
    pub leaf_calls: usize,
    #[serde(default)]
    pub reduce_calls: usize,
    #[serde(default)]
    pub total_calls: usize,
    #[serde(default)]
    pub estimated_cost: f64,
    #[serde(default)]
    pub estimated_accuracy: f64,
    #[serde(default)]
    pub neural_reduce: bool,
    /// blake3 checksum of the serialized plan fields (excluding this field).
    /// Verified on cache load to detect silent corruption of cached plans.
    #[serde(default)]
    pub checksum: String,
}

impl Plan {
    /// Compute blake3 checksum over deterministic plan fields.
    /// Excludes the checksum field itself to avoid circular dependency.
    fn compute_checksum(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&self.k.to_le_bytes());
        hasher.update(&self.tau.to_le_bytes());
        hasher.update(&self.depth.to_le_bytes());
        hasher.update(&self.leaf_calls.to_le_bytes());
        hasher.update(&self.reduce_calls.to_le_bytes());
        hasher.update(&self.total_calls.to_le_bytes());
        hasher.update(&self.estimated_cost.to_bits().to_le_bytes());
        hasher.update(&self.estimated_accuracy.to_bits().to_le_bytes());
        hasher.update(&[self.neural_reduce as u8]);
        hasher.finalize().to_hex()[..16].to_string()
    }

    /// Verify the checksum matches the plan fields.
    /// Returns true if checksum is empty (legacy plans without checksum) or matches.
    fn verify_checksum(&self) -> bool {
        if self.checksum.is_empty() {
            return true; // Legacy plan without checksum — accept and re-save with checksum
        }
        self.checksum == self.compute_checksum()
    }
}

fn default_k() -> usize {
    2
}

/// Classifies the neural cost profile of each task's reduce operator.
/// - Symbolic: zero neural calls (Classify, Aggregate)
/// - Depth0Only: one neural call at depth 0 (Search=FilterBest, Pairwise=neural comparison)
/// - AllDepths: neural call at every internal node (Summarise, MultiHop)
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ReduceNeuralType {
    Symbolic,
    Depth0Only,
    AllDepths,
}

pub fn reduce_neural_type(task: &TaskType) -> ReduceNeuralType {
    match task {
        TaskType::Search | TaskType::Pairwise => ReduceNeuralType::Depth0Only,
        TaskType::Summarise | TaskType::MultiHop => ReduceNeuralType::AllDepths,
        TaskType::Classify | TaskType::Aggregate | TaskType::Auto => ReduceNeuralType::Symbolic,
    }
}

/// Backward-compatible wrapper: returns true if any neural reduce calls exist.
#[allow(dead_code)]
pub fn is_neural_reduce(task: &TaskType) -> bool {
    reduce_neural_type(task) != ReduceNeuralType::Symbolic
}

pub fn composition_desc(task: &TaskType) -> &'static str {
    match task {
        TaskType::Search => "FilterBest (neural at depth 0)",
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
/// cached via ReplayCache keyed by (input_size, task_type, alpha, k, tau,
/// model, context_window). Model fingerprint ensures cache invalidation
/// when the underlying LLM changes.
pub fn compute_plan(
    input_size: usize,
    task: &TaskType,
    cost_model: &CostModel,
    alpha: f64,
    user_k: usize,
    user_tau: usize,
    cache: Option<&ReplayCache>,
    model: &str,
    context_window: usize,
) -> Plan {
    let rnt = reduce_neural_type(task);
    let neural = rnt != ReduceNeuralType::Symbolic;

    // Check plan cache (includes model fingerprint to invalidate on model change)
    let cache_key = ReplayCache::plan_key(
        input_size,
        &task.to_string(),
        alpha.to_bits(),
        user_k,
        user_tau,
        model,
        context_window,
    );
    if let Some(cache) = cache {
        if let Some(cached) = cache.get(&cache_key) {
            if let Ok(plan) = serde_json::from_str::<Plan>(&cached) {
                if plan.verify_checksum() {
                    verbose!("  Plan loaded from cache (deterministic replay)");
                    return plan;
                }
                tracing::warn!("cached plan checksum mismatch, recomputing");
                verbose!("  Plan cache checksum mismatch — recomputing");
            }
        }
    }

    let tau_default = if user_tau > 0 { user_tau } else { 6000 };
    if input_size <= tau_default {
        let mut plan = Plan {
            k: 2,
            tau: input_size,
            depth: 0,
            leaf_calls: 1,
            reduce_calls: 0,
            total_calls: 1,
            estimated_cost: cost_model.cost_leaf(input_size),
            estimated_accuracy: cost_model.accuracy_at(input_size),
            neural_reduce: neural,
            checksum: String::new(),
        };
        plan.checksum = plan.compute_checksum();
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

    // Ensure k >= 2: k=1 causes division by zero in depth calculation
    // (ln(1) = 0), and k=0 is nonsensical. Clamp user values.
    let k = if user_k >= 2 {
        user_k
    } else if user_k == 1 {
        2
    } else {
        let mut best_k = 2usize;
        let mut best_cost = f64::MAX;
        for kc in 2..=16 {
            let cost = cost_model.total_cost(input_size, kc, tau, rnt);
            let acc = cost_model.total_accuracy(input_size, kc, tau, rnt);
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

    // Cap depth to prevent pathological expansion. k=2, depth=64 means 2^64
    // chunks — physically impossible. Use powf (not powi) to avoid i32 truncation.
    let depth = {
        let raw = ((input_size as f64 / tau as f64).ln() / (k as f64).ln()).ceil();
        if raw.is_nan() || raw.is_infinite() || raw < 0.0 {
            1
        } else {
            (raw as usize).min(64)
        }
    };

    let leaf_calls = {
        let raw = (k as f64).powf(depth as f64);
        if raw > usize::MAX as f64 {
            usize::MAX
        } else {
            raw as usize
        }
    };

    let reduce_calls = match rnt {
        ReduceNeuralType::AllDepths if depth > 0 => {
            // Neural at every internal node: sum of k^d for d in 0..depth
            let mut internal = 0usize;
            for d in 0..depth {
                let level = (k as f64).powf(d as f64);
                let level_usize = if level > usize::MAX as f64 {
                    usize::MAX
                } else {
                    level as usize
                };
                internal = internal.saturating_add(level_usize);
            }
            internal
        }
        ReduceNeuralType::Depth0Only => 1, // Single neural call at root
        _ => 0,
    };

    let total_calls = leaf_calls.saturating_add(reduce_calls);
    let estimated_cost = cost_model.total_cost(input_size, k, tau, rnt);
    let estimated_accuracy = cost_model.total_accuracy(input_size, k, tau, rnt);

    let mut plan = Plan {
        k,
        tau,
        depth,
        leaf_calls,
        reduce_calls,
        total_calls,
        estimated_cost,
        estimated_accuracy,
        neural_reduce: neural,
        checksum: String::new(),
    };
    plan.checksum = plan.compute_checksum();

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
