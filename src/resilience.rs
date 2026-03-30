//! Resilience primitives: circuit breaker, call budget with RAII guard, replay cache.
//!
//! All state is lock-free (atomics). No Mutex, no contention.
//! Atomic orderings: Release-Acquire on all cross-thread state to ensure
//! correct visibility on ARM64 and other weakly-ordered architectures.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Wall-clock epoch milliseconds. Use only for timestamps in logs/telemetry.
/// NOT for interval measurement (vulnerable to NTP jumps).
#[allow(dead_code)]
pub fn epoch_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_millis() as u64
}

/// Monotonic milliseconds since process start. Safe for interval measurement
/// — immune to NTP jumps, clock drift, or wall-clock adjustments.
static BOOT: OnceLock<Instant> = OnceLock::new();

fn monotonic_ms() -> u64 {
    let boot = BOOT.get_or_init(Instant::now);
    boot.elapsed().as_millis() as u64
}

// ── Circuit Breaker ──────────────────────────────────────────────
// PRE: threshold > 0, cooldown > 0
// INV: consecutive_failures ∈ [0, ∞), monotonically increasing between resets
// POST (is_open): true iff failures >= threshold AND within cooldown window
//
// Uses monotonic clock for interval measurement — NTP jumps cannot
// cause premature cooldown reset or infinite open state.

pub struct CircuitBreaker {
    consecutive_failures: AtomicUsize,
    last_failure_mono_ms: AtomicU64,
    threshold: usize,
    cooldown_ms: u64,
}

impl CircuitBreaker {
    pub fn new(threshold: usize, cooldown: Duration) -> Self {
        // Ensure BOOT is initialized on first CircuitBreaker creation
        let _ = BOOT.get_or_init(Instant::now);
        Self {
            consecutive_failures: AtomicUsize::new(0),
            last_failure_mono_ms: AtomicU64::new(0),
            threshold,
            cooldown_ms: cooldown.as_millis() as u64,
        }
    }

    pub fn is_open(&self) -> bool {
        let failures = self.consecutive_failures.load(Ordering::Acquire);
        if failures < self.threshold {
            return false;
        }
        let last = self.last_failure_mono_ms.load(Ordering::Acquire);
        monotonic_ms() - last < self.cooldown_ms
    }

    pub fn failures(&self) -> usize {
        self.consecutive_failures.load(Ordering::Acquire)
    }

    pub fn record_success(&self) {
        self.consecutive_failures.store(0, Ordering::Release);
    }

    pub fn record_failure(&self) {
        self.consecutive_failures.fetch_add(1, Ordering::AcqRel);
        self.last_failure_mono_ms
            .store(monotonic_ms(), Ordering::Release);
    }
}

// ── Call Budget ──────────────────────────────────────────────────
// Hard cap on total LLM calls via lock-free CAS loop.
// PRE: max_calls = 0 means unlimited
// INV: remaining monotonically decreases (except release from BudgetGuard on panic)

pub struct CallBudget {
    remaining: AtomicUsize,
    unlimited: bool,
}

impl CallBudget {
    pub fn new(max_calls: usize) -> Self {
        Self {
            remaining: AtomicUsize::new(max_calls),
            unlimited: max_calls == 0,
        }
    }

    /// Atomically decrement remaining. Returns false if exhausted.
    pub fn try_acquire(&self) -> bool {
        if self.unlimited {
            return true;
        }
        loop {
            let current = self.remaining.load(Ordering::Acquire);
            if current == 0 {
                return false;
            }
            if self
                .remaining
                .compare_exchange_weak(
                    current,
                    current - 1,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
            {
                return true;
            }
        }
    }

    /// Restore one budget unit. Only called by BudgetGuard on panic/cancel.
    pub fn release(&self) {
        if !self.unlimited {
            self.remaining.fetch_add(1, Ordering::AcqRel);
        }
    }

    pub fn remaining(&self) -> usize {
        self.remaining.load(Ordering::Acquire)
    }

    pub fn is_unlimited(&self) -> bool {
        self.unlimited
    }
}

// ── Budget Guard (RAII) ──────────────────────────────────────────
// Acquires one budget unit. On normal exit (success or error), call commit().
// On panic/cancellation, Drop restores the unit — the call was never completed.
//
// PRE: caller has already called budget.try_acquire() and it returned true
// POST (commit): budget unit consumed, Drop is a no-op
// POST (drop without commit): budget unit restored

pub struct BudgetGuard<'a> {
    budget: &'a CallBudget,
    committed: bool,
}

impl<'a> BudgetGuard<'a> {
    pub fn new(budget: &'a CallBudget) -> Self {
        Self {
            budget,
            committed: false,
        }
    }

    /// Mark this budget unit as consumed. Must be called on all non-panic exits.
    pub fn commit(&mut self) {
        self.committed = true;
    }
}

impl Drop for BudgetGuard<'_> {
    fn drop(&mut self) {
        if !self.committed {
            self.budget.release();
        }
    }
}

// ── Replay Cache ─────────────────────────────────────────────────
// blake3 content-addressed, filesystem-backed.
// Hash(model + system + prompt + max_tokens) → cached response.
//
// PRE: dir is writable (created on init)
// POST (get): Some iff cache file exists and is readable
// POST (put): file written atomically via temp+rename (crash-safe)

pub struct ReplayCache {
    dir: PathBuf,
    enabled: bool,
    max_entries: usize, // 0 = unlimited
}

impl ReplayCache {
    pub fn new(dir: PathBuf, enabled: bool, max_entries: usize) -> Self {
        if enabled {
            if let Err(e) = std::fs::create_dir_all(&dir) {
                tracing::warn!(path = %dir.display(), error = %e, "failed to create cache directory");
            }
        }
        Self {
            dir,
            enabled,
            max_entries,
        }
    }

    pub fn key(model: &str, system: &str, user_prompt: &str, max_tokens: u32) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(model.as_bytes());
        hasher.update(b"\x00");
        hasher.update(system.as_bytes());
        hasher.update(b"\x00");
        hasher.update(user_prompt.as_bytes());
        hasher.update(b"\x00");
        hasher.update(&max_tokens.to_le_bytes());
        hasher.finalize().to_hex().to_string()
    }

    /// Deterministic plan cache key: hash of (input_size, task, alpha, user_k, user_tau).
    /// Plans with identical parameters always produce identical results.
    pub fn plan_key(
        input_size: usize,
        task: &str,
        alpha_bits: u64,
        user_k: usize,
        user_tau: usize,
    ) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"plan\x00");
        hasher.update(&input_size.to_le_bytes());
        hasher.update(b"\x00");
        hasher.update(task.as_bytes());
        hasher.update(b"\x00");
        hasher.update(&alpha_bits.to_le_bytes());
        hasher.update(b"\x00");
        hasher.update(&user_k.to_le_bytes());
        hasher.update(b"\x00");
        hasher.update(&user_tau.to_le_bytes());
        format!("plan_{}", hasher.finalize().to_hex())
    }

    pub fn get(&self, key: &str) -> Option<String> {
        if !self.enabled {
            return None;
        }
        std::fs::read_to_string(self.dir.join(key)).ok()
    }

    /// Evict oldest entries when max_entries is exceeded.
    /// Evicts down to 80% capacity to amortize the cost.
    fn evict_if_needed(&self) {
        if self.max_entries == 0 {
            return;
        }

        let Ok(read_dir) = std::fs::read_dir(&self.dir) else {
            return;
        };

        let entries: Vec<_> = read_dir
            .filter_map(|e| e.ok())
            .filter(|e| {
                let name = e.file_name();
                let s = name.to_string_lossy();
                !s.contains(".tmp.")
            })
            .collect();

        if entries.len() <= self.max_entries {
            return;
        }

        let target = self.max_entries * 4 / 5;
        let entry_count = entries.len();
        let to_remove = entry_count.saturating_sub(target);

        let mut by_mtime: Vec<_> = entries
            .into_iter()
            .filter_map(|e| {
                e.metadata()
                    .ok()
                    .and_then(|m| m.modified().ok())
                    .map(|t| (e.path(), t))
            })
            .collect();
        by_mtime.sort_by_key(|(_, t)| *t);

        let mut evicted = 0usize;
        for (path, _) in by_mtime.iter().take(to_remove) {
            if std::fs::remove_file(path).is_ok() {
                evicted += 1;
            }
        }

        if evicted > 0 {
            tracing::info!(
                evicted,
                remaining = entry_count - evicted,
                max = self.max_entries,
                "cache eviction"
            );
        }
    }

    /// Write to cache atomically: write to temp file, then rename.
    /// This ensures readers never see partial writes, even on crash.
    /// Errors are returned, not swallowed — callers decide how to handle.
    pub fn put(&self, key: &str, value: &str) -> anyhow::Result<()> {
        if !self.enabled {
            return Ok(());
        }
        self.evict_if_needed();
        let final_path = self.dir.join(key);
        let tmp_path = self.dir.join(format!("{}.tmp.{}", key, std::process::id()));

        // 1. Write to temp file
        std::fs::write(&tmp_path, value).map_err(|e| {
            anyhow::anyhow!("cache write to temp file failed: {e}")
        })?;

        // 2. Atomic rename (POSIX guarantees atomicity of rename within same filesystem)
        std::fs::rename(&tmp_path, &final_path).map_err(|e| {
            // Clean up temp file on rename failure
            let _ = std::fs::remove_file(&tmp_path);
            anyhow::anyhow!("cache rename failed: {e}")
        })?;

        // 3. Fsync the directory to ensure rename is durable
        if let Ok(dir) = std::fs::File::open(&self.dir) {
            let _ = dir.sync_all(); // Best-effort fsync on directory
        }

        Ok(())
    }

    #[allow(dead_code)]
    pub fn enabled(&self) -> bool {
        self.enabled
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn circuit_breaker_below_threshold_stays_closed() {
        let cb = CircuitBreaker::new(3, Duration::from_secs(30));
        cb.record_failure();
        cb.record_failure();
        assert!(!cb.is_open());
        assert_eq!(cb.failures(), 2);
    }

    #[test]
    fn circuit_breaker_at_threshold_opens() {
        let cb = CircuitBreaker::new(3, Duration::from_secs(30));
        cb.record_failure();
        cb.record_failure();
        cb.record_failure();
        assert!(cb.is_open());
        assert_eq!(cb.failures(), 3);
    }

    #[test]
    fn circuit_breaker_success_resets() {
        let cb = CircuitBreaker::new(2, Duration::from_secs(30));
        cb.record_failure();
        cb.record_failure();
        assert!(cb.is_open());
        cb.record_success();
        assert!(!cb.is_open());
        assert_eq!(cb.failures(), 0);
    }

    #[test]
    fn circuit_breaker_cooldown_closes() {
        let cb = CircuitBreaker::new(1, Duration::from_millis(10));
        cb.record_failure();
        assert!(cb.is_open());
        thread::sleep(Duration::from_millis(20));
        assert!(!cb.is_open());
    }

    #[test]
    fn call_budget_exhaustion() {
        let budget = CallBudget::new(2);
        assert!(budget.try_acquire());
        assert!(budget.try_acquire());
        assert!(!budget.try_acquire());
        assert_eq!(budget.remaining(), 0);
    }

    #[test]
    fn call_budget_release_restores() {
        let budget = CallBudget::new(1);
        assert!(budget.try_acquire());
        assert!(!budget.try_acquire());
        budget.release();
        assert!(budget.try_acquire());
    }

    #[test]
    fn call_budget_unlimited() {
        let budget = CallBudget::new(0);
        assert!(budget.is_unlimited());
        for _ in 0..1000 {
            assert!(budget.try_acquire());
        }
    }

    #[test]
    fn call_budget_concurrent_cas() {
        let budget = Arc::new(CallBudget::new(100));
        let mut handles = Vec::new();
        for _ in 0..10 {
            let b = Arc::clone(&budget);
            handles.push(thread::spawn(move || {
                let mut acquired = 0;
                for _ in 0..20 {
                    if b.try_acquire() {
                        acquired += 1;
                    }
                }
                acquired
            }));
        }
        let total: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();
        assert_eq!(total, 100);
        assert_eq!(budget.remaining(), 0);
    }

    #[test]
    fn budget_guard_commit_consumes() {
        let budget = CallBudget::new(1);
        assert!(budget.try_acquire());
        {
            let mut guard = BudgetGuard::new(&budget);
            guard.commit();
        }
        assert_eq!(budget.remaining(), 0);
    }

    #[test]
    fn budget_guard_drop_restores() {
        let budget = CallBudget::new(1);
        assert!(budget.try_acquire());
        {
            let _guard = BudgetGuard::new(&budget);
            // drop without commit
        }
        assert_eq!(budget.remaining(), 1);
    }

    #[test]
    fn replay_cache_atomic_write() {
        let dir = std::env::temp_dir().join(format!("rlm_test_{}", std::process::id()));
        let cache = ReplayCache::new(dir.clone(), true, 0);

        cache.put("test_key", "test_value").unwrap();
        assert_eq!(cache.get("test_key"), Some("test_value".to_string()));

        // No temp files left behind
        let temps: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains(".tmp."))
            .collect();
        assert!(temps.is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn replay_cache_disabled() {
        let cache = ReplayCache::new(PathBuf::from("/nonexistent"), false, 0);
        assert!(cache.put("key", "value").is_ok());
        assert_eq!(cache.get("key"), None);
    }

    #[test]
    fn replay_cache_key_deterministic() {
        let k1 = ReplayCache::key("model", "sys", "prompt", 100);
        let k2 = ReplayCache::key("model", "sys", "prompt", 100);
        let k3 = ReplayCache::key("model", "sys", "prompt", 101);
        assert_eq!(k1, k2);
        assert_ne!(k1, k3);
    }

    #[test]
    fn replay_cache_eviction() {
        let dir = std::env::temp_dir().join(format!("rlm_evict_{}", std::process::id()));
        let cache = ReplayCache::new(dir.clone(), true, 5);

        // Write 8 entries — should trigger eviction on the later puts
        for i in 0..8 {
            cache.put(&format!("key_{i}"), &format!("value_{i}")).unwrap();
            // Small sleep to ensure distinct mtimes for eviction ordering
            thread::sleep(Duration::from_millis(10));
        }

        // Eviction fires before each write, so worst case is max_entries + 1
        let count = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .count();
        assert!(count <= 6, "expected at most max_entries+1=6, got {count}");

        // Most recent entries should survive
        assert!(cache.get("key_7").is_some());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
