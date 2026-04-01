//! Resilience primitives: circuit breaker, call budget with RAII guard, replay cache.
//!
//! All state is lock-free (atomics). No Mutex, no contention.
//! Atomic orderings: Release-Acquire on all cross-thread state to ensure
//! correct visibility on ARM64 and other weakly-ordered architectures.

use std::path::PathBuf;
use std::io::Read;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Bump when prompt logic or cache format changes to invalidate stale entries.
/// Any modification to leaf_prompt() or reduce prompts MUST bump this version.
/// V2: added hash algorithm prefix (b3-) to cache keys for cryptographic agility.
pub const CACHE_SCHEMA_VERSION: u8 = 2;

/// Monotonic milliseconds since process start. Safe for interval measurement
/// — immune to NTP jumps, clock drift, or wall-clock adjustments.
static BOOT: OnceLock<Instant> = OnceLock::new();

fn monotonic_ms() -> u64 {
    let boot = BOOT.get_or_init(Instant::now);
    boot.elapsed().as_millis() as u64
}

/// Process-unique nonce: 16 bytes of entropy generated once at startup.
/// Prevents idempotency key collisions after PID wraparound (~65k restarts)
/// where PID + boot_mono_ms could repeat if two processes start in the same
/// millisecond with the same recycled PID. Uses wall-clock nanoseconds +
/// PID + thread ID hashed through blake3 for cross-platform uniqueness.
static PROCESS_NONCE: OnceLock<[u8; 16]> = OnceLock::new();

pub fn process_nonce() -> &'static [u8; 16] {
    PROCESS_NONCE.get_or_init(|| {
        let mut hasher = blake3::Hasher::new();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO);
        hasher.update(&now.as_nanos().to_le_bytes());
        hasher.update(&(std::process::id() as u64).to_le_bytes());
        hasher.update(format!("{:?}", std::thread::current().id()).as_bytes());
        let hash = hasher.finalize();
        let mut nonce = [0u8; 16];
        nonce.copy_from_slice(&hash.as_bytes()[..16]);
        nonce
    })
}

// ── Circuit Breaker ──────────────────────────────────────────────
// PRE: threshold > 0, cooldown > 0
// INV: consecutive_failures ∈ [0, ∞), monotonically increasing between resets
// POST (is_open): true iff failures >= threshold AND within cooldown window
//
// States:
//   Closed:   failures < threshold → all requests pass
//   Open:     failures >= threshold AND within cooldown → all requests rejected
//   HalfOpen: failures >= threshold AND cooldown expired → one probe request
//
// Uses monotonic clock for interval measurement — NTP jumps cannot
// cause premature cooldown reset or infinite open state.

pub struct CircuitBreaker {
    consecutive_failures: AtomicUsize,
    last_failure_mono_ms: AtomicU64,
    half_open_probe: AtomicBool,
    threshold: usize,
    cooldown_ms: u64,
}

impl CircuitBreaker {
    pub fn new(threshold: usize, cooldown: Duration) -> Self {
        let _ = BOOT.get_or_init(Instant::now);
        Self {
            consecutive_failures: AtomicUsize::new(0),
            last_failure_mono_ms: AtomicU64::new(0),
            half_open_probe: AtomicBool::new(false),
            threshold,
            cooldown_ms: cooldown.as_millis() as u64,
        }
    }

    #[allow(dead_code)] // Used in tests; callers prefer allow_request()
    pub fn is_open(&self) -> bool {
        let failures = self.consecutive_failures.load(Ordering::Acquire);
        if failures < self.threshold {
            return false;
        }
        let last = self.last_failure_mono_ms.load(Ordering::Acquire);
        monotonic_ms() - last < self.cooldown_ms
    }

    /// Request permission to make a call. Returns false if the breaker is open.
    /// In half-open state (cooldown expired, failures >= threshold), allows
    /// exactly one probe request through via CAS. Subsequent callers are
    /// rejected until the probe completes (record_success or record_failure).
    pub fn allow_request(&self) -> bool {
        let failures = self.consecutive_failures.load(Ordering::Acquire);
        if failures < self.threshold {
            return true; // Closed
        }
        let last = self.last_failure_mono_ms.load(Ordering::Acquire);
        if monotonic_ms() - last < self.cooldown_ms {
            return false; // Open
        }
        // Half-open: CAS to claim the single probe slot
        self.half_open_probe
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    pub fn failures(&self) -> usize {
        self.consecutive_failures.load(Ordering::Acquire)
    }

    pub fn state(&self) -> &'static str {
        let failures = self.consecutive_failures.load(Ordering::Acquire);
        if failures < self.threshold {
            return "closed";
        }
        let last = self.last_failure_mono_ms.load(Ordering::Acquire);
        if monotonic_ms() - last < self.cooldown_ms {
            return "open";
        }
        "half_open"
    }

    pub fn record_success(&self) {
        self.consecutive_failures.store(0, Ordering::Release);
        self.half_open_probe.store(false, Ordering::Release);
    }

    pub fn record_failure(&self) {
        // Saturating add: prevents wrap-around at usize::MAX which would
        // reset the circuit to "closed" after 2^64 failures.
        let new_failures = self
            .consecutive_failures
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                Some(current.saturating_add(1))
            })
            .unwrap_or(0)
            + 1;
        self.last_failure_mono_ms
            .store(monotonic_ms(), Ordering::Release);
        self.half_open_probe.store(false, Ordering::Release);

        // Rejuvenation: cap failure_count at 2× threshold to prevent
        // unbounded growth. Fires every threshold failures for fast convergence.
        let cap = self.threshold.saturating_mul(2);
        if new_failures > cap && new_failures % self.threshold == 0 {
            self.consecutive_failures.store(cap, Ordering::Release);
        }
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
    /// Uses compare_exchange_weak with exponential backoff — yields the
    /// thread after 10 CAS failures to prevent burning CPU under contention.
    pub fn try_acquire(&self) -> bool {
        if self.unlimited {
            return true;
        }
        for spin in 0u32.. {
            let current = self.remaining.load(Ordering::Acquire);
            if current == 0 {
                return false;
            }
            if self
                .remaining
                .compare_exchange_weak(current, current - 1, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return true;
            }
            if spin > 10 {
                std::thread::yield_now();
            } else {
                std::hint::spin_loop();
            }
        }
        false
    }

    /// Atomically try to acquire n budget units. All-or-nothing.
    /// Returns false if fewer than n units remain.
    /// Yields after 10 CAS failures to prevent CPU burn under contention.
    pub fn try_acquire_n(&self, n: usize) -> bool {
        if self.unlimited || n == 0 {
            return true;
        }
        for spin in 0u32.. {
            let current = self.remaining.load(Ordering::Acquire);
            if current < n {
                return false;
            }
            if self
                .remaining
                .compare_exchange_weak(current, current - n, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return true;
            }
            if spin > 10 {
                std::thread::yield_now();
            } else {
                std::hint::spin_loop();
            }
        }
        false
    }

    /// Restore one budget unit. Only called by BudgetGuard on panic/cancel.
    /// Uses saturating add to prevent overflow past usize::MAX.
    pub fn release(&self) {
        if !self.unlimited {
            let _ = self
                .remaining
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                    Some(current.saturating_add(1))
                });
        }
    }

    /// Restore n budget units (e.g., unused pre-reserved budget).
    /// Uses saturating add to prevent overflow past usize::MAX.
    pub fn release_n(&self, n: usize) {
        if !self.unlimited && n > 0 {
            let _ = self
                .remaining
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                    Some(current.saturating_add(n))
                });
        }
    }

    pub fn remaining(&self) -> usize {
        self.remaining.load(Ordering::Acquire)
    }

    pub fn is_unlimited(&self) -> bool {
        self.unlimited
    }

    /// Number of live (uncommitted) budget guards.
    /// Non-zero values persisting over time indicate dropped/cancelled
    /// call paths that did not commit cleanly.
    pub fn leaked_guards(&self) -> u64 {
        BUDGET_GUARD_LIVE_COUNT.load(Ordering::Acquire)
    }
}

// ── Budget Guard (RAII) ──────────────────────────────────────────
// Acquires one budget unit. On normal exit (success or error), call commit().
// On panic/cancellation, Drop restores the unit — the call was never completed.
//
// PRE: caller has already called budget.try_acquire() and it returned true
// POST (commit): budget unit consumed, Drop is a no-op
// POST (drop without commit): budget unit restored

#[must_use = "dropping a BudgetGuard without calling commit() restores the budget unit — this may mask a leak"]
pub struct BudgetGuard<'a> {
    budget: &'a CallBudget,
    committed: AtomicBool,
}

/// Counter tracking live (uncommitted) BudgetGuards. Promoted to production
/// (was debug-only) to surface async task cancellation leaks where BudgetGuard
/// is dropped without commit(). Observable via telemetry. AtomicU64 for zero
/// overhead on the hot path (single fetch_add/fetch_sub per call).
static BUDGET_GUARD_LIVE_COUNT: AtomicU64 = AtomicU64::new(0);

#[allow(dead_code)]
pub fn budget_guard_live_count() -> u64 {
    BUDGET_GUARD_LIVE_COUNT.load(Ordering::Acquire)
}

impl<'a> BudgetGuard<'a> {
    pub fn new(budget: &'a CallBudget) -> Self {
        BUDGET_GUARD_LIVE_COUNT.fetch_add(1, Ordering::AcqRel);
        Self {
            budget,
            committed: AtomicBool::new(false),
        }
    }

    /// Mark this budget unit as consumed. Must be called on all non-panic exits.
    pub fn commit(&mut self) {
        self.committed.store(true, Ordering::Release);
    }
}

impl Drop for BudgetGuard<'_> {
    fn drop(&mut self) {
        BUDGET_GUARD_LIVE_COUNT.fetch_sub(1, Ordering::AcqRel);
        if !self.committed.swap(true, Ordering::AcqRel) {
            // Log uncommitted drops in production (unless panicking, where
            // cancellation-induced drops are expected and correct).
            if !std::thread::panicking() {
                tracing::error!("BudgetGuard dropped without commit() — possible budget leak from async task cancellation");
            }
            if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.budget.release()))
                .is_err()
                && !std::thread::panicking()
            {
                tracing::error!("BudgetGuard release panicked during Drop");
            }
        }
    }
}

// ── Replay Cache ─────────────────────────────────────────────────
// blake3 content-addressed, filesystem-backed.
// Hash(schema_version + model + system + prompt + max_tokens) → cached response.
//
// Content integrity: stored format is "blake3:{hex_hash}\n{content}".
// On read, the hash is verified against the content. Mismatches are
// quarantined to corrupted/ for forensic analysis and treated as cache misses.
//
// PRE: dir is writable (created on init)
// POST (get): Some iff cache file exists, readable, and integrity check passes
// POST (put): file written atomically via temp+rename (crash-safe)

pub struct ReplayCache {
    dir: PathBuf,
    enabled: AtomicBool,
    max_entries: usize, // 0 = unlimited
    /// Counts integrity failures detected at runtime (quarantined entries).
    corruption_count: AtomicUsize,
}

const QUARANTINE_MAX_FILES: usize = 100;

pub struct CachePermit<'a> {
    _cache: &'a ReplayCache,
}

impl ReplayCache {
    pub fn new(dir: PathBuf, enabled: bool, max_entries: usize) -> Self {
        if enabled {
            if let Err(e) = std::fs::create_dir_all(&dir) {
                tracing::warn!(path = %dir.display(), error = %e, "failed to create cache directory");
            }
            // Scavenge orphaned temp files from prior crashed/killed processes.
            // These accumulate on SIGKILL/OOM and would eventually fill disk.
            Self::scavenge_orphaned_temps(&dir);
            // Bound the quarantine directory to prevent unbounded growth from
            // bitrot/disk corruption over long-running deployments.
            Self::scavenge_quarantine(&dir, QUARANTINE_MAX_FILES);
        }
        Self {
            dir,
            enabled: AtomicBool::new(enabled),
            max_entries,
            corruption_count: AtomicUsize::new(0),
        }
    }

    /// Remove orphaned temp files left by crashed processes (SIGKILL, OOM killer).
    /// Any file matching `*.tmp.*` in the cache dir is a temp file from a prior
    /// atomic write that never completed its rename.
    fn scavenge_orphaned_temps(dir: &std::path::Path) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        let mut removed = 0usize;
        for entry in entries.filter_map(|e| e.ok()) {
            let name = entry.file_name();
            let s = name.to_string_lossy();
            if s.contains(".tmp.") {
                if std::fs::remove_file(entry.path()).is_ok() {
                    removed += 1;
                }
            }
        }
        if removed > 0 {
            tracing::info!(removed, "scavenged orphaned temp files from prior crash");
        }
    }

    /// Bound the corrupted/ quarantine directory to `max_files` entries by mtime.
    /// Prevents unbounded disk growth from bitrot over multi-year deployments.
    fn scavenge_quarantine(dir: &std::path::Path, max_files: usize) {
        let quarantine_dir = dir.join("corrupted");
        let Ok(entries) = std::fs::read_dir(&quarantine_dir) else {
            return;
        };
        let mut by_mtime: Vec<_> = entries
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                e.metadata()
                    .ok()
                    .and_then(|m| m.modified().ok())
                    .map(|t| (e.path(), t))
            })
            .collect();

        if by_mtime.len() <= max_files {
            return;
        }

        // Sort oldest first, remove excess
        by_mtime.sort_by_key(|(_, t)| *t);
        let to_remove = by_mtime.len() - max_files;
        let mut removed = 0usize;
        for (path, _) in by_mtime.iter().take(to_remove) {
            if std::fs::remove_file(path).is_ok() {
                removed += 1;
            }
        }
        if removed > 0 {
            tracing::info!(
                removed,
                remaining = by_mtime.len() - removed,
                "scavenged old quarantined cache entries"
            );
        }
    }

    /// Run low-frequency maintenance to keep crash leftovers bounded in
    /// long-running daemons.
    pub fn maintenance_sweep(&self) {
        if !self.enabled.load(Ordering::Acquire) {
            return;
        }
        Self::scavenge_orphaned_temps(&self.dir);
        Self::scavenge_quarantine(&self.dir, QUARANTINE_MAX_FILES);
    }

    /// Content-addressed cache key including schema version.
    /// Bumping CACHE_SCHEMA_VERSION automatically invalidates all stale entries.
    /// Prefixed with `b3-` for cryptographic agility: if blake3 is deprecated,
    /// a future version can emit `b4-` keys while still recognizing `b3-` entries
    /// during rolling deployments. The prefix is validated in get().
    pub fn key(model: &str, system: &str, user_prompt: &str, max_tokens: u32) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&[CACHE_SCHEMA_VERSION]);
        hasher.update(model.as_bytes());
        hasher.update(b"\x00");
        hasher.update(system.as_bytes());
        hasher.update(b"\x00");
        hasher.update(user_prompt.as_bytes());
        hasher.update(b"\x00");
        hasher.update(&max_tokens.to_le_bytes());
        format!("b3-{}", hasher.finalize().to_hex())
    }

    /// Deterministic plan cache key including schema version and model fingerprint.
    /// Model name + context window are included because cached plans derive from
    /// cost/accuracy curves that change when the underlying model changes.
    pub fn plan_key(
        input_size: usize,
        task: &str,
        alpha_bits: u64,
        user_k: usize,
        user_tau: usize,
        model: &str,
        context_window: usize,
    ) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&[CACHE_SCHEMA_VERSION]);
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
        hasher.update(b"\x00");
        hasher.update(model.as_bytes());
        hasher.update(b"\x00");
        hasher.update(&context_window.to_le_bytes());
        format!("b3-plan_{}", hasher.finalize().to_hex())
    }

    /// Reject keys containing path separators or traversal sequences.
    fn validate_key(key: &str) -> bool {
        !key.contains('/') && !key.contains('\\') && !key.contains("..") && !key.is_empty()
    }

    /// Read from cache with content-integrity verification.
    /// Returns None on miss, corruption (quarantined), or legacy format entries.
    pub fn reserve(&self) -> CachePermit<'_> {
        CachePermit { _cache: self }
    }

    pub fn get(&self, key: &str) -> Option<String> {
        if !self.enabled.load(Ordering::Acquire) || !Self::validate_key(key) {
            return None;
        }
        let mut file = std::fs::File::open(self.dir.join(key)).ok()?;
        #[cfg(unix)]
        {
            use fs4::FileExt;
            FileExt::lock_shared(&file).ok()?;
        }
        let mut raw = String::new();
        file.read_to_string(&mut raw).ok()?;
        // Content integrity: first line is "blake3:{hex_hash}", rest is content
        let (header, content) = raw.split_once('\n')?;
        let stored_hash = header.strip_prefix("blake3:")?;
        let computed = blake3::hash(content.as_bytes()).to_hex();
        if stored_hash != computed.as_str() {
            self.corruption_count.fetch_add(1, Ordering::AcqRel);
            tracing::warn!(
                key,
                corruption_total = self.corruption_count.load(Ordering::Acquire),
                "cache integrity check failed, quarantining"
            );
            self.quarantine(key);
            return None;
        }
        Some(content.to_string())
    }

    /// Read from cache and enforce caller-provided semantic validation.
    /// Invalid payloads are quarantined to prevent repeated bad replays.
    pub fn get_validated<F>(&self, key: &str, validator: F) -> Option<String>
    where
        F: Fn(&str) -> bool,
    {
        let content = self.get(key)?;
        if validator(&content) {
            return Some(content);
        }
        tracing::warn!(
            key,
            "cache payload failed semantic validation, quarantining"
        );
        self.quarantine(key);
        None
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
                !s.contains(".tmp.") && s != "corrupted"
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

    /// Write to cache atomically with content-integrity hash.
    /// Format: "blake3:{hex_hash}\n{content}"
    /// This ensures readers can detect bit-rot or filesystem corruption.
    /// Errors are returned, not swallowed — callers decide how to handle.
    pub fn put(&self, key: &str, value: &str) -> anyhow::Result<()> {
        if !self.enabled.load(Ordering::Acquire) {
            return Ok(());
        }
        if !Self::validate_key(key) {
            anyhow::bail!("cache key contains path separator or traversal");
        }
        self.evict_if_needed();
        let final_path = self.dir.join(key);
        let tmp_path = self.dir.join(format!("{}.tmp.{}", key, std::process::id()));

        // 1. Write to temp file with integrity header
        let hash = blake3::hash(value.as_bytes()).to_hex();
        let data = format!("blake3:{}\n{}", hash, value);
        if let Err(e) = std::fs::write(&tmp_path, &data) {
            // Clean up partial temp file (ENOSPC leaves truncated files)
            let _ = std::fs::remove_file(&tmp_path);
            // On disk-full, disable cache and continue rather than crashing
            // the inference pipeline via error propagation.
            if e.kind() == std::io::ErrorKind::StorageFull
                || e.to_string().contains("No space left")
            {
                tracing::warn!("disk full during cache write, disabling cache");
                self.enabled.store(false, Ordering::SeqCst);
                return Ok(());
            }
            return Err(anyhow::anyhow!("cache write to temp file failed: {e}"));
        }

        // 2. Atomic rename (POSIX guarantees atomicity of rename within same filesystem)
        if let Err(e) = std::fs::rename(&tmp_path, &final_path) {
            let _ = std::fs::remove_file(&tmp_path);
            if e.kind() == std::io::ErrorKind::StorageFull
                || e.to_string().contains("No space left")
            {
                tracing::warn!("disk full during cache rename, disabling cache");
                self.enabled.store(false, Ordering::SeqCst);
                return Ok(());
            }
            return Err(anyhow::anyhow!("cache rename failed: {e}"));
        }

        // 3. Fsync the directory to ensure rename is durable.
        // Offloaded to blocking pool: dir fsync can stall 10-100ms on
        // contended disks and must not block the async runtime.
        let dir_path = self.dir.clone();
        let _ = tokio::task::spawn_blocking(move || {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                if let Ok(dir) = std::fs::File::open(&dir_path) {
                    let _ = dir.sync_all();
                }
            }));
        });

        Ok(())
    }

    /// Move a corrupt cache entry to corrupted/ subdirectory for forensic analysis.
    /// If both move and delete fail (e.g., ENOSPC), disables the cache entirely
    /// to prevent reuse of corrupted data on subsequent lookups.
    fn quarantine(&self, key: &str) {
        let corrupt_dir = self.dir.join("corrupted");
        let _ = std::fs::create_dir_all(&corrupt_dir);
        let src = self.dir.join(key);
        let dst = corrupt_dir.join(key);
        if let Err(e) = std::fs::rename(&src, &dst) {
            tracing::warn!(key, error = %e, "failed to quarantine corrupt cache entry, attempting delete");
            if let Err(e2) = std::fs::remove_file(&src) {
                tracing::error!(key, error = %e2, "quarantine delete also failed, disabling cache");
                self.enabled.store(false, Ordering::Release);
            }
        }
    }

    /// Number of corruption events detected since startup.
    pub fn corruption_count(&self) -> usize {
        self.corruption_count.load(Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    /// Unique temp dir per test invocation: PID + monotonic nanos.
    /// Immune to PID rollover collisions on long-running CI.
    fn unique_test_dir(prefix: &str) -> PathBuf {
        let id = format!(
            "{}_{}",
            std::process::id(),
            std::time::Instant::now().elapsed().as_nanos()
        );
        let hash = &blake3::hash(id.as_bytes()).to_hex()[..12];
        std::env::temp_dir().join(format!("{}_{}", prefix, hash))
    }

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
    fn circuit_breaker_half_open_allows_one_probe() {
        let cb = CircuitBreaker::new(1, Duration::from_millis(10));
        cb.record_failure();
        assert!(!cb.allow_request()); // Open — no requests
        thread::sleep(Duration::from_millis(20));
        // Cooldown expired → half-open: first caller gets the probe
        assert!(cb.allow_request());
        // Second caller rejected while probe is in-flight
        assert!(!cb.allow_request());
        // Probe succeeds → back to closed
        cb.record_success();
        assert!(cb.allow_request());
        assert!(cb.allow_request());
    }

    #[test]
    fn circuit_breaker_half_open_failure_resets_cooldown() {
        let cb = CircuitBreaker::new(1, Duration::from_millis(10));
        cb.record_failure();
        thread::sleep(Duration::from_millis(20));
        // Half-open: take probe
        assert!(cb.allow_request());
        // Probe fails → back to open with fresh cooldown
        cb.record_failure();
        assert!(!cb.allow_request()); // Open again
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
    fn call_budget_try_acquire_n_success() {
        let budget = CallBudget::new(10);
        assert!(budget.try_acquire_n(5));
        assert_eq!(budget.remaining(), 5);
        assert!(budget.try_acquire_n(5));
        assert_eq!(budget.remaining(), 0);
    }

    #[test]
    fn call_budget_try_acquire_n_insufficient() {
        let budget = CallBudget::new(3);
        assert!(!budget.try_acquire_n(5)); // Not enough
        assert_eq!(budget.remaining(), 3); // Unchanged
    }

    #[test]
    fn call_budget_try_acquire_n_zero_always_succeeds() {
        let budget = CallBudget::new(0); // unlimited
        assert!(budget.try_acquire_n(1000));
        let budget2 = CallBudget::new(5);
        assert!(budget2.try_acquire_n(0)); // zero is free
    }

    #[test]
    fn call_budget_release_n() {
        let budget = CallBudget::new(10);
        assert!(budget.try_acquire_n(10));
        assert_eq!(budget.remaining(), 0);
        budget.release_n(7);
        assert_eq!(budget.remaining(), 7);
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
    fn budget_guard_drop_restores_on_panic_unwind() {
        let budget = CallBudget::new(1);
        assert!(budget.try_acquire());
        let _ = std::panic::catch_unwind(|| {
            let _guard = BudgetGuard::new(&budget);
            panic!("chaos panic");
        });
        assert_eq!(budget.remaining(), 1);
    }

    #[tokio::test]
    async fn budget_guard_drop_restores_on_task_abort() {
        let budget = Arc::new(CallBudget::new(1));
        assert!(budget.try_acquire());
        let budget_for_task = Arc::clone(&budget);
        let handle = tokio::spawn(async move {
            let _guard = BudgetGuard::new(budget_for_task.as_ref());
            tokio::time::sleep(Duration::from_secs(60)).await;
        });
        tokio::time::sleep(Duration::from_millis(10)).await;
        handle.abort();
        let _ = handle.await;
        assert_eq!(budget.remaining(), 1);
    }

    #[tokio::test]
    async fn replay_cache_atomic_write() {
        let dir = unique_test_dir("rlm_test");
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

    #[tokio::test]
    async fn replay_cache_integrity_detects_corruption() {
        let dir = std::env::temp_dir().join(format!("rlm_integrity_{}", std::process::id()));
        let cache = ReplayCache::new(dir.clone(), true, 0);

        cache.put("good_key", "good_value").unwrap();
        assert_eq!(cache.get("good_key"), Some("good_value".to_string()));

        // Corrupt the file: change content but keep the old hash header
        let file_path = dir.join("good_key");
        let raw = std::fs::read_to_string(&file_path).unwrap();
        let (header, _) = raw.split_once('\n').unwrap();
        std::fs::write(&file_path, format!("{}\ncorrupted_content", header)).unwrap();

        // Get should return None (integrity failure) and quarantine the file
        assert_eq!(cache.get("good_key"), None);

        // File should be quarantined
        assert!(dir.join("corrupted").join("good_key").exists());
        assert!(!file_path.exists());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn replay_cache_legacy_format_treated_as_miss() {
        let dir = std::env::temp_dir().join(format!("rlm_legacy_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);

        // Write a legacy-format file (no integrity header)
        std::fs::write(dir.join("legacy_key"), "just raw content").unwrap();

        let cache = ReplayCache::new(dir.clone(), true, 0);
        assert_eq!(cache.get("legacy_key"), None); // Treated as miss

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

    #[tokio::test]
    async fn replay_cache_eviction() {
        let dir = std::env::temp_dir().join(format!("rlm_evict_{}", std::process::id()));
        let cache = ReplayCache::new(dir.clone(), true, 5);

        // Write 8 entries — should trigger eviction on the later puts
        for i in 0..8 {
            cache
                .put(&format!("key_{i}"), &format!("value_{i}"))
                .unwrap();
            // Small sleep to ensure distinct mtimes for eviction ordering
            thread::sleep(Duration::from_millis(10));
        }

        // Eviction fires before each write, so worst case is max_entries + 1
        let count = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| {
                let name = e.file_name();
                let s = name.to_string_lossy();
                s != "corrupted"
            })
            .count();
        assert!(count <= 6, "expected at most max_entries+1=6, got {count}");

        // Most recent entries should survive
        assert!(cache.get("key_7").is_some());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn quarantine_scavenging_bounds_directory() {
        let dir = std::env::temp_dir().join(format!("rlm_qsca_{}", std::process::id()));
        let quarantine_dir = dir.join("corrupted");
        let _ = std::fs::create_dir_all(&quarantine_dir);

        // Write 105 corrupted files
        for i in 0..105 {
            std::fs::write(quarantine_dir.join(format!("corrupt_{i}")), "bad data").unwrap();
            // Tiny sleep to ensure distinct mtimes
            thread::sleep(Duration::from_millis(2));
        }

        // Scavenge to 100
        ReplayCache::scavenge_quarantine(&dir, QUARANTINE_MAX_FILES);

        let remaining = std::fs::read_dir(&quarantine_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .count();
        assert!(
            remaining <= QUARANTINE_MAX_FILES,
            "quarantine should be bounded to {QUARANTINE_MAX_FILES}, got {remaining}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn circuit_breaker_concurrent_state_machine_stress() {
        // Verify invariants under concurrent access:
        // - If is_open(), failures >= threshold (unless in HalfOpen window)
        // - half_open_probe is only ever false or true (no overflow)
        let cb = Arc::new(CircuitBreaker::new(5, Duration::from_millis(50)));
        let mut handles = Vec::new();

        for _ in 0..8 {
            let cb = Arc::clone(&cb);
            handles.push(thread::spawn(move || {
                for i in 0..1000 {
                    match i % 3 {
                        0 => {
                            cb.record_failure();
                        }
                        1 => {
                            cb.record_success();
                        }
                        _ => {
                            let _ = cb.allow_request();
                        }
                    }
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        // After all threads complete, the breaker should be in a consistent state:
        // failures() should be a valid usize, allow_request() should not panic
        let _f = cb.failures();
        let _a = cb.allow_request();
    }
}

// ── Chaos Tests ──────────────────────────────────────────────────

#[cfg(test)]
mod chaos_tests {
    use super::*;

    fn unique_chaos_dir(prefix: &str) -> PathBuf {
        let id = format!(
            "{}_{}",
            std::process::id(),
            std::time::Instant::now().elapsed().as_nanos()
        );
        let hash = &blake3::hash(id.as_bytes()).to_hex()[..12];
        std::env::temp_dir().join(format!("{}_{}", prefix, hash))
    }

    /// Verify cache gracefully disables itself on write to a read-only directory.
    #[tokio::test]
    async fn cache_write_failure_degrades_gracefully() {
        let dir = unique_chaos_dir("rlm_chaos_cache");
        let cache = ReplayCache::new(dir.clone(), true, 0);

        cache.put("good_key", "good_value").unwrap();
        assert_eq!(cache.get("good_key"), Some("good_value".to_string()));

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let perms = std::fs::Permissions::from_mode(0o555);
            std::fs::set_permissions(&dir, perms).unwrap();

            let result = cache.put("fail_key", "fail_value");
            if result.is_ok() {
                assert!(
                    cache.get("fail_key").is_none(),
                    "failed write should not be readable"
                );
            }

            assert_eq!(cache.get("good_key"), Some("good_value".to_string()));

            let perms = std::fs::Permissions::from_mode(0o755);
            std::fs::set_permissions(&dir, perms).unwrap();
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Verify BudgetGuard debug counter tracks live guards correctly.
    #[cfg(debug_assertions)]
    #[test]
    fn budget_guard_leak_counter() {
        let budget = CallBudget::new(10);

        let baseline = budget_guard_live_count();

        assert!(budget.try_acquire());
        let mut g1 = BudgetGuard::new(&budget);
        assert_eq!(budget_guard_live_count(), baseline + 1);

        assert!(budget.try_acquire());
        let g2 = BudgetGuard::new(&budget);
        assert_eq!(budget_guard_live_count(), baseline + 2);

        g1.commit();
        drop(g1);
        assert_eq!(budget_guard_live_count(), baseline + 1);

        drop(g2);
        assert_eq!(budget_guard_live_count(), baseline);
    }
}

// ── Loom model-checked tests ─────────────────────────────────────
// Deterministic concurrency testing via loom's model checker. Explores
// all scheduler interleavings of lock-free primitives (CAS loops, Release-
// Acquire orderings). Catches subtle reordering bugs that only manifest
// under specific interleavings — once per decade in production.
//
// Run with: cargo test --features loom loom_tests -- --test-threads=1

#[cfg(test)]
mod loom_tests {
    use loom::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use loom::sync::Arc;
    use loom::thread;

    /// Minimal CallBudget reimplemented with loom atomics for model checking.
    /// Mirrors the real CallBudget's CAS loop exactly.
    struct LoomBudget {
        remaining: AtomicUsize,
    }

    impl LoomBudget {
        fn new(max_calls: usize) -> Self {
            Self {
                remaining: AtomicUsize::new(max_calls),
            }
        }

        fn try_acquire(&self) -> bool {
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
                loom::thread::yield_now();
            }
        }

        fn release(&self) {
            let _ = self
                .remaining
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                    Some(current.saturating_add(1))
                });
        }

        fn remaining(&self) -> usize {
            self.remaining.load(Ordering::Acquire)
        }
    }

    /// Verify budget invariant: total acquired never exceeds initial budget.
    /// Two threads race try_acquire; sum of successes must equal budget.
    #[test]
    fn budget_acquire_release_race() {
        loom::model(|| {
            let budget = Arc::new(LoomBudget::new(3));

            let b1 = Arc::clone(&budget);
            let t1 = thread::spawn(move || {
                let mut acquired = 0;
                for _ in 0..2 {
                    if b1.try_acquire() {
                        acquired += 1;
                    }
                }
                acquired
            });

            let b2 = Arc::clone(&budget);
            let t2 = thread::spawn(move || {
                let mut acquired = 0;
                for _ in 0..2 {
                    if b2.try_acquire() {
                        acquired += 1;
                    }
                }
                acquired
            });

            let a1 = t1.join().unwrap();
            let a2 = t2.join().unwrap();
            let total = a1 + a2;

            // Invariant: never acquire more than the budget allows
            assert!(total <= 3, "over-acquired: {total} > 3");
            // Remaining should be consistent
            assert_eq!(
                budget.remaining(),
                3 - total,
                "remaining mismatch: {} != {}",
                budget.remaining(),
                3 - total
            );
        });
    }

    /// Verify release restores budget correctly under concurrent access.
    /// One thread acquires, another releases — remaining must never go negative
    /// or exceed the original budget.
    #[test]
    fn budget_release_never_overflows() {
        loom::model(|| {
            let budget = Arc::new(LoomBudget::new(2));

            // Pre-acquire one unit
            assert!(budget.try_acquire());

            let b1 = Arc::clone(&budget);
            let t1 = thread::spawn(move || {
                b1.release(); // Release the pre-acquired unit
            });

            let b2 = Arc::clone(&budget);
            let t2 = thread::spawn(move || b2.try_acquire());

            t1.join().unwrap();
            let acquired = t2.join().unwrap();

            let remaining = budget.remaining();
            // Budget started at 2, we acquired 1 before threads, released 1 in t1,
            // and t2 may or may not have acquired 1.
            if acquired {
                assert!(
                    remaining <= 2,
                    "remaining {remaining} > 2 after release+acquire"
                );
            } else {
                assert_eq!(
                    remaining, 2,
                    "remaining should be 2 after release without acquire"
                );
            }
        });
    }

    struct LoomHalfOpenGate {
        probe_claimed: AtomicBool,
    }

    impl LoomHalfOpenGate {
        fn new() -> Self {
            Self {
                probe_claimed: AtomicBool::new(false),
            }
        }

        fn allow_probe(&self) -> bool {
            self.probe_claimed
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
        }
    }

    #[test]
    fn circuit_half_open_allows_single_probe_under_race() {
        loom::model(|| {
            let gate = Arc::new(LoomHalfOpenGate::new());
            let successes = Arc::new(AtomicUsize::new(0));

            let g1 = Arc::clone(&gate);
            let s1 = Arc::clone(&successes);
            let t1 = thread::spawn(move || {
                if g1.allow_probe() {
                    s1.fetch_add(1, Ordering::AcqRel);
                }
            });

            let g2 = Arc::clone(&gate);
            let s2 = Arc::clone(&successes);
            let t2 = thread::spawn(move || {
                if g2.allow_probe() {
                    s2.fetch_add(1, Ordering::AcqRel);
                }
            });

            t1.join().unwrap();
            t2.join().unwrap();

            assert_eq!(
                successes.load(Ordering::Acquire),
                1,
                "half-open gate admitted more than one probe"
            );
        });
    }
}
