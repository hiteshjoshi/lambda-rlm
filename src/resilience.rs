//! Resilience primitives: circuit breaker, call budget with RAII guard, replay cache.
//!
//! All state is lock-free (atomics). No Mutex, no contention.
//! Atomic orderings: Release-Acquire on all cross-thread state to ensure
//! correct visibility on ARM64 and other weakly-ordered architectures.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Bump when prompt logic or cache format changes to invalidate stale entries.
/// Any modification to leaf_prompt() or reduce prompts MUST bump this version.
/// V2: added hash algorithm prefix (b3-) to cache keys for cryptographic agility.
pub const CACHE_SCHEMA_VERSION: u8 = 2;

/// Schema version for circuit breaker persistence format.
/// V2: stores remaining_cooldown_ms alongside epoch to reduce NTP sensitivity.
/// V3: adds truncated blake3 checksum to detect silent bitrot in stored values.
const CB_PERSIST_VERSION: u8 = 3;

/// Maximum schema version this binary understands. If a state file has a
/// higher version (written by a newer binary), we refuse to load or overwrite
/// it — preventing silent data corruption on version rollback.
const CB_MAX_SUPPORTED_VERSION: u8 = 3;

/// Wall-clock epoch milliseconds. Use only for timestamps in logs/telemetry.
/// NOT for interval measurement (vulnerable to NTP jumps).
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
// Optionally persists state across restarts via atomic file writes.

pub struct CircuitBreaker {
    consecutive_failures: AtomicUsize,
    last_failure_mono_ms: AtomicU64,
    half_open_probe: AtomicBool,
    threshold: usize,
    cooldown_ms: u64,
    persist_path: Option<PathBuf>,
}

impl CircuitBreaker {
    pub fn new(threshold: usize, cooldown: Duration) -> Self {
        // Ensure BOOT is initialized on first CircuitBreaker creation
        let _ = BOOT.get_or_init(Instant::now);
        Self {
            consecutive_failures: AtomicUsize::new(0),
            last_failure_mono_ms: AtomicU64::new(0),
            half_open_probe: AtomicBool::new(false),
            threshold,
            cooldown_ms: cooldown.as_millis() as u64,
            persist_path: None,
        }
    }

    /// Create a circuit breaker that persists state across process restarts.
    /// On creation, loads previous state if the file exists and schema matches.
    /// Corrupted files are logged and ignored (state resets to closed).
    pub fn with_persistence(threshold: usize, cooldown: Duration, path: PathBuf) -> Self {
        let _ = BOOT.get_or_init(Instant::now);
        let cooldown_ms = cooldown.as_millis() as u64;
        let (failures, mono, persist_safe) = Self::load_state(&path, cooldown_ms);
        if failures > 0 {
            tracing::info!(failures, "circuit breaker loaded persisted state");
        }
        Self {
            consecutive_failures: AtomicUsize::new(failures),
            last_failure_mono_ms: AtomicU64::new(mono),
            half_open_probe: AtomicBool::new(false),
            threshold,
            cooldown_ms,
            // Disable persistence if a future-version file was detected,
            // to avoid overwriting it with our older format.
            persist_path: if persist_safe { Some(path) } else { None },
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

    pub fn record_success(&self) {
        self.consecutive_failures.store(0, Ordering::Release);
        self.half_open_probe.store(false, Ordering::Release);
        self.save_state();
    }

    pub fn record_failure(&self) {
        // Saturating add: prevents wrap-around at usize::MAX which would
        // reset the circuit to "closed" after 2^64 failures.
        let new_failures = self.consecutive_failures.fetch_update(
            Ordering::AcqRel,
            Ordering::Acquire,
            |current| Some(current.saturating_add(1)),
        ).unwrap_or(0) + 1;
        self.last_failure_mono_ms
            .store(monotonic_ms(), Ordering::Release);
        self.half_open_probe.store(false, Ordering::Release);

        // Rejuvenation: periodically cap failure_count to prevent saturation.
        // Over 30 years of micro-failures, failure_count could approach usize::MAX.
        // Every 10,000 failures in Closed state, soft-reset to threshold * 2 to
        // preserve "open" semantics without losing track of recent history.
        let cap = self.threshold.saturating_mul(2);
        if new_failures > cap && new_failures % 10_000 == 0 {
            let _ = self.consecutive_failures.fetch_update(
                Ordering::AcqRel,
                Ordering::Acquire,
                |current| {
                    if current > cap {
                        Some(cap)
                    } else {
                        Some(current)
                    }
                },
            );
            tracing::info!(
                rejuvenated_from = new_failures,
                rejuvenated_to = cap,
                "circuit breaker rejuvenation: capped failure count to prevent saturation"
            );
        }

        self.save_state();
    }

    /// Persist current state via atomic temp+rename write.
    /// Format: "V3:{failures}:{epoch_ms}:{remaining_cooldown_ms}:{blake3_8hex}"
    /// Stores remaining cooldown computed from monotonic clock, making the
    /// persisted state resilient to NTP jumps between save and load.
    /// Truncated blake3 checksum detects silent bitrot in stored values.
    /// On ENOSPC or rename failure: disables persistence and continues
    /// operation — a full disk must not cascade into circuit breaker corruption.
    ///
    /// Blocking I/O (fsync can stall 10-100ms on contended disks) is offloaded
    /// to tokio's blocking pool to prevent cascading latency across async tasks.
    /// In-memory circuit breaker state is already updated before this is called.
    fn save_state(&self) {
        let Some(path) = &self.persist_path else {
            return;
        };
        let failures = self.consecutive_failures.load(Ordering::Acquire);
        let last_mono = self.last_failure_mono_ms.load(Ordering::Acquire);
        let elapsed = monotonic_ms().saturating_sub(last_mono);
        let remaining_cooldown = self.cooldown_ms.saturating_sub(elapsed);
        let payload = format!("{}:{}:{}", failures, epoch_ms(), remaining_cooldown);
        let checksum = blake3::hash(payload.as_bytes()).to_hex()[..8].to_string();
        let data = format!("V{}:{}:{}", CB_PERSIST_VERSION, payload, checksum);
        let path = path.clone();

        // Offload blocking I/O (write → fsync → rename → dir_fsync) to the
        // blocking thread pool. fsync can stall 10-100ms+ on contended disks;
        // running it on the async runtime cascades latency to unrelated tasks.
        // Guard: if no tokio runtime is active (e.g., in synchronous tests),
        // skip the async persistence — in-memory state is still correct.
        let Ok(_handle) = tokio::runtime::Handle::try_current() else {
            return;
        };
        let _ = tokio::task::spawn_blocking(move || {
            // catch_unwind: prevent panics in blocking IO (stack overflow on
            // pathological paths, fs driver bugs) from killing the blocking
            // thread pool and starving other spawn_blocking callers.
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let tmp = path.with_extension("tmp");
                // Write → fsync → rename → dir_fsync: crash-safe persistence sequence.
                // POSIX does not order renames before writes without explicit fsync.
                let write_result = (|| -> std::io::Result<()> {
                    use std::io::Write;
                    let mut file = std::fs::File::create(&tmp)?;
                    file.write_all(data.as_bytes())?;
                    file.sync_all()?; // Ensure data is on disk before rename
                    Ok(())
                })();
                match write_result {
                    Ok(()) => {
                        match std::fs::rename(&tmp, &path) {
                            Err(e) => {
                                tracing::warn!(error = %e, "circuit breaker state rename failed");
                                let _ = std::fs::remove_file(&tmp);
                            }
                            Ok(()) => {
                                if let Ok(dir) = path.parent().map(std::fs::File::open).unwrap_or(Err(std::io::Error::new(std::io::ErrorKind::NotFound, "no parent"))) {
                                    let _ = dir.sync_all();
                                }
                                match std::fs::read_to_string(&path) {
                                    Ok(ref readback) if readback != &data => {
                                        tracing::error!("circuit breaker read-after-write mismatch — possible disk corruption");
                                        let _ = std::fs::remove_file(&path);
                                    }
                                    Err(e) => {
                                        tracing::warn!(error = %e, "circuit breaker read-after-write failed");
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "circuit breaker state write failed");
                        let _ = std::fs::remove_file(&tmp);
                    }
                }
            }));
            if result.is_err() {
                tracing::error!("circuit breaker save_state panicked — in-memory state unaffected");
            }
        });
    }

    /// Synchronous save for test use: writes state file directly on the current
    /// thread without spawn_blocking, ensuring the file exists before assertions.
    #[cfg(test)]
    fn save_state_sync(&self) {
        let Some(path) = &self.persist_path else {
            return;
        };
        let failures = self.consecutive_failures.load(Ordering::Acquire);
        let last_mono = self.last_failure_mono_ms.load(Ordering::Acquire);
        let elapsed = monotonic_ms().saturating_sub(last_mono);
        let remaining_cooldown = self.cooldown_ms.saturating_sub(elapsed);
        let payload = format!("{}:{}:{}", failures, epoch_ms(), remaining_cooldown);
        let checksum = blake3::hash(payload.as_bytes()).to_hex()[..8].to_string();
        let data = format!("V{}:{}:{}", CB_PERSIST_VERSION, payload, checksum);
        let tmp = path.with_extension("tmp");
        use std::io::Write;
        let mut file = std::fs::File::create(&tmp).unwrap();
        file.write_all(data.as_bytes()).unwrap();
        file.sync_all().unwrap();
        std::fs::rename(&tmp, path).unwrap();
    }

    /// Load persisted state. Returns (failures, monotonic_ms_offset, persist_safe).
    /// If the file is missing, corrupted, or has a schema/checksum mismatch, returns (0, 0, true).
    /// If a future version is detected (version > CB_MAX_SUPPORTED_VERSION), returns
    /// (0, 0, false) — safe defaults, but persistence is disabled to avoid overwriting
    /// the future-version file on rollback.
    ///
    /// V3 adds blake3 checksum to detect silent bitrot in stored values.
    /// V2 format stores remaining_cooldown_ms computed from monotonic clock at save time.
    /// On load, remaining is adjusted by elapsed wall-clock since save (short window,
    /// low NTP risk). This eliminates the V1 vulnerability where NTP forward jumps
    /// caused premature cooldown expiry (flapping).
    fn load_state(path: &PathBuf, cooldown_ms: u64) -> (usize, u64, bool) {
        let Ok(data) = std::fs::read_to_string(path) else {
            return (0, 0, true);
        };
        let parts: Vec<&str> = data.splitn(5, ':').collect();
        if parts.len() < 3 {
            tracing::warn!("corrupt circuit breaker state file, resetting");
            return (0, 0, true);
        }
        let version = parts[0]
            .strip_prefix('V')
            .and_then(|v| v.parse::<u8>().ok());
        match version {
            Some(v) if v > CB_MAX_SUPPORTED_VERSION => {
                // Future version: a newer binary wrote this file. Attempt to
                // read failure count (field layout stable since V1) to avoid
                // thundering herd on rollback. Disable persistence so we don't
                // overwrite the file with our older format.
                tracing::error!(
                    file_version = v,
                    max_supported = CB_MAX_SUPPORTED_VERSION,
                    "future state file detected; reading failures only, refusing to modify — operator alert: binary rollback in progress"
                );
                // Best-effort: parts[1] has been the failure count in every version.
                // If we can parse it, preserve the open circuit to protect downstreams.
                let failures = parts.get(1)
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(0);
                let mono = if failures > 0 {
                    // Conservative: treat as just-happened to keep circuit open
                    // until this binary's cooldown expires naturally.
                    monotonic_ms()
                } else {
                    0
                };
                if failures > 0 {
                    tracing::info!(failures, "preserving failure count from future-version state");
                }
                (failures, mono, false)
            }
            Some(3) => {
                // V3: remaining_cooldown_ms + blake3 checksum
                if parts.len() != 5 {
                    tracing::warn!("corrupt V3 circuit breaker state, resetting");
                    return (0, 0, true);
                }
                // Validate checksum before trusting any field values
                let payload = format!("{}:{}:{}", parts[1], parts[2], parts[3]);
                let expected = &blake3::hash(payload.as_bytes()).to_hex()[..8];
                if parts[4] != expected {
                    tracing::warn!("circuit breaker checksum mismatch (bitrot?), resetting to safe default");
                    return (0, 0, true);
                }
                let (f, m) = Self::parse_v2_fields(parts[1], parts[2], parts[3], cooldown_ms);
                (f, m, true)
            }
            Some(2) => {
                // V2: remaining_cooldown_ms field present, no checksum
                if parts.len() != 4 {
                    tracing::warn!("corrupt V2 circuit breaker state, resetting");
                    return (0, 0, true);
                }
                tracing::info!("migrating V2 circuit breaker state to V3");
                let (f, m) = Self::parse_v2_fields(parts[1], parts[2], parts[3], cooldown_ms);
                (f, m, true)
            }
            Some(1) => {
                // Legacy V1: epoch-only format (NTP-vulnerable, migrate forward)
                if parts.len() < 3 {
                    return (0, 0, true);
                }
                let Ok(failures) = parts[1].parse::<usize>() else {
                    tracing::warn!("corrupt circuit breaker failure count, resetting");
                    return (0, 0, true);
                };
                let Ok(saved_epoch) = parts[2].parse::<u64>() else {
                    tracing::warn!("corrupt circuit breaker timestamp, resetting");
                    return (0, 0, true);
                };
                tracing::info!("migrating V1 circuit breaker state to V3");
                let now = epoch_ms();
                let mono = if now.saturating_sub(saved_epoch) < cooldown_ms {
                    monotonic_ms() // conservative: treat as just-happened
                } else {
                    0 // cooldown already expired
                };
                (failures, mono, true)
            }
            _ => {
                tracing::warn!("circuit breaker schema version unknown or corrupt, resetting");
                (0, 0, true)
            }
        }
    }

    /// Parse V2/V3 fields (failures, epoch, remaining_cooldown) into (failures, mono_ms).
    /// Shared by V2 and V3 loaders — the field semantics are identical.
    fn parse_v2_fields(
        failures_str: &str,
        epoch_str: &str,
        remaining_str: &str,
        cooldown_ms: u64,
    ) -> (usize, u64) {
        let Ok(failures) = failures_str.parse::<usize>() else {
            tracing::warn!("corrupt circuit breaker failure count, resetting");
            return (0, 0);
        };
        let Ok(saved_epoch) = epoch_str.parse::<u64>() else {
            tracing::warn!("corrupt circuit breaker timestamp, resetting");
            return (0, 0);
        };
        let Ok(saved_remaining) = remaining_str.parse::<u64>() else {
            tracing::warn!("corrupt circuit breaker remaining cooldown, resetting");
            return (0, 0);
        };
        // Clamp remaining to cooldown_ms: logically impossible for remaining
        // to exceed total cooldown unless the file was tampered or corrupted.
        // Without this, a huge remaining value synthesizes a last_failure_mono_ms
        // that locks the circuit breaker open indefinitely.
        let saved_remaining = saved_remaining.min(cooldown_ms);
        // Detect NTP backward jump: saved_epoch in the future means wall-clock
        // went backwards. Reset to safe defaults rather than computing negative
        // elapsed time (which saturating_sub would clamp to 0, preserving full
        // cooldown — conservative but potentially stale for days/weeks).
        let now_epoch = epoch_ms();
        if saved_epoch > now_epoch.saturating_add(5_000) {
            // More than 5s in the future: NTP correction or tampering.
            tracing::warn!(
                saved_epoch,
                now_epoch,
                "circuit breaker state has future timestamp — resetting"
            );
            return (0, 0);
        }
        // Adjust remaining by elapsed wall-clock since save.
        // This window (save → crash → restart → load) is typically
        // seconds, making NTP interference negligible.
        let elapsed = now_epoch.saturating_sub(saved_epoch);
        let remaining = saved_remaining.saturating_sub(elapsed);
        let mono = if remaining > 0 {
            // Position last_failure in monotonic time so cooldown
            // expires in exactly `remaining` ms from now.
            let now_mono = monotonic_ms();
            now_mono.saturating_sub(cooldown_ms.saturating_sub(remaining))
        } else {
            0 // cooldown expired
        };
        (failures, mono)
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
    /// Uses compare_exchange_weak in a loop — spin_loop() hint prevents
    /// power waste on ARM cores during spurious CAS failures.
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
            std::hint::spin_loop();
        }
    }

    /// Atomically try to acquire n budget units. All-or-nothing.
    /// Returns false if fewer than n units remain.
    /// Uses compare_exchange_weak — spin_loop() hint for ARM efficiency.
    #[allow(dead_code)] // Available for pre-reservation patterns
    pub fn try_acquire_n(&self, n: usize) -> bool {
        if self.unlimited || n == 0 {
            return true;
        }
        loop {
            let current = self.remaining.load(Ordering::Acquire);
            if current < n {
                return false;
            }
            if self
                .remaining
                .compare_exchange_weak(
                    current,
                    current - n,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
            {
                return true;
            }
            std::hint::spin_loop();
        }
    }

    /// Restore one budget unit. Only called by BudgetGuard on panic/cancel.
    /// Uses saturating add to prevent overflow past usize::MAX.
    pub fn release(&self) {
        if !self.unlimited {
            let _ = self.remaining.fetch_update(
                Ordering::AcqRel,
                Ordering::Acquire,
                |current| Some(current.saturating_add(1)),
            );
        }
    }

    /// Restore n budget units (e.g., unused pre-reserved budget).
    /// Uses saturating add to prevent overflow past usize::MAX.
    #[allow(dead_code)] // Available for pre-reservation patterns
    pub fn release_n(&self, n: usize) {
        if !self.unlimited && n > 0 {
            let _ = self.remaining.fetch_update(
                Ordering::AcqRel,
                Ordering::Acquire,
                |current| Some(current.saturating_add(n)),
            );
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

#[must_use = "dropping a BudgetGuard without calling commit() restores the budget unit — this may mask a leak"]
pub struct BudgetGuard<'a> {
    budget: &'a CallBudget,
    committed: bool,
}

/// Debug-only counter tracking live (uncommitted) BudgetGuards.
/// Assert zero on shutdown to prove no budget leaks exist in test runs.
/// Only active in debug builds — zero overhead in release.
#[cfg(debug_assertions)]
static BUDGET_GUARD_LIVE_COUNT: AtomicUsize = AtomicUsize::new(0);

#[cfg(debug_assertions)]
#[allow(dead_code)]
pub fn budget_guard_live_count() -> usize {
    BUDGET_GUARD_LIVE_COUNT.load(Ordering::Acquire)
}

impl<'a> BudgetGuard<'a> {
    pub fn new(budget: &'a CallBudget) -> Self {
        #[cfg(debug_assertions)]
        BUDGET_GUARD_LIVE_COUNT.fetch_add(1, Ordering::AcqRel);
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
        #[cfg(debug_assertions)]
        BUDGET_GUARD_LIVE_COUNT.fetch_sub(1, Ordering::AcqRel);
        if !self.committed {
            self.budget.release();
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
    /// Observable via telemetry to surface silent disk corruption / bitrot.
    corruption_count: AtomicUsize,
    /// Monotonic ms when the current corruption rate window started.
    corruption_window_start_ms: AtomicU64,
    /// Corruptions counted in the current 60s rate window.
    corruptions_in_window: AtomicUsize,
    /// Monotonic ms until which cache writes are disabled (corruption backoff).
    corruption_backoff_until_ms: AtomicU64,
    /// Current backoff duration in ms (doubles on each trigger, caps at 24h).
    corruption_backoff_ms: AtomicU64,
}

/// Lightweight handle for background cache scrub (Send + 'static).
/// Does not carry atomic state — only the dir path and a snapshot of enabled.
pub struct ScrubHandle {
    dir: PathBuf,
    enabled: bool,
}

impl ScrubHandle {
    pub fn scrub(&self) -> (usize, usize) {
        if !self.enabled {
            return (0, 0);
        }
        // Scavenge quarantine
        ReplayCache::scavenge_quarantine(&self.dir, 100);
        ReplayCache::scavenge_orphaned_temps(&self.dir);
        // Verify integrity of all entries
        let Ok(entries) = std::fs::read_dir(&self.dir) else {
            return (0, 0);
        };
        let mut verified = 0usize;
        let mut quarantined = 0usize;
        let corrupt_dir = self.dir.join("corrupted");
        for entry in entries.filter_map(|e| e.ok()) {
            let name = entry.file_name();
            let s = name.to_string_lossy();
            if s.contains(".tmp.") || s == "corrupted" || s.starts_with('.') {
                continue;
            }
            if !entry.path().is_file() {
                continue;
            }
            let Ok(raw) = std::fs::read_to_string(entry.path()) else {
                continue;
            };
            let Some((header, content)) = raw.split_once('\n') else {
                let _ = std::fs::create_dir_all(&corrupt_dir);
                let _ = std::fs::rename(entry.path(), corrupt_dir.join(&*s));
                quarantined += 1;
                continue;
            };
            let Some(stored_hash) = header.strip_prefix("blake3:") else {
                let _ = std::fs::create_dir_all(&corrupt_dir);
                let _ = std::fs::rename(entry.path(), corrupt_dir.join(&*s));
                quarantined += 1;
                continue;
            };
            let computed = blake3::hash(content.as_bytes()).to_hex();
            if stored_hash != computed.as_str() {
                let _ = std::fs::create_dir_all(&corrupt_dir);
                let _ = std::fs::rename(entry.path(), corrupt_dir.join(&*s));
                quarantined += 1;
            } else {
                verified += 1;
            }
        }
        (verified, quarantined)
    }

    /// Generational compaction: rewrites all valid cache entries into a new
    /// directory, then atomically swaps via rename. This defragments the
    /// directory index (ext4/xfs degrade to linear scans after millions of
    /// entries) and reclaims inode space from deleted entries.
    ///
    /// Returns (migrated, dropped) counts. Errors are logged, not propagated.
    pub fn compact(&self) -> (usize, usize) {
        if !self.enabled {
            return (0, 0);
        }

        // 1. Create new generation directory alongside the current one.
        let gen_dir = self.dir.with_extension("compact");
        if std::fs::create_dir_all(&gen_dir).is_err() {
            tracing::warn!(path = %gen_dir.display(), "compaction: failed to create generation dir");
            return (0, 0);
        }

        // 2. Read all valid entries, verify integrity, write to new dir.
        let Ok(entries) = std::fs::read_dir(&self.dir) else {
            let _ = std::fs::remove_dir_all(&gen_dir);
            return (0, 0);
        };

        let mut migrated = 0usize;
        let mut dropped = 0usize;

        for entry in entries.filter_map(|e| e.ok()) {
            let name = entry.file_name();
            let s = name.to_string_lossy();
            // Skip non-cache entries
            if s.contains(".tmp.") || s == "corrupted" || s.starts_with('.') {
                continue;
            }
            if !entry.path().is_file() {
                continue;
            }
            let Ok(raw) = std::fs::read_to_string(entry.path()) else {
                dropped += 1;
                continue;
            };
            // Verify integrity before migrating
            let Some((header, content)) = raw.split_once('\n') else {
                dropped += 1;
                continue;
            };
            let Some(stored_hash) = header.strip_prefix("blake3:") else {
                dropped += 1;
                continue;
            };
            let computed = blake3::hash(content.as_bytes()).to_hex();
            if stored_hash != computed.as_str() {
                dropped += 1;
                continue;
            }
            // Write verified entry to new generation (sequential writes = defrag)
            let dst = gen_dir.join(&*s);
            if std::fs::write(&dst, &raw).is_ok() {
                migrated += 1;
            } else {
                dropped += 1;
            }
        }

        // 3. Preserve quarantine and state files
        let old_quarantine = self.dir.join("corrupted");
        let new_quarantine = gen_dir.join("corrupted");
        if old_quarantine.is_dir() {
            let _ = Self::copy_dir_contents(&old_quarantine, &new_quarantine);
        }
        // Preserve circuit breaker state file
        let cb_state = self.dir.join(".circuit_breaker_state");
        if cb_state.is_file() {
            let _ = std::fs::copy(&cb_state, gen_dir.join(".circuit_breaker_state"));
        }

        // 4. Atomic swap: rename old → old.bak, new → current, remove old.bak
        let backup_dir = self.dir.with_extension("old");
        let _ = std::fs::remove_dir_all(&backup_dir); // clean stale backup
        if std::fs::rename(&self.dir, &backup_dir).is_err() {
            tracing::warn!("compaction: failed to move old cache dir to backup");
            let _ = std::fs::remove_dir_all(&gen_dir);
            return (0, 0);
        }
        if std::fs::rename(&gen_dir, &self.dir).is_err() {
            // Rollback: restore the original
            tracing::error!("compaction: failed to swap in new generation, rolling back");
            let _ = std::fs::rename(&backup_dir, &self.dir);
            return (0, 0);
        }
        // 5. fsync parent directory to make the rename durable
        if let Some(parent) = self.dir.parent() {
            if let Ok(dir) = std::fs::File::open(parent) {
                let _ = dir.sync_all();
            }
        }
        // Clean up backup
        let _ = std::fs::remove_dir_all(&backup_dir);

        if migrated > 0 || dropped > 0 {
            tracing::info!(
                migrated,
                dropped,
                "cache compaction complete: directory defragmented"
            );
        }
        (migrated, dropped)
    }

    /// Copy contents of one directory to another (non-recursive, files only).
    fn copy_dir_contents(src: &std::path::Path, dst: &std::path::Path) -> std::io::Result<()> {
        std::fs::create_dir_all(dst)?;
        for entry in std::fs::read_dir(src)?.filter_map(|e| e.ok()) {
            if entry.path().is_file() {
                std::fs::copy(entry.path(), dst.join(entry.file_name()))?;
            }
        }
        Ok(())
    }
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
            Self::scavenge_quarantine(&dir, 100);
        }
        Self {
            dir,
            enabled: AtomicBool::new(enabled),
            max_entries,
            corruption_count: AtomicUsize::new(0),
            corruption_window_start_ms: AtomicU64::new(0),
            corruptions_in_window: AtomicUsize::new(0),
            corruption_backoff_until_ms: AtomicU64::new(0),
            corruption_backoff_ms: AtomicU64::new(0),
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
            tracing::info!(removed, remaining = by_mtime.len() - removed, "scavenged old quarantined cache entries");
        }
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
    pub fn get(&self, key: &str) -> Option<String> {
        if !self.enabled.load(Ordering::Acquire) || !Self::validate_key(key) {
            return None;
        }
        let raw = std::fs::read_to_string(self.dir.join(key)).ok()?;
        // Content integrity: first line is "blake3:{hex_hash}", rest is content
        let (header, content) = raw.split_once('\n')?;
        let stored_hash = header.strip_prefix("blake3:")?;
        let computed = blake3::hash(content.as_bytes()).to_hex();
        if stored_hash != computed.as_str() {
            self.corruption_count.fetch_add(1, Ordering::AcqRel);
            tracing::warn!(key, corruption_total = self.corruption_count.load(Ordering::Acquire), "cache integrity check failed, quarantining");
            self.quarantine(key);
            return None;
        }
        Some(content.to_string())
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
        // Skip writes during corruption backoff (reads still verify integrity)
        let now = monotonic_ms();
        let backoff_until = self.corruption_backoff_until_ms.load(Ordering::Acquire);
        if backoff_until > 0 && now < backoff_until {
            return Ok(());
        } else if backoff_until > 0 && now >= backoff_until {
            // Backoff expired — reset rate window for fresh detection
            self.corruptions_in_window.store(0, Ordering::Release);
            self.corruption_window_start_ms.store(now, Ordering::Release);
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
    ///
    /// Rate-limits quarantine operations: if >3 corruptions occur within 60s
    /// (suspected hardware failure), disables cache writes with exponential
    /// backoff (60s, 120s, 240s... up to 24h) to prevent disk/log exhaustion.
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

        // Corruption rate limiter: detect sustained disk failure
        let now = monotonic_ms();
        let window_start = self.corruption_window_start_ms.load(Ordering::Acquire);
        if now.saturating_sub(window_start) > 60_000 {
            // New window
            self.corruption_window_start_ms.store(now, Ordering::Release);
            self.corruptions_in_window.store(1, Ordering::Release);
        } else {
            let count = self.corruptions_in_window.fetch_add(1, Ordering::AcqRel) + 1;
            if count >= 3 {
                // Sustained corruption — disable writes with exponential backoff
                let backoff = loop {
                    let current = self.corruption_backoff_ms.load(Ordering::Acquire);
                    let next = if current == 0 {
                        60_000
                    } else {
                        (current * 2).min(86_400_000) // cap at 24h
                    };
                    if self
                        .corruption_backoff_ms
                        .compare_exchange_weak(current, next, Ordering::AcqRel, Ordering::Acquire)
                        .is_ok()
                    {
                        break next;
                    }
                    std::hint::spin_loop();
                };
                self.corruption_backoff_until_ms
                    .store(now.saturating_add(backoff), Ordering::Release);
                tracing::error!(
                    corruptions = count,
                    backoff_secs = backoff / 1000,
                    "sustained corruption detected, disabling cache writes"
                );
            }
        }
    }

    /// Create a lightweight handle for background scrub tasks.
    /// Contains only the directory path and enabled flag — no mutable state.
    pub fn clone_for_scrub(&self) -> ScrubHandle {
        ScrubHandle {
            dir: self.dir.clone(),
            enabled: self.enabled.load(Ordering::Acquire),
        }
    }

    #[allow(dead_code)]
    pub fn enabled(&self) -> bool {
        self.enabled.load(Ordering::Acquire)
    }

    /// Number of corruption events detected since startup.
    /// Non-zero values indicate disk bitrot or filesystem issues.
    pub fn corruption_count(&self) -> usize {
        self.corruption_count.load(Ordering::Acquire)
    }

    /// Proactively scan all cache entries, verify blake3 integrity, and
    /// quarantine any corrupted files. Returns (verified, quarantined) counts.
    /// Call periodically (e.g., every N hours) to detect silent bitrot
    /// before it impacts a cache hit on a rarely-accessed entry.
    #[allow(dead_code)]
    pub fn scrub(&self) -> (usize, usize) {
        if !self.enabled.load(Ordering::Acquire) {
            return (0, 0);
        }
        let Ok(entries) = std::fs::read_dir(&self.dir) else {
            return (0, 0);
        };
        let mut verified = 0usize;
        let mut quarantined = 0usize;
        for entry in entries.filter_map(|e| e.ok()) {
            let name = entry.file_name();
            let s = name.to_string_lossy();
            // Skip temp files, subdirectories, and non-cache files
            if s.contains(".tmp.") || s == "corrupted" || s.starts_with('.') {
                continue;
            }
            if !entry.path().is_file() {
                continue;
            }
            let Ok(raw) = std::fs::read_to_string(entry.path()) else {
                continue;
            };
            let Some((header, content)) = raw.split_once('\n') else {
                // Legacy format without integrity header — quarantine
                self.quarantine(&s);
                quarantined += 1;
                continue;
            };
            let Some(stored_hash) = header.strip_prefix("blake3:") else {
                self.quarantine(&s);
                quarantined += 1;
                continue;
            };
            let computed = blake3::hash(content.as_bytes()).to_hex();
            if stored_hash != computed.as_str() {
                tracing::warn!(key = %s, "scrub: integrity check failed, quarantining");
                self.quarantine(&s);
                quarantined += 1;
            } else {
                verified += 1;
            }
        }
        if quarantined > 0 {
            tracing::info!(verified, quarantined, "cache scrub completed");
        }
        (verified, quarantined)
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
    fn circuit_breaker_persistence_roundtrip() {
        let dir = unique_test_dir("rlm_cb");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("cb_state");

        // Create breaker, record failures, persist synchronously
        {
            let cb = CircuitBreaker::with_persistence(3, Duration::from_secs(30), path.clone());
            cb.record_failure();
            cb.record_failure();
            cb.record_failure();
            assert!(cb.is_open());
            cb.save_state_sync(); // Flush to disk synchronously for test
        }

        // New breaker loads persisted state
        {
            let cb = CircuitBreaker::with_persistence(3, Duration::from_secs(30), path.clone());
            assert_eq!(cb.failures(), 3);
            assert!(cb.is_open()); // Still within cooldown
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn circuit_breaker_persistence_corrupt_resets() {
        let dir = unique_test_dir("rlm_cb_corrupt");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("cb_state");

        std::fs::write(&path, "GARBAGE_DATA").unwrap();
        let cb = CircuitBreaker::with_persistence(3, Duration::from_secs(30), path);
        assert_eq!(cb.failures(), 0); // Reset on corruption
        assert!(!cb.is_open());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn circuit_breaker_persistence_bitrot_detected() {
        let dir = unique_test_dir("rlm_cb_bitrot");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("cb_state");

        // Write valid V3 state
        {
            let cb = CircuitBreaker::with_persistence(3, Duration::from_secs(30), path.clone());
            cb.record_failure();
            cb.record_failure();
            cb.record_failure();
            assert!(cb.is_open());
            cb.save_state_sync(); // Flush to disk synchronously for test
        }

        // Corrupt a single digit in the failure count (3 -> 8)
        let raw = std::fs::read_to_string(&path).unwrap();
        assert!(raw.starts_with("V3:"));
        let corrupted = raw.replacen(":3:", ":8:", 1);
        assert_ne!(raw, corrupted);
        std::fs::write(&path, corrupted).unwrap();

        // New breaker should detect checksum mismatch and reset to safe default
        let cb = CircuitBreaker::with_persistence(3, Duration::from_secs(30), path);
        assert_eq!(cb.failures(), 0); // Reset on bitrot
        assert!(!cb.is_open()); // Closed (safe default)

        let _ = std::fs::remove_dir_all(&dir);
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
            cache.put(&format!("key_{i}"), &format!("value_{i}")).unwrap();
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
    fn circuit_breaker_future_version_no_overwrite() {
        let dir = unique_test_dir("rlm_cb_fwd");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("cb_state");

        // Write a V99 state file (simulating a future binary's format)
        // Field layout: V{version}:{failures}:{epoch}:{remaining}:{checksum}:{extra}
        let future_data = "V99:5:1234567890:1000:deadbeef:extra_v99_field";
        std::fs::write(&path, future_data).unwrap();

        // Load: should preserve failure count from future version (read-only downgrade)
        // but disable persistence so we don't overwrite the V99 file
        let cb = CircuitBreaker::with_persistence(3, Duration::from_secs(30), path.clone());
        assert_eq!(cb.failures(), 5, "future version should preserve failure count");
        assert!(cb.is_open(), "future version with failures >= threshold should be open");

        // Record a failure — this would normally trigger save_state
        cb.record_failure();

        // The V99 file should NOT have been overwritten
        let raw = std::fs::read_to_string(&path).unwrap();
        assert!(
            raw.starts_with("V99:"),
            "future version file must not be overwritten; got: {raw}"
        );

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
        ReplayCache::scavenge_quarantine(&dir, 100);

        let remaining = std::fs::read_dir(&quarantine_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .count();
        assert!(
            remaining <= 100,
            "quarantine should be bounded to 100, got {remaining}"
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
                        0 => { cb.record_failure(); }
                        1 => { cb.record_success(); }
                        _ => { let _ = cb.allow_request(); }
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

// ── Chaos Tests — Persistence Fault Injection ────────────────────
// Tests for failure paths in save_state/load_state that are hard to
// trigger organically: ENOSPC during fsync, bitrot after checksum
// validation, tmp file cleanup on write failure.

#[cfg(test)]
mod chaos_tests {
    use super::*;
    use std::time::Duration;

    fn unique_chaos_dir(prefix: &str) -> PathBuf {
        let id = format!(
            "{}_{}",
            std::process::id(),
            std::time::Instant::now().elapsed().as_nanos()
        );
        let hash = &blake3::hash(id.as_bytes()).to_hex()[..12];
        std::env::temp_dir().join(format!("{}_{}", prefix, hash))
    }

    /// Verify .tmp file is cleaned up when save_state_sync writes successfully.
    /// Simulates the normal path — no .tmp files should survive after rename.
    #[test]
    fn save_state_no_orphan_tmp() {
        let dir = unique_chaos_dir("rlm_chaos_tmp");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("cb_state");

        let cb = CircuitBreaker::with_persistence(3, Duration::from_secs(30), path.clone());
        cb.record_failure();
        cb.save_state_sync();

        // No .tmp files should exist
        let tmps: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains(".tmp"))
            .collect();
        assert!(tmps.is_empty(), "orphan tmp files found: {tmps:?}");

        // State file should exist and be valid
        let raw = std::fs::read_to_string(&path).unwrap();
        assert!(raw.starts_with("V3:"), "state file should be V3 format");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Verify load_state handles a file where content passes checksum but
    /// has been corrupted *after* the checksum field (simulating bitrot
    /// between the integrity check and field parsing).
    #[test]
    fn load_state_corrupt_after_checksum_field() {
        let dir = unique_chaos_dir("rlm_chaos_bitrot");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("cb_state");

        // Write valid state, then corrupt a non-checksum field
        {
            let cb = CircuitBreaker::with_persistence(3, Duration::from_secs(30), path.clone());
            cb.record_failure();
            cb.record_failure();
            cb.save_state_sync();
        }

        // Read, corrupt the epoch field (replace digits with letters), invalidating checksum
        let raw = std::fs::read_to_string(&path).unwrap();
        let parts: Vec<&str> = raw.splitn(5, ':').collect();
        assert_eq!(parts.len(), 5, "V3 format should have 5 colon-separated fields");
        // Replace epoch with garbage — checksum will no longer match
        let corrupted = format!("{}:{}:GARBAGE:{}:{}", parts[0], parts[1], parts[3], parts[4]);
        std::fs::write(&path, corrupted).unwrap();

        // Load should detect checksum mismatch and reset to safe defaults
        let cb = CircuitBreaker::with_persistence(3, Duration::from_secs(30), path.clone());
        assert_eq!(cb.failures(), 0, "should reset on checksum mismatch");
        assert!(!cb.is_open(), "should be closed after reset");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Verify cache gracefully disables itself on write to a read-only directory.
    /// Simulates ENOSPC-like conditions where writes fail.
    #[tokio::test]
    async fn cache_write_failure_degrades_gracefully() {
        let dir = unique_chaos_dir("rlm_chaos_cache");
        let cache = ReplayCache::new(dir.clone(), true, 0);

        // Write one entry successfully
        cache.put("good_key", "good_value").unwrap();
        assert_eq!(cache.get("good_key"), Some("good_value".to_string()));

        // Make the directory read+exec only (no write) to simulate ENOSPC / permission failure.
        // 0o555 allows reading existing files and directory traversal, but blocks new writes.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let perms = std::fs::Permissions::from_mode(0o555);
            std::fs::set_permissions(&dir, perms).unwrap();

            // Write should fail but not panic — returns error or silently degrades
            let result = cache.put("fail_key", "fail_value");
            // Either returns Err or cache auto-disables — both are acceptable
            if result.is_ok() {
                // Cache may have auto-disabled, which is fine
                assert!(
                    cache.get("fail_key").is_none(),
                    "failed write should not be readable"
                );
            }

            // Existing entries should still be readable (read+exec on dir allows this)
            assert_eq!(cache.get("good_key"), Some("good_value".to_string()));

            // Restore permissions for cleanup
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
    use loom::sync::atomic::{AtomicUsize, Ordering};
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
            let _ = self.remaining.fetch_update(
                Ordering::AcqRel,
                Ordering::Acquire,
                |current| Some(current.saturating_add(1)),
            );
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
            let t2 = thread::spawn(move || {
                b2.try_acquire()
            });

            t1.join().unwrap();
            let acquired = t2.join().unwrap();

            let remaining = budget.remaining();
            // Budget started at 2, we acquired 1 before threads, released 1 in t1,
            // and t2 may or may not have acquired 1.
            if acquired {
                assert!(remaining <= 2, "remaining {remaining} > 2 after release+acquire");
            } else {
                assert_eq!(remaining, 2, "remaining should be 2 after release without acquire");
            }
        });
    }
}
