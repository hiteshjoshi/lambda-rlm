# Changelog

### v3.8 — OpenCode Resilience Hardening (`a3a007e`)

**Hardened:**
- Added ENOSPC degradation in `src/codegen.rs`: when `.lambda-rlm-result.md` cannot be written due to disk-full conditions, the fix loop falls back to an inline bounded analysis prompt instead of aborting.
- Added atomic, symlink-safe log writes in `src/codegen.rs` for both Claude and OpenCode iteration logs.
- Added strict shutdown leak enforcement in `src/main.rs` to fail the run if live guard counters are non-zero.

**Secured:**
- Expanded secret scrubbing in `src/main.rs`, `src/oracle.rs`, and `src/codegen.rs` to redact both `FIREWORKS_API` and `OPENCODE_*` values.
- Hardened single-file collection path in `src/main.rs` with `open_file_no_follow`, plus saturating aggregate-byte accounting and owned FD semaphore permits.

**Tested:**
- Added `child_cleanup_drop_reaps_process` in `src/codegen.rs` to assert dropped generator children are reaped within 5 seconds.

**Net: +205 / -47 across runtime hardening paths. 56/56 tests pass.**

### v3.7 — Codegen Process Hardening (`f58f5b4`)

**Hardened:**
- Added bounded execution for both generators: Claude now has a 15-minute timeout and OpenCode keeps the 5-minute cap.
- Enabled `kill_on_drop(true)` and Unix process-group isolation to prevent orphaned subprocesses in abrupt shutdown scenarios.
- Switched to generic sanitized user-facing codegen errors while preserving full internal tracing details.

**Secured:**
- Added input validation for generated result payloads (non-empty, max size, no NUL bytes).
- Added atomic write+rename flow for `.lambda-rlm-result.md` and symlink guardrails.

**Operational:**
- Added log rotation for `.lambda-rlm-claude-*.log` and `.lambda-rlm-opencode-*.log` with LRU-style retention (10 files/generator).
- Added unit tests for result validation and summary-line parsing helpers.

**Net: +185 / -35 in `src/codegen.rs`. 42/42 tests pass.**

### v3.6 — OpenCode Code Generator (`84b9bb5`)

**Added:**
- `--opencode` CLI flag: enables fix loop using OpenCode CLI as an alternative to `--claude`.
- `CodeGenerator` enum (`Claude | Opencode`) in `types.rs` for extensible code generator selection.
- `codegen.rs` module: shared prompt builder, dispatch function, and per-generator subprocess runners.
- Mutual exclusivity validation: `--claude` and `--opencode` cannot be used together.

**Refactored:**
- Extracted `run_claude()` from `main.rs` into `codegen.rs` with shared `build_prompt()`.
- Config fingerprint now includes the `opencode` flag.

**Net: +1 file (codegen.rs), -88 lines from main.rs refactor. 38/38 tests pass.**

### v3.5 — Stale Docstring Purge (`0c0c42f`)

**Removed:**
- CircuitBreaker header comment referencing disk persistence (removed in v3.2).
- `quarantine()` docstring describing rate-limiting/exponential backoff (removed in v3.2).

**Net: -5 lines removed. 38/38 tests pass.**

### v3.4 — Env Var Purge & CB Rejuvenation (`40c0188`)

**Removed:**
- `LAMBDA_RLM_MAX_INPUT_BYTES` env var override: `phi()` input limit is now a compile-time constant (`MAX_PHI_INPUT_BYTES`). Eliminates `OnceLock` + env lookup on hot path.
- `LAMBDA_RLM_BULKHEAD_LLM_PERMITS` env var override: bulkhead permits now come strictly from `--concurrency` CLI flag. No runtime env var bypass.
- `KNOWN_VARS` env var validation block in `main()`: no longer needed with zero `LAMBDA_RLM_*` env vars.
- `#[allow(dead_code)]` annotations on `try_acquire_n` and `release_n` — both are actively used by budget pre-flight reservation in `phi.rs` and `oracle.rs`.

**Optimized:**
- CircuitBreaker rejuvenation: replaced CAS fetch_update loop (fires every 10,000 failures) with a direct `store` (fires every `threshold` failures). Faster convergence, simpler code.

**Net: -48 lines removed. 38/38 tests pass.**

### v3.3 — Dead Code & Dependency Purge (`efe9873`)

**Removed:**
- `unicode-normalization` crate: NFKC normalization removed from hot-path `keyword_matches()` and `merge_dedup()`. Source code is ASCII — simple `to_lowercase()` is sufficient and eliminates per-chunk String allocation from normalization.
- `unicode-segmentation` crate: grapheme-aware truncation in `phi()` input guard replaced with char-boundary truncation. Grapheme clusters add no value for source code processing.
- `ErrorKind` enum (dead code, never used outside `#[allow(dead_code)]`).
- `LlmProvider` trait and `ProviderFuture` type (dead code, unused abstraction).
- `Bulkhead::cpu` semaphore and `Oracle::bulkhead()` accessor (dead code).
- `OracleError::kind()` method (dead code).

**Net: -153 lines removed, +6 added. 2 crate dependencies eliminated. 38/38 tests pass.**

### v3.2 — Subtractive Optimization (`327eacd`)

**Removed:**
- Circuit breaker disk persistence: `save_state`, `load_state`, V1/V2/V3 schema parsing, blake3 checksums, EXDEV fallback, read-after-write verification. In-memory CB is sufficient for CLI usage.
- Background cache scavenger: 6h scrub / 24h compaction timers (never fire for a CLI tool). Removes `ScrubHandle`, `compact()`, and background `tokio::spawn` task.
- Corruption rate-limiter: exponential backoff with 4 atomic fields and CAS spin loop. Quarantine still works, rate-limiting was over-engineered.

**Fixed:**
- JoinSet drain on depth-0 errors in `phi.rs`: remaining tasks are now `abort_all` + drained before returning, ensuring `SemaphorePermit` and `BudgetGuard` RAII guards drop cleanly.

**Net: -894 lines removed, +26 added. 38/38 tests pass.**

### v3.1 — Resilience & Performance Hardening (`4527ee1`)

**Removed:**
- Quorum consensus (`--quorum`, 3x LLM calls, trigram/token similarity voting) — correctness is cryptographic, not statistical. Saves 3x token cost and removes non-deterministic latency.
- `walkdir` crate — replaced with manual `std::fs::read_dir` stack-based recursion. Removes ~10 transitive dependencies, smaller binary.
- 4 quorum-related env vars (`LAMBDA_RLM_QUORUM_TIMEOUT_SECS`, `LAMBDA_RLM_MIN_QUORUM_SIZE`, `LAMBDA_RLM_MIN_CONSENSUS_SIMILARITY`, `LAMBDA_RLM_QUORUM_DEGRADE_ON_SPLIT`).

**Optimized:**
- CAS spin loops in `CallBudget` now yield after 10 failures instead of spinning forever under contention.
- Symbolic reducers (`reduce_aggregate`, `reduce_pairwise_intermediate`) pre-allocate output strings to eliminate heap fragmentation.
- CLI validates `k <= 16` and checks `k^depth < 100K` at parse time to prevent pathological expansion.

**Hardened:**
- `BudgetGuard` live count promoted from debug-only to production `AtomicU64`. Uncommitted drops now log `error!` (catches async cancellation leaks).
- Circuit breaker `save_state` handles EXDEV (cross-device rename) via copy+delete fallback — safe when auto-TLS places cache on a different mount.

**Net: -650 lines removed, +177 added. 44/44 tests pass.**
