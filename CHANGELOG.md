# Changelog

## Recent fix-loop hardening commits (moved from README)

### `a065094`

- Fixed interactive TTY hangs in `src/codegen.rs` by removing Unix `process_group(0)` from interactive generator command setup so child TUIs stay in the terminal foreground process group and can read stdin reliably.
- Split interactive timeout semantics in `src/codegen.rs`: `--interactive-timeout` now bounds startup probing only, then interactive sessions wait for natural exit without forced timeout cancellation.
- Added interactive regression coverage in `src/codegen.rs` for startup-timeout behavior and early non-success interactive exits.

### `ea8b71c`

- Fixed the interactive codegen deadlock in `src/codegen.rs` by moving code-generator bulkhead/budget acquisition to the non-interactive path only, so long-lived TTY sessions no longer hold permits that block other runtime work.
- Hardened interactive failure classification in `src/codegen.rs`: timeout/wait failures are tagged `codegen_retryable`, missing-binary spawn failures are tagged `codegen_fatal`, and regression tests now assert both markers.
- Added post-cleanup invariants in `src/phi.rs` with `debug_assert!(set.is_empty())` after `abort_and_drain` and `return_joinset`, making JoinSet handle leaks fail fast in debug/test runs.

### `d860423`

- Tightened leak detection at process exit in `src/main.rs` by extending `ensure_no_live_guards` to fail on non-zero `inflight_guards_live` in addition to budget/codegen guard counters.
- Hardened guard lifecycles in `src/codegen.rs` and `src/main.rs` with `#[must_use]` on runtime guards (`CodegenFlightGuard`, `ChildCleanup`, `GuardedFile`) and panic-safe `catch_unwind` drop wrappers on their cleanup paths.
- Improved shutdown responsiveness in `src/phi.rs` by probing `watch::Receiver::has_changed()` before invoking expensive reduce work, reducing wasted compute after shutdown is signaled.
- Added regression coverage in `src/codegen.rs` for `is_storage_full_error` (`ErrorKind::StorageFull` + common ENOSPC text signatures).

### `9d30728`

- Hardened OpenCode runtime safety in `src/codegen.rs` by caching `opencode --version` validation in-process, pre-warming the OpenCode summary fallback regex during preflight, and re-validating version compatibility in `run_opencode` before each spawn so fatal drift is isolated by the OpenCode circuit.
- Added deterministic ENOSPC degradation coverage in `tests/e2e_codegen.rs` with an OpenCode CLI stub to prove fix-loop execution falls back to inline analysis prompts (without `--file`) when `.lambda-rlm-result.md` cannot be persisted.
- Tightened single-file ingest FD hygiene in `src/main.rs` by routing the file path through `GuardedFile` + `FD_SEMAPHORE`, aligning one-file and directory scans under the same permit lifecycle guarantees.

### `4d9085c`

- Hardened `src/codegen.rs` with a TTL scavenger for `CODEGEN_SINGLE_FLIGHT`, generator-prefixed single-flight keys (`claude:`/`opencode:`), and stronger no-runtime child-process reaping so stale leader slots and orphaned generator processes do not accumulate.
- Tightened leak and preflight guarantees in `src/main.rs` by releasing `FD_SEMAPHORE` permits before file-handle drop in `GuardedFile`, adding an unwind-path permit-return test, and adding a fail-fast `validate_generator_access` regression test for missing `opencode` binaries.
- Unified secret redaction in `src/main.rs`, `src/codegen.rs`, and `src/oracle.rs` to explicitly cover `CLAUDE_API_KEY` alongside `FIREWORKS_API`/`OPENCODE_*`, and added integration coverage in `tests/e2e_codegen.rs` to assert `--opencode` fails before fix-loop execution when PATH does not contain the binary.
- Added `#[must_use]` to `merge_dedup_arc` in `src/combinator.rs` to guard against accidental drop of deduplicated `Arc<str>` results in hot aggregation paths.

### `1756c75`

- Hardened code-generator execution in `src/codegen.rs` with content-addressed replay caching (`b3-codegen_*`), per-generator concurrency bulkheads, symlink-safe result-file checks, and regex-backed OpenCode summary fallback that degrades safely on format drift.
- Tightened file-ingest and result-target safety in `src/main.rs` by binding open-file permits to file handles (`GuardedFile`) and enforcing explicit symlink-metadata validation before codegen handoff.
- Added shutdown checks before every Phi LLM/reduce call in `src/phi.rs`, documented architecture trade-offs in `ARCHITECTURE.md`, and introduced `tests/e2e_codegen.rs` as the idempotency boundary harness.

### `e7dea1b`

- Added code-generator single-flight deduplication in `src/codegen.rs` (`DashMap` + `watch`) so concurrent identical fix-loop retries share one Claude/OpenCode subprocess instead of spawning duplicates.
- Kept Claude/OpenCode failure-domain isolation while single-flight is active by preserving per-generator circuit breakers, budget-guard semantics, and generator-scoped unavailable error contexts.
- Added a dedicated loom-model test scaffold in `src/codegen.rs` for codegen-budget restoration race analysis (marked ignored for explicit model-check runs).

### `a3a007e`

- Hardened `src/codegen.rs` for disk-pressure resilience: if writing `.lambda-rlm-result.md` hits ENOSPC, the fix loop degrades to an inline in-memory prompt instead of failing the iteration.
- Added atomic log writes with symlink-safe preflight in `src/codegen.rs` so `.lambda-rlm-claude-*.log` / `.lambda-rlm-opencode-*.log` are written via temp+persist flow.
- Tightened shutdown-leak guarantees in `src/main.rs` by failing the run if budget/codegen guard live-counters are non-zero at exit.
- Expanded secret redaction in `src/main.rs`, `src/oracle.rs`, and `src/codegen.rs` to scrub both `FIREWORKS_API` and `OPENCODE_*` values from error surfaces.

### `ddf3f6c`

- Added OpenCode production hardening in `src/codegen.rs` and `src/main.rs`: version handshake (`opencode 1.x`), stricter per-generator timeouts, control-character rejection in OpenCode summary parsing, and safer child-process reap behavior in drop paths.
- Added semantic replay-cache validation in `src/resilience.rs` and wired it in `src/oracle.rs` so malformed cached payloads are quarantined and never replayed into runtime logic.
- Added guard-lifecycle hardening (`#[must_use]` for codegen/inflight guards, live codegen guard telemetry at fix-loop shutdown), API-key scrubbing in error logs, and a new GitHub CI workflow with nightly Miri leak checks.
- Added FD admission control in `src/main.rs` file collection path to prevent runaway open-file pressure under very large repository scans.

### `4578ddc`

- Hardened `src/codegen.rs` result-file durability by adding timeout-bounded blocking writes with fsync on temp/persisted files and best-effort directory sync.
- Added panic-safe budget restoration in `CodegenBudgetGuard::drop` so failed/unwound generator paths cannot cascade into double-panic abort behavior.
- Improved OpenCode summary extraction to skip common CLI preamble/header lines and added regression coverage for header-heavy OpenCode output.

### `0e6ba39`

- Updated code-generator summary extraction in `src/codegen.rs` to honor generator-specific contracts: Claude uses the last non-empty line and OpenCode uses the first non-empty line.
- Moved generator binary preflight checks into `Cli::validate()` in `src/main.rs` so invalid `--claude`/`--opencode` setups fail fast during startup validation.

### `5e87b79`

- Added `INFLIGHT_GUARD_LIVE_COUNT` tracking in `src/oracle.rs`, wired it into `OracleMetrics`, and exposed both `Budget guards live` and `Inflight guards live` in telemetry output for leak visibility at steady state.
- Added a loom race test in `src/oracle.rs` (`inflight_guard_always_drops_under_race`) to prove single-flight guard cleanup leaves no stuck ownership under concurrent leader races.
- Hardened `ChildCleanup::drop` in `src/codegen.rs` to avoid runtime-assumption failures by doing async kill+wait when a runtime handle exists and synchronous best-effort kill/reap fallback otherwise.

### `892138e`

- Enforced Oracle resource acquisition as a typed RAII stack in `src/oracle.rs` (`Budget -> Cache -> Bulkhead`), added inflight leak gauges/invariants, and exposed production metrics (`cache hit rate`, `circuit state`, `budget guard live`, `inflight count`).
- Switched SSE buffering in `src/oracle.rs` from `Vec<u8>` growth/drain patterns to `bytes::BytesMut` to reduce hot-path realloc/copy pressure during streamed token decoding.
- Added JoinSet pooling to `src/phi.rs` and wired shared pool ownership from `src/main.rs` to reduce recursive task-allocation churn under deep decomposition.
- Hardened `.lambda-rlm-result.md` write validation in `src/codegen.rs` with `O_NOFOLLOW`-style preflight and aligned Claude/OpenCode summary extraction to a single terminal-line contract.
- Changed oversized source collection in `src/main.rs` from hard-fail to bounded truncation with explicit marker comments, preserving forward progress on large repositories.

### Earlier hardening batches

- Replaced Oracle single-flight dedup state in `src/oracle.rs` from a global `Mutex<HashMap<...>>` to sharded `DashMap`, removing the global lock bottleneck under concurrent same-key requests.
- Added panic-safe `catch_unwind` protection in `BudgetGuard::drop` (`src/resilience.rs`) so budget restoration cannot trigger double-panic abort paths.
- Added non-Windows `jemallocator` and wired it as the global allocator in `src/main.rs` to reduce long-run heap fragmentation in high-throughput deployments.
- Aligned source collection ingress cap in `src/main.rs` to 256MB so file ingest now matches Phi's hard input safety boundary.
- Hardened `src/codegen.rs` with cancellation-safe child-process cleanup (`Drop` reaper + explicit kill/reap paths), bounded stdout/stderr readers, exit-code classification (retryable vs fatal), and degraded handling for empty generator output.
- Switched codegen summary passing to `Arc<str>` in `src/codegen.rs` to avoid extra string copies across fix-loop boundaries.
- Moved shutdown signal handler ownership out of per-iteration analysis in `src/main.rs` so the fix loop no longer spawns a new signal task each iteration.
- Reused a single Oracle instance across fix-loop iterations in `src/main.rs`, preserving budget/telemetry continuity and preventing per-iteration state resets.
- Enforced a hard safety cap of 10 fix-loop iterations in `src/main.rs` and added budget-exhaustion termination checks for bounded production behavior.
- Hardened code-generator subprocess lifecycle in `src/codegen.rs` so spawned processes are explicitly killed and reaped on timeout and on stdout/stderr capture failures, eliminating zombie risk under repeated fix-loop iterations.
- Switched codegen breaker initialization in `src/codegen.rs` to independent per-generator `OnceLock` instances, preserving Claude/OpenCode fault isolation.
- Made `BudgetGuard` in `src/resilience.rs` atomic-commit safe to prevent double-release on unwind/cancellation edge paths, and added loom race coverage for half-open probe gating.
- Updated `abort_and_drain` in `src/phi.rs` to `detach_all()` after bounded abort-drain timeout so lingering blocked tasks cannot accumulate join handles.
- Added final top-level error sanitization in `src/main.rs` to return `Not Found` while logging full internal error chains, and documented/verified Oracle↔Phi single-flight ownership invariants with race-focused tests.
- Centralized Claude/OpenCode summary extraction in `src/codegen.rs` with explicit per-generator output invariants and dedicated tests, reducing parser drift risk.
- Added `CodegenBudgetGuard` drop-path telemetry in `src/codegen.rs` so cancelled/failed codegen runs visibly restore reserved budget units.
- Added bounded JoinSet abort-drain cleanup in `src/phi.rs` to prevent semaphore/budget guard retention during cancellation and fatal child errors.
- Reduced peak file collection memory in `src/main.rs` by streaming file content directly into the aggregate buffer instead of buffering per-file copies.
- Added panic/abort chaos tests for `BudgetGuard` in `src/resilience.rs` to prove budget units are restored across unwind and task cancellation paths.

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
