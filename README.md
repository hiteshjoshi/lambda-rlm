# lambda-RLM

Recursive map-reduce runtime for LLMs. Feed it a codebase and a question — it chunks at function boundaries, processes each chunk with an LLM, and reduces results back up. Like MapReduce, but for reasoning over code.

Based on [arXiv:2603.20105](https://arxiv.org/abs/2603.20105) (Roy et al., 2026).

## Why this exists

LLMs degrade past ~8K tokens. Feed a 100K-char codebase into one call and you get hallucinations, missed details, and vague answers. Lambda-RLM fixes this by recursively splitting the input into chunks small enough for accurate processing, then merging results back up with task-specific reduce operators.

## Install

```bash
curl -sSf https://raw.githubusercontent.com/hiteshjoshi/lambda-rlm/main/install.sh | bash
```

This clones the repo to `~/.lambda-rlm`, builds a release binary, and copies it to `~/.cargo/bin/lambda_rlm`. Installs Rust automatically if you don't have it.

To update later, run the same command — it pulls and rebuilds.

## Setup

Lambda-RLM needs two things: a Fireworks API key (required) and Claude Code CLI (optional, for the fix loop).

### Fireworks API key

Lambda-RLM uses [Fireworks AI](https://fireworks.ai) for inference. The default model is **Kimi K2.5 Turbo** — a strong reasoning model available for free through Fireworks' **Fire Pass**.

1. Sign up at [fireworks.ai](https://fireworks.ai)
2. Activate **Fire Pass** — this is their free tier that includes Kimi K2.5 Turbo (no credit card needed)
3. Go to **API Keys**, create one
4. Add to your shell profile:

```bash
# ~/.zshrc or ~/.bashrc
export FIREWORKS_API="your-key-here"
```

Why Fireworks and not OpenAI/Anthropic directly? Fireworks routes to the best available model behind their API, Kimi K2.5 Turbo is free via Fire Pass, and their endpoint handles streaming for large outputs. You can swap models with `--model` if you have access to others.

### Code generator CLI (optional)

The fix loop uses a code generator (Claude Code or OpenCode) to act on lambda-RLM's analysis. Pick one:

**Claude Code** (`--claude`):
```bash
npm install -g @anthropic-ai/claude-code
```

**OpenCode** (`--opencode`):
```bash
# Install opencode per https://github.com/sst/opencode
```

Both work the same way: Fireworks does the cheap, parallelized analysis; the code generator does the targeted code editing. The two flags are mutually exclusive.

## Usage

```bash
# Search for specific code
lambda_rlm -p ./src -q "Where is authentication handled?" -t search

# Aggregate items across files
lambda_rlm -p ./src -q "Find all API endpoints" -t aggregate

# Summarize a codebase
lambda_rlm -p ./src -q "What does this project do?"

# Trace a flow across modules
lambda_rlm -p ./src -q "How does a request go from handler to database?" -t multi-hop

# Classify each module
lambda_rlm -p ./src -q "What does each module do?" -t classify

# Find duplicates
lambda_rlm -p ./src -q "Find duplicate logic" -t pairwise

# Dry run (no API calls — test your setup)
lambda_rlm -p ./src -q "test" --dry-run
```

Task type is auto-detected if you don't pass `-t`.

## Fix loop

This is the main thing. Point lambda-RLM at your code with a question, add `--claude` or `--opencode`, and walk away:

```bash
# Using Claude Code
lambda_rlm -p ./src -q "Find and fix security vulnerabilities" --claude

# Using OpenCode
lambda_rlm -p ./src -q "Find and fix security vulnerabilities" --opencode
```

What happens:

```
 Iteration 1:
   lambda-RLM analyzes codebase (Fireworks, parallelized)
     → finds 5 issues
   Code generator reads findings, edits files, commits
     → fixes 5 issues

 Iteration 2:
   lambda-RLM re-analyzes (fresh scan of modified code)
     → finds 1 remaining issue
   Code generator fixes it, commits

 Iteration 3:
   lambda-RLM re-analyzes
     → clean
   Done.
```

Each iteration: analyze → fix → re-analyze. Stops when clean or max iterations hit.

```bash
# 5 iterations max
lambda_rlm -p ./src -q "Fix all bugs" --claude --max-iterations 5

# Run until clean
lambda_rlm -p ./src -q "Fix all bugs" --claude --max-iterations 0
```

Both generators run in print mode — fully autonomous, no prompts. Logs are saved to `.lambda-rlm-claude-{n}.log` or `.lambda-rlm-opencode-{n}.log` in the target directory.

## How it works

```
Input (100K chars)
  │
  Chunk (structural split at fn/struct/class boundaries)
  │
  ├── Leaf (6K) → LLM → Verify → result
  ├── Leaf (6K) → LLM → Verify → result
  ├── Leaf (6K) → LLM → Verify → result
  └── ...
  │
  Reduce (task-specific merge)
  │
Output (verified, grounded answer)
```

The recursive executor (Phi) applies the core equation from the paper:

```
fix(λf. λP.
  if |P| ≤ τ  then  Verify(M(P))
  else  Reduce(⊕, Map(λpi. f(pi), Chunk(P, k)))
)
```

Six task types, each with specialized leaf prompts and reduce operators:

| Task | Reduce type | What it does |
|------|------------|--------------|
| `search` | Symbolic | Concatenates matching results |
| `classify` | Symbolic | Merges classification labels |
| `aggregate` | Symbolic | Deduplicates extracted items |
| `pairwise` | Hybrid | Cross-product pairs → LLM comparison |
| `summarise` | Neural | LLM synthesizes section summaries |
| `multi-hop` | Neural | LLM chains evidence fragments |

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `-p, --path` | required | File or directory to analyze |
| `-q, --question` | required | Question or task description |
| `-t, --task` | `auto` | search, classify, aggregate, pairwise, summarise, multi-hop |
| `-w, --window` | `6000` | Leaf chunk size (chars) |
| `-k` | `0` (auto) | Split factor per recursion level |
| `--concurrency` | `8` | Max parallel LLM calls |
| `--model` | `kimi-k2p5-turbo` | Fireworks model ID |
| `--max-tokens` | `8192` | Max output tokens |
| `--claude` | `false` | Enable fix loop with Claude Code |
| `--opencode` | `false` | Enable fix loop with OpenCode |
| `--max-iterations` | `10` | Fix loop iterations (0 = unlimited) |
| `--dry-run` | `false` | No API calls |
| `--no-cache` | `false` | Disable replay cache |
| `--timeout` | `120` | Per-call timeout (seconds) |
| `--max-calls` | `0` | Max total LLM calls (0 = unlimited) |
| `--max-retries` | `2` | Retries per failed call |

## Architecture

```
src/
  main.rs        — CLI, file collector, fix loop driver
  types.rs       — TaskType, CodeGenerator enums (shared across modules)
  codegen.rs     — post-RLM code generation: Claude CLI / OpenCode CLI dispatch
  combinator.rs  — structural chunking, keyword extraction, text splitting
  oracle.rs      — Fireworks API client, semaphore, retries, streaming
  resilience.rs  — circuit breaker, RAII budget guard, replay cache (blake3)
  cost.rs        — cost model, optimal plan computation (Theorem 4)
  verify.rs      — leaf output validation, refusal detection
  reduce.rs      — task-specific reduce operators (symbolic + neural)
  phi.rs         — recursive executor (Algorithm 2)
```

Hardening: circuit breaker (3 failures / 30s cooloff, in-memory), RAII budget guards (panic-safe via Drop, production leak detection), blake3 content-addressed replay cache, structural chunking at definition boundaries, leaf verification with refusal detection, CAS backoff under contention.

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `FIREWORKS_API` | Yes | Fireworks AI API key ([get one free](https://fireworks.ai)) |
| `RUST_LOG` | No | Log level: `warn` (default), `info`, `debug` |

## Changelog

Latest fix-loop hardening commit: `9d30728`

- Hardened OpenCode runtime safety in `src/codegen.rs` by caching `opencode --version` validation in-process, pre-warming the OpenCode summary fallback regex during preflight, and re-validating version compatibility in `run_opencode` before each spawn so fatal drift is isolated by the OpenCode circuit.
- Added deterministic ENOSPC degradation coverage in `tests/e2e_codegen.rs` with an OpenCode CLI stub to prove fix-loop execution falls back to inline analysis prompts (without `--file`) when `.lambda-rlm-result.md` cannot be persisted.
- Tightened single-file ingest FD hygiene in `src/main.rs` by routing the file path through `GuardedFile` + `FD_SEMAPHORE`, aligning one-file and directory scans under the same permit lifecycle guarantees.

Latest fix-loop hardening commit: `4d9085c`

- Hardened `src/codegen.rs` with a TTL scavenger for `CODEGEN_SINGLE_FLIGHT`, generator-prefixed single-flight keys (`claude:`/`opencode:`), and stronger no-runtime child-process reaping so stale leader slots and orphaned generator processes do not accumulate.
- Tightened leak and preflight guarantees in `src/main.rs` by releasing `FD_SEMAPHORE` permits before file-handle drop in `GuardedFile`, adding an unwind-path permit-return test, and adding a fail-fast `validate_generator_access` regression test for missing `opencode` binaries.
- Unified secret redaction in `src/main.rs`, `src/codegen.rs`, and `src/oracle.rs` to explicitly cover `CLAUDE_API_KEY` alongside `FIREWORKS_API`/`OPENCODE_*`, and added integration coverage in `tests/e2e_codegen.rs` to assert `--opencode` fails before fix-loop execution when PATH does not contain the binary.
- Added `#[must_use]` to `merge_dedup_arc` in `src/combinator.rs` to guard against accidental drop of deduplicated `Arc<str>` results in hot aggregation paths.

Latest fix-loop hardening commit: `1756c75`

- Hardened code-generator execution in `src/codegen.rs` with content-addressed replay caching (`b3-codegen_*`), per-generator concurrency bulkheads, symlink-safe result-file checks, and regex-backed OpenCode summary fallback that degrades safely on format drift.
- Tightened file-ingest and result-target safety in `src/main.rs` by binding open-file permits to file handles (`GuardedFile`) and enforcing explicit symlink-metadata validation before codegen handoff.
- Added shutdown checks before every Phi LLM/reduce call in `src/phi.rs`, documented architecture trade-offs in `ARCHITECTURE.md`, and introduced `tests/e2e_codegen.rs` as the idempotency boundary harness.

Latest fix-loop hardening commit: `e7dea1b`

- Added code-generator single-flight deduplication in `src/codegen.rs` (`DashMap` + `watch`) so concurrent identical fix-loop retries share one Claude/OpenCode subprocess instead of spawning duplicates.
- Kept Claude/OpenCode failure-domain isolation while single-flight is active by preserving per-generator circuit breakers, budget-guard semantics, and generator-scoped unavailable error contexts.
- Added a dedicated loom-model test scaffold in `src/codegen.rs` for codegen-budget restoration race analysis (marked ignored for explicit model-check runs).

Latest fix-loop hardening commit: `a3a007e`

- Hardened `src/codegen.rs` for disk-pressure resilience: if writing `.lambda-rlm-result.md` hits ENOSPC, the fix loop degrades to an inline in-memory prompt instead of failing the iteration.
- Added atomic log writes with symlink-safe preflight in `src/codegen.rs` so `.lambda-rlm-claude-*.log` / `.lambda-rlm-opencode-*.log` are written via temp+persist flow.
- Tightened shutdown-leak guarantees in `src/main.rs` by failing the run if budget/codegen guard live-counters are non-zero at exit.
- Expanded secret redaction in `src/main.rs`, `src/oracle.rs`, and `src/codegen.rs` to scrub both `FIREWORKS_API` and `OPENCODE_*` values from error surfaces.

Latest fix-loop hardening commit: `ddf3f6c`

- Added OpenCode production hardening in `src/codegen.rs` and `src/main.rs`: version handshake (`opencode 1.x`), stricter per-generator timeouts, control-character rejection in OpenCode summary parsing, and safer child-process reap behavior in drop paths.
- Added semantic replay-cache validation in `src/resilience.rs` and wired it in `src/oracle.rs` so malformed cached payloads are quarantined and never replayed into runtime logic.
- Added guard-lifecycle hardening (`#[must_use]` for codegen/inflight guards, live codegen guard telemetry at fix-loop shutdown), API-key scrubbing in error logs, and a new GitHub CI workflow with nightly Miri leak checks.
- Added FD admission control in `src/main.rs` file collection path to prevent runaway open-file pressure under very large repository scans.

Latest fix-loop hardening commit: `4578ddc`

- Hardened `src/codegen.rs` result-file durability by adding timeout-bounded blocking writes with fsync on temp/persisted files and best-effort directory sync.
- Added panic-safe budget restoration in `CodegenBudgetGuard::drop` so failed/unwound generator paths cannot cascade into double-panic abort behavior.
- Improved OpenCode summary extraction to skip common CLI preamble/header lines and added regression coverage for header-heavy OpenCode output.

Latest fix-loop hardening commit: `0e6ba39`

- Updated code-generator summary extraction in `src/codegen.rs` to honor generator-specific contracts: Claude uses the last non-empty line and OpenCode uses the first non-empty line.
- Moved generator binary preflight checks into `Cli::validate()` in `src/main.rs` so invalid `--claude`/`--opencode` setups fail fast during startup validation.

Latest fix-loop hardening commit: `5e87b79`

- Added `INFLIGHT_GUARD_LIVE_COUNT` tracking in `src/oracle.rs`, wired it into `OracleMetrics`, and exposed both `Budget guards live` and `Inflight guards live` in telemetry output for leak visibility at steady state.
- Added a loom race test in `src/oracle.rs` (`inflight_guard_always_drops_under_race`) to prove single-flight guard cleanup leaves no stuck ownership under concurrent leader races.
- Hardened `ChildCleanup::drop` in `src/codegen.rs` to avoid runtime-assumption failures by doing async kill+wait when a runtime handle exists and synchronous best-effort kill/reap fallback otherwise.

Latest fix-loop hardening commit: `892138e`

- Enforced Oracle resource acquisition as a typed RAII stack in `src/oracle.rs` (`Budget -> Cache -> Bulkhead`), added inflight leak gauges/invariants, and exposed production metrics (`cache hit rate`, `circuit state`, `budget guard live`, `inflight count`).
- Switched SSE buffering in `src/oracle.rs` from `Vec<u8>` growth/drain patterns to `bytes::BytesMut` to reduce hot-path realloc/copy pressure during streamed token decoding.
- Added JoinSet pooling to `src/phi.rs` and wired shared pool ownership from `src/main.rs` to reduce recursive task-allocation churn under deep decomposition.
- Hardened `.lambda-rlm-result.md` write validation in `src/codegen.rs` with `O_NOFOLLOW`-style preflight and aligned Claude/OpenCode summary extraction to a single terminal-line contract.
- Changed oversized source collection in `src/main.rs` from hard-fail to bounded truncation with explicit marker comments, preserving forward progress on large repositories.

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

See [CHANGELOG.md](CHANGELOG.md).

## License

MIT
