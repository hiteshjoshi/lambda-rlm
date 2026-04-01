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

# Interactive Claude session (TTY required)
lambda_rlm -p ./src -q "Find and fix security vulnerabilities" --claude --interactive

# Interactive OpenCode session (TTY required)
lambda_rlm -p ./src -q "Find and fix security vulnerabilities" --opencode --interactive
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

Headless mode (default) runs fully autonomous and writes logs to `.lambda-rlm-claude-{n}.log` or `.lambda-rlm-opencode-{n}.log` in the target directory. Interactive mode inherits your terminal, marks the session active immediately after spawn to avoid TTY startup deadlocks, races child-exit/shutdown/timeout fairly, enforces a 30-minute session safety timeout to prevent stuck children, and requires an attached TTY.

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
| `-q, --question, --prompt` | required | Question or task prompt description |
| `-t, --task` | `auto` | search, classify, aggregate, pairwise, summarise, multi-hop |
| `-w, --window` | `6000` | Leaf chunk size (chars) |
| `-k` | `0` (auto) | Split factor per recursion level |
| `--concurrency` | `8` | Max parallel LLM calls |
| `--model` | `kimi-k2p5-turbo` | Fireworks model ID |
| `--max-tokens` | `8192` | Max output tokens |
| `--claude` | `false` | Enable fix loop with Claude Code |
| `--opencode` | `false` | Enable fix loop with OpenCode |
| `--interactive` | `false` | Run generator in interactive TTY mode |
| `--interactive-timeout` | `15` | Interactive startup timeout in seconds (5..=30) |
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

Recent releases:

- v3.34 (`c3af55c`): fixed the interactive `.. state` freeze by removing interactive child process-group isolation (keeps the TUI in the foreground terminal group), eliminated the startup-readiness gate that could deadlock on TTY sessions, and kept timeout/shutdown watchdog enforcement so runaway sessions still terminate safely.
- v3.33 (`697e9ff`): fixed interactive-mode hang paths by adding a watchdog RAII stop guard that always signals shutdown on every return/abort path, added regression tests proving interactive children are reaped on both clean exit and parent task abort, added leak telemetry for overlong interactive session permits, and added periodic single-flight `DashMap::shrink_to_fit()` during maintenance sweeps to curb long-run memory retention.
- v3.32 (`60196f1`): hardened interactive startup by adding a real readiness probe (child must stay alive before session activation), tightened startup timeout bounds to 5-30 seconds for faster failure on stuck launches, switched shutdown-path interactive termination to SIGTERM-first escalation, and made result-file atomic writes durably fsync both file and parent directory before interactive spawn.
- v3.31 (`e155be2`): added OpenCode-specific 5-minute global interactive timeout wrapping to prevent stuck sessions from hanging the fix loop indefinitely, switched interactive timeout termination to SIGTERM-with-SIGKILL escalation for process-group cleanup (including a regression test that proves TERM then forced kill), and hardened guard/resource cleanup with panic-safe inflight map drops, bounded JoinSet return draining, and preallocated `parse_items` output buffers.
- v3.30 (`2a82930`): made `ChildCleanup` drop wait up to 5s for confirmed background reaping so interactive OpenCode/Claude exits cannot orphan lingering children during runtime shutdown, tightened Phi abort-drain timeout to 500ms for faster cancellation cleanup, added explicit runtime telemetry for codegen/interactive live guards, and reduced overlap-split allocations with pre-sized chunk builders.
- v3.29 (`a7b1be4`): restored Unix interactive child process-group isolation for Claude/OpenCode sessions, force-kill/reap now targets the full process group so orphaned descendants cannot hang fix-loop shutdown, added a hard interactive deadline (`session timeout + 30s`) plus non-TTY revalidation before spawn, and added regression coverage for process-group cleanup on timeout.
- v3.28 (`608172c`): added explicit interactive session live-permit leak tracking with shutdown enforcement, added periodic codegen single-flight maintenance in the global cache/maintenance loop to evict stale flights even when idle, and tightened child cleanup non-blocking regression coverage to assert sub-10ms drop latency under long-running children.
- v3.27 (`de62103`): added 30s timeout guards around interactive session admission and phi/oracle/codegen semaphore acquisition to fail fast instead of hanging under permit starvation, moved codegen budget reservation ahead of bulkhead acquisition, and finalized codegen budget consumption on all completed headless runs to prevent budget leaks under error exits.
- v3.26 (`d96a6e7`): made SIGINT handling interactive-session-aware so Ctrl-C is delivered to foreground OpenCode/Claude TTY children without prematurely shutting down the parent loop, added explicit interactive-session activity tracking tied to the permit lifecycle, and hardened `ChildCleanup` drop to skip background reaping when the child is already exited.
- v3.25 (`abb02b5`): bypassed codegen single-flight coalescing for interactive sessions so duplicate interactive calls no longer await each other, removed Unix process-group isolation in interactive generator launches to preserve direct TTY signal flow, removed biased select ordering in interactive process waits, hardened Phi JoinSet return-drop by aborting/draining before pool return, and extended shutdown guard-drain windows to 30s in interactive mode.
- v3.24 (`0c2841a`): initialized the interactive session semaphore at startup, wrapped interactive admission in an explicit panic-safe RAII permit guard with coverage proving single-session exclusivity/release, and added periodic `inflight` `DashMap::shrink_to_fit()` maintenance every 1000 Oracle calls to reduce long-run map fragmentation.
- v3.23 (`9b0d2c1`): switched interactive fix-loop admission to `acquire_owned()` so semaphore permits are reliably released with owned lifetimes, wrapped full interactive codegen execution in a hard session timeout guard, tracked live codegen child PIDs for explicit shutdown cleanup, and added forced child termination on leak-drain timeout before final shutdown failure.
- v3.22 (`5dfc586`): hardened interactive child cleanup with runtime-shutdown-safe background reaping, added pressure-based single-flight eviction and `/tmp` filesystem validation for codegen work directories, bounded JoinSet pool reuse with shrink-on-pressure plus batched semaphore acquisition in Phi, added budget-aware replanning plus timed leak-drain checks, and tightened cache/telemetry resilience with shared read locks and stderr flush guarantees.
- v3.21 (`714fab8`): added a 30s timeout for codegen single-flight subscribers so orphaned leaders cannot hang interactive runs indefinitely, force-evicted stale flights before waiting to favor liveness over coalescing, validated codegen work directories with no-follow symlink checks, moved drop-time child reaping onto async/background cleanup paths, and switched Phi JoinSet pooling to RAII return guards with extra leak assertions at shutdown.
- v3.20 (`962b34d`): bounded interactive child kill/reap waits to avoid indefinite runtime stalls, made fallback `ChildCleanup` drop reaping force-kill and wait after a 5s deadline to prevent zombie leaks, and added regression coverage for non-blocking drop plus fail-fast interactive preflight when no TTY is attached.
- v3.19 (`7feda6a`): hardened interactive OpenCode/Claude process isolation by putting interactive children in their own process group, added explicit telemetry counters for codegen budget/flight/child-cleanup guards with shutdown leak enforcement, degraded root-level Phi child budget exhaustion instead of failing the whole tree, and added regression tests for sub-100ms interactive shutdown plus disarm/drop race modeling.
- v3.18 (`81e63a2`): removed interactive startup probing so OpenCode/Claude TTY sessions no longer sit in a preflight wait state, prioritized shutdown in interactive wait selection, split circuit breakers by generator+mode, and hardened interactive concurrency/leak boundaries with a single-session semaphore plus per-iteration live-guard checks.
- v3.17 (`230a307`): added a bounded interactive session timeout to kill/reap hung Claude/OpenCode TTY children, kept startup probing behavior intact, added a regression test for session-timeout termination, and moved full source collection into `spawn_blocking` to avoid blocking the async runtime on large repository scans.
- v3.16 (`108b289`): made interactive Claude/OpenCode sessions shutdown-aware by wiring the fix-loop shutdown channel into interactive child waits, racing process exit vs shutdown to avoid stuck sessions on SIGTERM/SIGINT, clamping interactive startup timeouts defensively (1..=120 minutes), and adding tests that prove shutdown-triggered child kill/reap behavior.
- v3.15 (`a065094`): fixed interactive TTY hangs by keeping interactive generators in the foreground process group, split `--interactive-timeout` to startup probing only (no session hard-kill), and added interactive process tests for startup-timeout semantics and early non-success exits.
- v3.14 (`ea8b71c`): removed codegen bulkhead/budget retention from interactive generator sessions so TTY-driven OpenCode/Claude runs no longer starve fix-loop concurrency, tightened interactive error classification (`codegen_retryable` timeout/wait and `codegen_fatal` spawn), and added JoinSet emptiness assertions after abort-drain/pool return.
- v3.13 (`bedafe5`): removed interactive `spawn_blocking` polling in favor of direct Tokio child waits with async timeouts/reaping, and updated timeout coverage to exercise the interactive path without blocking-pool starvation.
- v3.12 (`c17d7a7`): moved interactive Claude/OpenCode execution onto a blocking process path with explicit timeout polling and forced reap on timeout, eliminating async/TTY hangs where interactive sessions stalled after launch.
- v3.11 (`032b6cc`): added periodic replay-cache maintenance during long-running fix loops, exposed Oracle cache sweep hooks, and added a test proving `ChildCleanup` reaps processes even when dropped outside a Tokio runtime.
- v3.10 (`d860423`): enforced shutdown leak checks for Oracle inflight guards, added panic-safe drop hardening and `#[must_use]` coverage for runtime guard types, and tightened shutdown responsiveness before expensive Phi reduction.
- v3.9 (`f47b483`): added `--interactive` codegen mode for Claude/OpenCode with TTY preflight checks, hard interactive timeouts, and mode-aware single-flight/cache keys.
- v3.8 (`a3a007e`): ENOSPC fallback for `.lambda-rlm-result.md`, atomic/symlink-safe codegen logs, shutdown leak enforcement, and expanded secret redaction.
- v3.7 (`f58f5b4`): codegen process hardening with bounded execution, safer child cleanup, and stricter output validation.
- v3.6 (`84b9bb5`): added `--opencode` fix-loop support and shared codegen dispatch.
- v3.5 (`0c0c42f`): removed stale resilience docstrings.
- v3.4 (`40c0188`): removed runtime env-var overrides and simplified circuit-breaker rejuvenation.

See [CHANGELOG.md](CHANGELOG.md) for full release notes and hardening history.

## License

MIT
