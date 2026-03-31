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

### Claude Code CLI (optional)

The `--claude` flag enables an autonomous fix loop where lambda-RLM analyzes your code, hands findings to Claude Code, Claude fixes the issues, and lambda-RLM re-analyzes until clean. This requires Claude Code installed:

```bash
npm install -g @anthropic-ai/claude-code
```

This uses your Anthropic API key / Claude subscription separately from Fireworks. The two systems work together: Fireworks does the cheap, parallelized analysis; Claude does the expensive, targeted code editing.

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

## Claude fix loop

This is the main thing. Point lambda-RLM at your code with a question, add `--claude`, and walk away:

```bash
lambda_rlm -p ./src -q "Find and fix security vulnerabilities" --claude
```

What happens:

```
 Iteration 1:
   lambda-RLM analyzes codebase (Fireworks, parallelized)
     → finds 5 issues
   Claude Code reads findings, edits files, commits
     → fixes 5 issues

 Iteration 2:
   lambda-RLM re-analyzes (fresh scan of modified code)
     → finds 1 remaining issue
   Claude Code fixes it, commits

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

Claude runs with `--dangerously-skip-permissions` in `-p` (print) mode — fully autonomous, no prompts. Logs for each iteration are saved to `.lambda-rlm-claude-{n}.log` in the target directory.

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
| `--claude` | `false` | Enable autonomous fix loop |
| `--max-iterations` | `10` | Fix loop iterations (0 = unlimited) |
| `--dry-run` | `false` | No API calls |
| `--no-cache` | `false` | Disable replay cache |
| `--timeout` | `120` | Per-call timeout (seconds) |
| `--max-calls` | `0` | Max total LLM calls (0 = unlimited) |
| `--max-retries` | `2` | Retries per failed call |

## Architecture

```
src/
  main.rs        — CLI, file collector, claude loop driver
  types.rs       — TaskType enum (shared across modules)
  combinator.rs  — structural chunking, keyword extraction, text splitting
  oracle.rs      — Fireworks API client, semaphore, retries, streaming
  resilience.rs  — circuit breaker, RAII budget guard, replay cache (blake3)
  cost.rs        — cost model, optimal plan computation (Theorem 4)
  verify.rs      — leaf output validation, refusal detection
  reduce.rs      — task-specific reduce operators (symbolic + neural)
  phi.rs         — recursive executor (Algorithm 2)
```

Hardening: circuit breaker (3 failures / 30s cooloff), RAII budget guards (panic-safe via Drop, production leak detection), blake3 content-addressed replay cache, structural chunking at definition boundaries, leaf verification with refusal detection, CAS backoff under contention, EXDEV-safe state persistence.

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `FIREWORKS_API` | Yes | Fireworks AI API key ([get one free](https://fireworks.ai)) |
| `RUST_LOG` | No | Log level: `warn` (default), `info`, `debug` |

## Changelog

### v3.1 — Resilience & Performance Hardening

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

## License

MIT
