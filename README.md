# lambda-RLM

Recursive map-reduce runtime for LLMs. Implements the framework from [arXiv:2603.20105](https://arxiv.org/abs/2603.20105) (Roy et al., 2026).

Solves context-rot: instead of feeding an entire codebase into one LLM call (where accuracy degrades past ~8K tokens), lambda-RLM recursively chunks, processes leaves, and reduces results back up — like MapReduce for reasoning.

## How it works

```
Input (100K chars)
  |
  Chunk (structural split at fn/class boundaries)
  |
  ├── Leaf (6K) → LLM → Verify → result
  ├── Leaf (6K) → LLM → Verify → result
  ├── Leaf (6K) → LLM → Verify → result
  └── ...
  |
  Reduce (task-specific merge)
  |
Output (verified, grounded answer)
```

**Six task types**, each with specialized leaf prompts and reduce operators:

| Task | Reduce | Use case |
|------|--------|----------|
| `search` | Symbolic concat | Find specific code |
| `classify` | Symbolic concat | Label/categorize modules |
| `aggregate` | Dedup merge | Extract items across files |
| `pairwise` | Cross-product + LLM | Find duplicates, compare |
| `summarise` | LLM synthesis | Explain a codebase |
| `multi-hop` | LLM chain reasoning | Trace data flows |

Auto-detects the task type if you don't specify one.

## Install

```bash
git clone https://github.com/anthropics/lambda-rlm.git
cd lambda-rlm
./install.sh
```

Requires Rust toolchain. Installs to `~/.cargo/bin/lambda_rlm`.

## Usage

```bash
export FIREWORKS_API="your-api-key"

# Search
lambda_rlm -p ./src -q "Where is authentication handled?" -t search

# Aggregate
lambda_rlm -p ./src -q "Find all API endpoints" -t aggregate

# Summarize
lambda_rlm -p ./src -q "Summarize this codebase"

# Multi-hop reasoning
lambda_rlm -p ./src -q "Trace a request from handler to database" -t multi-hop

# Auto-detect task type
lambda_rlm -p ./src -q "Any question here"

# Dry run (no API calls)
lambda_rlm -p ./src -q "test" --dry-run
```

## Claude loop mode

Pass `--claude` to enter an autonomous fix loop. Lambda-RLM analyzes your codebase, hands findings to Claude Code, Claude fixes the issues, then lambda-RLM re-analyzes. Repeats until clean.

```bash
lambda_rlm -p ./src -q "Find and fix security vulnerabilities" --claude
```

Each iteration:
1. lambda-RLM analyzes the codebase
2. Claude reads findings, applies fixes, commits
3. lambda-RLM re-analyzes the (now modified) code
4. Loop until clean or `--max-iterations` reached

```bash
# Custom iteration limit
lambda_rlm -p ./src -q "Fix all bugs" --claude --max-iterations 5

# Unlimited iterations
lambda_rlm -p ./src -q "Fix all bugs" --claude --max-iterations 0
```

Requires [Claude Code](https://claude.ai/claude-code) CLI installed (`claude` on PATH).

## Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `-p, --path` | required | File or directory to analyze |
| `-q, --question` | required | Question or task |
| `-t, --task` | `auto` | Task type (search, classify, aggregate, pairwise, summarise, multi-hop) |
| `-w, --window` | `6000` | Leaf chunk size in chars |
| `-k` | `0` (auto) | Split factor |
| `--concurrency` | `8` | Max parallel LLM calls |
| `--model` | `kimi-k2p5-turbo` | Fireworks model ID |
| `--max-tokens` | `8192` | Max output tokens |
| `--quorum` | `false` | 3x consensus voting for critical paths |
| `--claude` | `false` | Enable Claude fix loop |
| `--max-iterations` | `10` | Max fix loop iterations (0 = unlimited) |
| `--dry-run` | `false` | No API calls |
| `--no-cache` | `false` | Disable replay cache |

## Architecture

```
src/
  main.rs        — CLI, file collector, claude loop
  types.rs       — TaskType enum
  combinator.rs  — chunking, keyword extraction, structural split
  oracle.rs      — Fireworks API client, semaphore, retries, cache
  resilience.rs  — circuit breaker, budget guard (RAII), replay cache
  cost.rs        — cost model, optimal planner (Theorem 4)
  verify.rs      — leaf output validation (refusal detection, task-specific checks)
  reduce.rs      — task-specific reduce operators
  phi.rs         — recursive executor (Algorithm 2)
```

Hardening: circuit breaker (3 failures / 30s cooloff), RAII budget guards, replay cache (blake3 content-addressed), structural chunking at definition boundaries, leaf verification, trigram-based quorum consensus.

## Environment

| Variable | Required | Description |
|----------|----------|-------------|
| `FIREWORKS_API` | Yes (unless `--dry-run`) | Fireworks AI API key |
| `RUST_LOG` | No | Logging level (`warn` default, `info`, `debug`) |

## License

MIT
