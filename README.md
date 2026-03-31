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

See [CHANGELOG.md](CHANGELOG.md).

## License

MIT
