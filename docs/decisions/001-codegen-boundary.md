## ADR 001: External CLI Code Generation Boundary

### Context
After lambda-RLM produces `.lambda-rlm-result.md`, remediation is delegated to external code-generator CLIs (`claude`, `opencode`). These tools are separate processes with independent release cycles and failure modes (timeouts, crashes, non-zero exits, malformed output, or local environment drift).

### Decision
Use file-based IPC with atomic writes and isolated subprocess execution.

- Write `.lambda-rlm-result.md` through a temp file + atomic persist.
- Execute generators via `tokio::process::Command` with null stdin, piped stdout/stderr, `kill_on_drop(true)`, and process-group isolation on Unix.
- Enforce strict output contracts (non-empty, UTF-8, max size, no NUL bytes).
- Maintain per-generator circuit breakers and telemetry to avoid cross-generator blast radius.
- Return sanitized external errors to callers while preserving detailed internal logs.

### Consequences
Positive:
- Generator crashes cannot directly corrupt in-process runtime state.
- Atomic file writes avoid partial-result reads and reduce TOCTOU exposure.
- Per-generator circuits prevent one flaky backend from suppressing the other.
- Logs and iteration artifacts remain auditable for post-incident analysis.

Trade-offs:
- Process spawn and file I/O add latency compared to in-process APIs.
- More orchestration code (timeouts, cleanup, validation, log rotation).

### Alternatives Rejected
- Direct library/RPC integration into lambda-RLM process: tighter coupling, larger trust boundary, weaker failure isolation.
- Shared single circuit breaker: simpler state, but unacceptable correlated failures between generators.

### Why This Is Acceptable
The fix loop is batch-oriented and correctness/safety-sensitive. The bounded overhead from process isolation is acceptable relative to reduced outage and leak risk in agent-hosted production deployments.
