# Architecture Decisions

This document records decisions in lambda-RLM where we intentionally traded one property for another to maximize production resilience, performance, and debuggability.

| Decision | Sacrifice | Gain |
|---|---|---|
| `Arc<str>` over `String` in high-fanout paths | Immutable buffers require explicit re-allocation on mutation | Lower peak heap usage from shared immutable slices and less allocator churn in recursive trees |
| Manual stack DFS in `collect_source_files` over `walkdir` | More in-house traversal code to maintain | Tighter control over symlink policy and fewer supply-chain/runtime dependencies in the hot file-ingest path |
| `blake3` for replay/cache keys over `sha2` | Less universal familiarity than SHA-256 | Faster content-addressed hashing and lower CPU cost in repeated cache key generation |
| Loom tests for lock-free primitives | Slower CI for concurrency-model checks | Deterministic interleaving coverage for race-prone guards/circuit behavior that normal tests miss |
| Deterministic jitter keyed by content hash | Slightly more hashing overhead than plain RNG | Reproducible backoff behavior across retries, reducing burst amplification and making failures easier to replay |
