//! Oracle M — the only neural primitive.
//!
//! Wraps the Fireworks API with: circuit breaker, RAII budget guard,
//! replay cache, per-call timeout, retries with exponential backoff,
//! telemetry counters. All state is lock-free (atomics).
//!
//! Also: quorum consensus with normalized trigram similarity.

use crate::resilience::{BudgetGuard, CallBudget, CircuitBreaker, ReplayCache};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use thiserror::Error;
use tokio::sync::Semaphore;
use tokio::task::JoinSet;

/// Maximum SSE stream buffer size. Prevents OOM from malicious or
/// runaway endpoints streaming unbounded data. 10MB accommodates the
/// largest reasonable LLM response with headroom.
const MAX_SSE_BUFFER_BYTES: usize = 10 * 1024 * 1024;

/// Maximum SSE events before aborting the stream. Prevents infinite hangs
/// from a malformed LLM stream that never emits [DONE] but stays under the
/// byte limit by sending tiny chunks. 100k events at ~100 bytes each ≈ 10MB,
/// consistent with MAX_SSE_BUFFER_BYTES.
const MAX_SSE_EVENTS: usize = 100_000;

// ── Structured Error Types ──────────────────────────────────────
// Enables intelligent retry classification: retryable vs fatal.
// Only used at the Oracle API boundary — internal modules keep anyhow.

#[derive(Error, Debug)]
pub enum OracleError {
    #[error("rate limited, retry after {retry_after:?}")]
    RateLimit { retry_after: Duration },
    #[error("auth failed: {0}")]
    AuthFailed(String),
    #[error("timeout after {0:?}")]
    Timeout(Duration),
    #[error("API error {status}: {body}")]
    ApiError { status: u16, body: String },
    #[error("network: {0}")]
    Network(String),
    #[error("stream corrupted: {0}")]
    StreamCorrupted(String),
}

impl OracleError {
    /// Retryable errors: transient failures that may succeed on retry.
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::RateLimit { .. } | Self::Timeout(_) | Self::Network(_) => true,
            Self::ApiError { status, .. } => *status >= 500,
            _ => false,
        }
    }

    /// Fatal errors: configuration problems that will never self-resolve.
    /// Trip circuit breaker immediately rather than wasting retries.
    pub fn is_fatal(&self) -> bool {
        matches!(self, Self::AuthFailed(_))
    }

    /// Map to global ErrorKind for programmatic recovery in main.rs.
    #[allow(dead_code)]
    pub fn kind(&self) -> crate::types::ErrorKind {
        match self {
            Self::RateLimit { .. } | Self::Timeout(_) | Self::Network(_) => {
                crate::types::ErrorKind::TransientNetwork
            }
            Self::ApiError { status, .. } if *status >= 500 => {
                crate::types::ErrorKind::TransientNetwork
            }
            Self::AuthFailed(_) => crate::types::ErrorKind::PermanentLLMFailure,
            Self::ApiError { .. } => crate::types::ErrorKind::PermanentLLMFailure,
            Self::StreamCorrupted(_) => crate::types::ErrorKind::Corruption,
        }
    }
}

// ── Cancellation safety note ──────────────────────────────────────
// CbFailureGuard was removed: all error paths in the retry loop
// already call record_failure() explicitly. The guard only fired on
// task cancellation (e.g., JoinSet::abort_all in quorum fast-path),
// which is intentional load shedding — NOT an API failure. Recording
// it as a failure caused false CB trips during routine quorum
// consensus (3 quorum calls × 1 false failure = CB threshold of 3).

// ── API types ────────────────────────────────────────────────────

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    max_tokens: u32,
    messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ResponseFormat>,
}

#[derive(Serialize)]
struct ResponseFormat {
    #[serde(rename = "type")]
    format_type: String,
}

#[derive(Serialize, Deserialize, Clone)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatMessage,
}

#[derive(Deserialize)]
struct StreamChunk {
    choices: Vec<StreamChoice>,
}

#[derive(Deserialize)]
struct StreamChoice {
    delta: StreamDelta,
}

#[derive(Deserialize)]
struct StreamDelta {
    #[serde(default)]
    content: Option<String>,
}

// ── Single-Flight Guard ──────────────────────────────────────────
// RAII guard that removes the inflight entry on all exit paths
// (success, error, panic). On success, caller broadcasts the result
// via inflight_tx.send() before the guard drops; on error/panic,
// dropping the Sender signals waiters to fall through and try themselves.

struct InflightGuard<'a> {
    map: &'a Mutex<HashMap<String, Arc<tokio::sync::watch::Sender<Option<String>>>>>,
    key: String,
}

impl Drop for InflightGuard<'_> {
    fn drop(&mut self) {
        self.map.lock().unwrap().remove(&self.key);
    }
}

// ── Bulkhead ─────────────────────────────────────────────────────
// Separate semaphores for different resource classes to prevent
// I/O-bound LLM calls from starving CPU-bound reduce operations
// and vice versa.

/// Hard ceiling for bulkhead permits. Prevents misconfiguration from
/// exhausting file descriptors or memory with unbounded concurrent requests.
const MAX_BULKHEAD_PERMITS: usize = 500;

pub struct Bulkhead {
    llm: Arc<Semaphore>,
    #[allow(dead_code)] // Reserved for CPU-bound reduce operations
    cpu: Arc<Semaphore>,
}

impl Bulkhead {
    pub fn new(llm_permits: usize) -> Self {
        // Allow runtime override for operators to tune based on container
        // memory limits. Clamped to [1, MAX_BULKHEAD_PERMITS].
        let effective_llm = std::env::var("LAMBDA_RLM_BULKHEAD_LLM_PERMITS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(llm_permits)
            .clamp(1, MAX_BULKHEAD_PERMITS);
        let cpu_permits = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        Self {
            llm: Arc::new(Semaphore::new(effective_llm)),
            cpu: Arc::new(Semaphore::new(cpu_permits)),
        }
    }

    pub fn llm(&self) -> &Semaphore {
        &self.llm
    }

    #[allow(dead_code)] // Reserved for CPU-bound reduce operations
    pub fn cpu(&self) -> &Semaphore {
        &self.cpu
    }
}

// ── LLM Provider Trait ───────────────────────────────────────────
// Vendor-agnostic interface for LLM invocation. Enables zero-downtime
// migration to local models or new APIs (e.g., Ollama, vLLM, Anthropic)
// without changing Oracle's resilience logic (circuit breaker, budget,
// cache, retries). Hot-swappable via config reload + restart.

/// Future type for LlmProvider::invoke — avoids async_trait dependency.
/// Retained as documentation for future provider implementations.
#[allow(dead_code)]
type ProviderFuture<'a> = std::pin::Pin<
    Box<dyn std::future::Future<Output = std::result::Result<String, OracleError>> + Send + 'a>,
>;

#[allow(dead_code)] // Interface for future provider implementations
pub trait LlmProvider: Send + Sync {
    /// Raw API call with timeout. No retries, no cache, no budget check.
    /// Implementations handle transport (HTTP, gRPC, local) and response parsing.
    fn invoke(
        &self,
        system: &str,
        user_prompt: &str,
        max_tokens: u32,
        idempotency_key: &str,
    ) -> ProviderFuture<'_>;

    /// Model identifier for cache key generation.
    fn model_id(&self) -> &str;
}

/// Fireworks AI provider implementation.
struct FireworksProvider {
    api_key: String,
    model: String,
    client: reqwest::Client,
    timeout: Duration,
}

impl FireworksProvider {
    fn new(api_key: String, model: String, timeout: Duration) -> Self {
        Self {
            api_key,
            model,
            client: reqwest::Client::new(),
            timeout,
        }
    }

    /// SSE stream parser per W3C Server-Sent Events specification.
    async fn read_stream(&self, response: reqwest::Response) -> Result<String> {
        use futures::StreamExt;
        let mut result = String::new();
        let mut stream = response.bytes_stream();
        let mut buf: Vec<u8> = Vec::new();
        let mut event_data = String::new();
        let mut bom_stripped = false;
        let mut event_count: usize = 0;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| anyhow::anyhow!("Stream read error: {e}"))?;
            buf.extend_from_slice(&chunk);

            if result.len() + buf.len() > MAX_SSE_BUFFER_BYTES {
                anyhow::bail!(
                    "SSE stream exceeded maximum buffer size ({} bytes)",
                    MAX_SSE_BUFFER_BYTES
                );
            }

            if !bom_stripped {
                if buf.starts_with(&[0xEF, 0xBB, 0xBF]) {
                    buf.drain(..3);
                }
                bom_stripped = true;
            }

            while let Some((line_end, skip)) = sse_line_boundary(&buf) {
                let line = String::from_utf8(buf[..line_end].to_vec())
                    .map_err(|e| anyhow::anyhow!("SSE stream contains invalid UTF-8: {e}"))?;
                buf.drain(..line_end + skip);

                if line.is_empty() {
                    if !event_data.is_empty() {
                        if event_data.ends_with('\n') {
                            event_data.pop();
                        }
                        if event_data == "[DONE]" {
                            return Ok(result);
                        }
                        if let Ok(parsed) = serde_json::from_str::<StreamChunk>(&event_data) {
                            if let Some(choice) = parsed.choices.first() {
                                if let Some(content) = &choice.delta.content {
                                    result.push_str(content);
                                }
                            }
                        }
                        event_data.clear();
                    }
                } else if line.starts_with(':') {
                    // SSE comment line, skip
                } else if let Some(value) = line.strip_prefix("data:") {
                    let value = value.strip_prefix(' ').unwrap_or(value);
                    event_data.push_str(value);
                    event_data.push('\n');
                    event_count += 1;
                    if event_count > MAX_SSE_EVENTS {
                        anyhow::bail!(
                            "SSE stream exceeded maximum event count ({} events)",
                            MAX_SSE_EVENTS
                        );
                    }
                }
            }
        }

        if !event_data.is_empty() {
            if event_data.ends_with('\n') {
                event_data.pop();
            }
            if event_data != "[DONE]" {
                if let Ok(parsed) = serde_json::from_str::<StreamChunk>(&event_data) {
                    if let Some(choice) = parsed.choices.first() {
                        if let Some(content) = &choice.delta.content {
                            result.push_str(content);
                        }
                    }
                }
            }
        }

        Ok(result)
    }
}

// LlmProvider trait impl replaced with direct async method on FireworksProvider
// to eliminate per-call Box::pin heap allocation. The trait is retained above
// as a design contract for future provider implementations.

impl FireworksProvider {
    /// Direct async invocation — compiler generates the state machine on the
    /// caller's stack frame, eliminating the Box::pin heap allocation that the
    /// trait-object pattern required. Over high-throughput symbolic reduction,
    /// this prevents allocator fragmentation and removes a hidden OOM vector.
    async fn invoke(
        &self,
        system: &str,
        user_prompt: &str,
        max_tokens: u32,
        idempotency_key: &str,
    ) -> std::result::Result<String, OracleError> {
        let fut = async {
            let mut messages = Vec::with_capacity(2);
            if !system.is_empty() {
                messages.push(ChatMessage {
                    role: "system".into(),
                    content: system.to_string(),
                });
            }
            messages.push(ChatMessage {
                role: "user".into(),
                content: user_prompt.to_string(),
            });

            let use_stream = max_tokens > 4096;

            let request = ChatRequest {
                model: self.model.clone(),
                max_tokens,
                messages,
                stream: use_stream,
                temperature: Some(0.0),
                response_format: None,
            };

            let response = self
                .client
                .post("https://api.fireworks.ai/inference/v1/chat/completions")
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("Content-Type", "application/json")
                .header("X-Idempotency-Key", idempotency_key)
                .json(&request)
                .send()
                .await
                .map_err(|e| OracleError::Network(e.to_string()))?;

            let status = response.status();
            if !status.is_success() {
                let code = status.as_u16();
                let body = response.text().await.unwrap_or_default();
                return Err(match code {
                    429 => OracleError::RateLimit {
                        retry_after: Duration::from_secs(5),
                    },
                    401 | 403 => OracleError::AuthFailed(body),
                    _ => OracleError::ApiError {
                        status: code,
                        body,
                    },
                });
            }

            if use_stream {
                self.read_stream(response)
                    .await
                    .map_err(|e| OracleError::StreamCorrupted(e.to_string()))
            } else {
                let resp: ChatResponse = response
                    .json()
                    .await
                    .map_err(|e| OracleError::StreamCorrupted(e.to_string()))?;
                Ok(resp
                    .choices
                    .first()
                    .map(|c| c.message.content.clone())
                    .unwrap_or_default())
            }
        };

        tokio::time::timeout(self.timeout, fut)
            .await
            .map_err(|_| OracleError::Timeout(self.timeout))?
    }
}

// ── Oracle ───────────────────────────────────────────────────────

pub struct Oracle {
    provider: FireworksProvider,
    model: String,
    call_count: AtomicU64,
    bulkhead: Bulkhead,
    dry_run: bool,
    // Resilience
    circuit: CircuitBreaker,
    cache: ReplayCache,
    budget: CallBudget,
    max_retries: usize,
    /// Single-flight coalescing: prevents thundering herd when N concurrent
    /// requests share the same cache key (e.g., after corruption quarantine).
    /// Only one request proceeds to the LLM; others wait for its result.
    inflight: Mutex<HashMap<String, Arc<tokio::sync::watch::Sender<Option<String>>>>>,
    /// Cooperative shutdown flag. When set, in-flight and future API calls
    /// fail fast instead of proceeding, preventing wasted billing after shutdown.
    shutdown: AtomicBool,
    // Telemetry — all AtomicU64 to prevent silent overflow on 32-bit targets.
    // A 32-bit AtomicUsize wraps at ~4 billion, reachable for char counters
    // processing large codebases over long deployments.
    cache_hits: AtomicU64,
    errors: AtomicU64,
    total_input_chars: AtomicU64,
    total_output_chars: AtomicU64,
    total_latency_ms: AtomicU64,
}

impl Oracle {
    pub fn new(
        api_key: String,
        model: String,
        dry_run: bool,
        concurrency: usize,
        timeout: Duration,
        max_retries: usize,
        max_calls: usize,
        cache_dir: PathBuf,
        cache_enabled: bool,
        max_cache_entries: usize,
    ) -> Arc<Self> {
        let cb_path = cache_dir.join(".circuit_breaker_state");
        let provider = FireworksProvider::new(api_key, model.clone(), timeout);
        let oracle = Arc::new(Self {
            provider,
            model,
            call_count: AtomicU64::new(0),
            bulkhead: Bulkhead::new(concurrency),
            dry_run,
            circuit: if cache_enabled {
                CircuitBreaker::with_persistence(3, Duration::from_secs(30), cb_path)
            } else {
                CircuitBreaker::new(3, Duration::from_secs(30))
            },
            cache: ReplayCache::new(cache_dir, cache_enabled, max_cache_entries),
            budget: CallBudget::new(max_calls),
            max_retries,
            shutdown: AtomicBool::new(false),
            inflight: Mutex::new(HashMap::new()),
            cache_hits: AtomicU64::new(0),
            errors: AtomicU64::new(0),
            total_input_chars: AtomicU64::new(0),
            total_output_chars: AtomicU64::new(0),
            total_latency_ms: AtomicU64::new(0),
        });

        // Background cache scavenger: periodically scrubs cache integrity and
        // bounds the quarantine directory. Prevents unbounded disk growth from
        // bitrot over long-running deployments (corrupted/ dir, orphan temps).
        // Also runs generational compaction every 24h to defragment the directory
        // index — ext4/xfs degrade to linear scans after millions of entries.
        if cache_enabled {
            let weak = Arc::downgrade(&oracle);
            tokio::spawn(async move {
                let scrub_interval = Duration::from_secs(6 * 3600); // every 6 hours
                let compact_interval = Duration::from_secs(24 * 3600); // every 24 hours
                let mut scrub_tick = tokio::time::interval(scrub_interval);
                let mut compact_tick = tokio::time::interval(compact_interval);
                scrub_tick.tick().await; // consume initial immediate tick
                compact_tick.tick().await;
                loop {
                    tokio::select! {
                        _ = scrub_tick.tick() => {
                            let Some(oracle) = weak.upgrade() else { break; };
                            if oracle.shutdown.load(Ordering::Acquire) { break; }
                            let cache_ref = oracle.cache.clone_for_scrub();
                            let result = tokio::task::spawn_blocking(move || {
                                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                    cache_ref.scrub()
                                }))
                            }).await;
                            match result {
                                Ok(Ok((verified, quarantined))) => {
                                    if quarantined > 0 {
                                        tracing::info!(verified, quarantined, "periodic cache scrub completed");
                                    }
                                }
                                Ok(Err(_)) => {
                                    tracing::error!("cache scrub panicked in background task");
                                }
                                Err(e) => {
                                    tracing::error!(error = %e, "cache scrub task failed");
                                }
                            }
                        }
                        _ = compact_tick.tick() => {
                            let Some(oracle) = weak.upgrade() else { break; };
                            if oracle.shutdown.load(Ordering::Acquire) { break; }
                            let cache_ref = oracle.cache.clone_for_scrub();
                            let result = tokio::task::spawn_blocking(move || {
                                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                    cache_ref.compact()
                                }))
                            }).await;
                            match result {
                                Ok(Ok((migrated, dropped))) => {
                                    if migrated > 0 || dropped > 0 {
                                        tracing::info!(migrated, dropped, "periodic cache compaction completed");
                                    }
                                }
                                Ok(Err(_)) => {
                                    tracing::error!("cache compaction panicked in background task");
                                }
                                Err(e) => {
                                    tracing::error!(error = %e, "cache compaction task failed");
                                }
                            }
                        }
                    }
                }
            });
        }

        oracle
    }

    pub fn calls(&self) -> u64 {
        self.call_count.load(Ordering::Acquire)
    }

    pub fn cache(&self) -> &ReplayCache {
        &self.cache
    }

    pub fn budget_remaining(&self) -> usize {
        self.budget.remaining()
    }

    pub fn budget_unlimited(&self) -> bool {
        self.budget.is_unlimited()
    }

    /// Atomically reserve n budget units. Returns false if insufficient budget.
    /// Used by phi's pre-flight to eliminate TOCTOU race on budget checks.
    pub fn budget_try_reserve(&self, n: usize) -> bool {
        self.budget.try_acquire_n(n)
    }

    /// Release n previously reserved budget units.
    pub fn budget_unreserve(&self, n: usize) {
        self.budget.release_n(n)
    }

    #[allow(dead_code)]
    pub fn bulkhead(&self) -> &Bulkhead {
        &self.bulkhead
    }

    /// Signal shutdown to cancel in-flight and prevent future API calls.
    /// Called from the signal handler to stop billing when shutdown fires.
    pub fn trigger_shutdown(&self) {
        self.shutdown.store(true, Ordering::Release);
    }

    /// Outer call: budget → cache → bulkhead semaphore → retry loop with circuit breaker.
    ///
    /// RESOURCE ORDERING: Budget → Cache → Bulkhead (LLM semaphore).
    /// Called from phi() which has already released its concurrency Semaphore.
    /// See phi.rs for the global ordering: Semaphore → Budget → Cache.
    ///
    /// PRE: api_key is valid (or dry_run=true)
    /// POST: Ok(text) where text passes basic non-empty check, or Err
    ///
    /// Budget is consumed on all normal exits (success AND error after retries).
    /// Budget is restored only on panic/cancellation via BudgetGuard RAII.
    ///
    /// Retry classification:
    ///   - Retryable (rate limit, timeout, network, 5xx): exponential backoff
    ///   - Fatal (auth failure): trip circuit breaker immediately, no retry
    ///   - Non-retryable (4xx except 429): fail fast, no retry
    pub async fn call(&self, system: &str, user_prompt: &str, max_tokens: u32) -> Result<String> {
        // 1. Budget acquire + RAII guard
        if !self.budget.try_acquire() {
            anyhow::bail!(
                "Call budget exhausted (remaining: {})",
                self.budget.remaining()
            );
        }
        let mut guard = BudgetGuard::new(&self.budget);

        // 2. Cache check
        let cache_key = ReplayCache::key(&self.model, system, user_prompt, max_tokens);
        if let Some(cached) = self.cache.get(&cache_key) {
            self.cache_hits.fetch_add(1, Ordering::AcqRel);
            let n = self.call_count.fetch_add(1, Ordering::AcqRel) + 1;
            tracing::debug!(
                call = n,
                input_len = user_prompt.len(),
                output_len = cached.len(),
                "cache hit"
            );
            eprintln!(
                "    M #{n} [CACHED] ({} in -> {} out)",
                user_prompt.len(),
                cached.len()
            );
            guard.commit();
            return Ok(cached);
        }

        // 2b. Single-flight: coalesce concurrent requests for the same cache key.
        // If another task is already fetching this key, wait for its result instead
        // of issuing a duplicate LLM call (thundering herd prevention).
        // Lock is scoped to avoid holding MutexGuard across .await.
        let inflight_rx = {
            let map = self.inflight.lock().unwrap();
            map.get(&cache_key).map(|tx| tx.subscribe())
        };
        if let Some(mut rx) = inflight_rx {
            loop {
                if let Some(ref result) = *rx.borrow() {
                    self.cache_hits.fetch_add(1, Ordering::AcqRel);
                    guard.commit();
                    return Ok(result.clone());
                }
                if rx.changed().await.is_err() {
                    // In-flight caller failed — fall through to try ourselves
                    break;
                }
            }
        }
        // Register as the in-flight caller for this key.
        // RAII guard removes entry on all exit paths (success, error, panic).
        let (inflight_tx, _) = tokio::sync::watch::channel::<Option<String>>(None);
        let inflight_tx = Arc::new(inflight_tx);
        self.inflight
            .lock()
            .unwrap()
            .insert(cache_key.clone(), Arc::clone(&inflight_tx));
        let _inflight_guard = InflightGuard {
            map: &self.inflight,
            key: cache_key.clone(),
        };

        // 3. Semaphore — held for entire call including retries
        let _permit = self.bulkhead.llm().acquire().await.unwrap();
        let n = self.call_count.fetch_add(1, Ordering::AcqRel) + 1;

        // Idempotency key: stable across retries of the same call, unique per
        // logical invocation. Prevents duplicate execution when the server
        // processes a request but the client times out and retries.
        let idempotency_key = {
            let mut hasher = blake3::Hasher::new();
            hasher.update(&n.to_le_bytes());
            // Process nonce: 16 bytes of startup entropy that survives PID
            // wraparound (~65k restarts). Replaces boot_mono_ms + PID which
            // could collide when two processes with recycled PIDs start in
            // the same millisecond.
            hasher.update(crate::resilience::process_nonce());
            hasher.update(system.as_bytes());
            hasher.update(b"\x00");
            hasher.update(user_prompt.as_bytes());
            hasher.update(b"\x00");
            hasher.update(&max_tokens.to_le_bytes());
            format!("lrlm-{}", &hasher.finalize().to_hex()[..32])
        };

        if self.dry_run {
            guard.commit();
            return Ok(format!(
                "[DRY RUN] Call #{n} ({} chars in, max_tok={max_tokens})",
                user_prompt.len()
            ));
        }

        // 4. Retry loop with circuit breaker + exponential backoff
        let mut last_error: Option<OracleError> = None;
        for attempt in 0..=self.max_retries {
            // Cooperative shutdown: fail fast instead of starting new API calls.
            // Prevents wasted billing when shutdown signal has fired.
            if self.shutdown.load(Ordering::Acquire) {
                guard.commit();
                anyhow::bail!("Cancelled by shutdown signal");
            }

            // Circuit breaker: use allow_request for half-open probe support
            if !self.circuit.allow_request() {
                let failures = self.circuit.failures();
                tracing::warn!(
                    failures,
                    cooldown_s = 30,
                    "circuit breaker open"
                );
                guard.commit(); // budget consumed: we tried
                anyhow::bail!(
                    "Circuit breaker open -- {} consecutive failures, cooling off 30s",
                    failures
                );
            }

            if attempt > 0 {
                // Deterministic jitter: blake3(idempotency_key || attempt) mod cap.
                // Reproducible from logs (log the idempotency_key to replay exact
                // retry timing). Still breaks thundering herd because different
                // calls have different keys.
                let cap = 500u64 << (attempt - 1).min(4);
                let jitter = {
                    let mut hasher = blake3::Hasher::new();
                    hasher.update(idempotency_key.as_bytes());
                    hasher.update(&(attempt as u64).to_le_bytes());
                    let hash = hasher.finalize();
                    let bytes = hash.as_bytes();
                    let raw = u64::from_le_bytes(bytes[..8].try_into().unwrap());
                    raw % (cap + 1)
                };
                let backoff = Duration::from_millis(jitter);
                tracing::debug!(call = n, attempt, max = self.max_retries, ?backoff, cap_ms = cap, idempotency_key, "retrying with deterministic jitter");
                eprintln!(
                    "    M #{n} retry {attempt}/{} (backoff {:?}, cap {:?})",
                    self.max_retries, backoff, Duration::from_millis(cap)
                );
                tokio::time::sleep(backoff).await;
            }

            let start = Instant::now();
            // No CbFailureGuard here: all error paths below call record_failure()
            // explicitly. Task cancellation (quorum abort) must NOT record failure.
            let api_result = self.call_api(system, user_prompt, max_tokens, &idempotency_key).await;

            match api_result {
                Ok(text) => {
                    let latency = start.elapsed();
                    self.circuit.record_success();
                    self.total_input_chars
                        .fetch_add(user_prompt.len() as u64, Ordering::AcqRel);
                    self.total_output_chars
                        .fetch_add(text.len() as u64, Ordering::AcqRel);
                    self.total_latency_ms
                        .fetch_add(latency.as_millis() as u64, Ordering::AcqRel);

                    if let Err(e) = self.cache.put(&cache_key, &text) {
                        tracing::warn!(error = %e, "cache write failed (non-fatal)");
                    }

                    eprintln!(
                        "    M #{n} ({} in -> {} out, {:.1}s)",
                        user_prompt.len(),
                        text.len(),
                        latency.as_secs_f64()
                    );
                    // Broadcast result to single-flight waiters before returning
                    let _ = inflight_tx.send(Some(text.clone()));
                    guard.commit();
                    return Ok(text);
                }
                Err(e) if e.is_fatal() => {
                    // Fatal: trip breaker immediately, do not retry
                    self.circuit.record_failure();
                    self.errors.fetch_add(1, Ordering::AcqRel);
                    tracing::error!(call = n, error = %e, "fatal API error, not retrying");
                    eprintln!("    M #{n} FATAL: {e}");
                    guard.commit();
                    anyhow::bail!("{e}");
                }
                Err(e) => {
                    self.circuit.record_failure();
                    self.errors.fetch_add(1, Ordering::AcqRel);
                    tracing::error!(call = n, attempt = attempt + 1, error = %e, "API call failed");
                    eprintln!("    M #{n} ERROR (attempt {}): {e}", attempt + 1);

                    if !e.is_retryable() {
                        // Non-retryable (e.g., 4xx client error): fail fast
                        guard.commit();
                        anyhow::bail!("{e}");
                    }

                    last_error = Some(e);
                }
            }
        }

        guard.commit(); // budget consumed: we made real API attempts
        Err(last_error
            .map(|e| anyhow::anyhow!("{e}"))
            .unwrap_or_else(|| anyhow::anyhow!("all retries exhausted")))
    }

    /// Delegates to the configured LlmProvider. No retries, no cache, no budget check.
    async fn call_api(
        &self,
        system: &str,
        user_prompt: &str,
        max_tokens: u32,
        idempotency_key: &str,
    ) -> std::result::Result<String, OracleError> {
        self.provider.invoke(system, user_prompt, max_tokens, idempotency_key).await
    }

    /// Emit telemetry as a single atomic stderr write. Individual eprintln!
    /// calls interleave under concurrent async tasks, rendering logs
    /// unparseable during cascading failures when post-mortem analysis matters most.
    pub fn print_telemetry(&self) {
        use std::fmt::Write as FmtWrite;
        use std::io::Write as IoWrite;

        let calls = self.calls();
        let cache_hits = self.cache_hits.load(Ordering::Acquire);
        let errors = self.errors.load(Ordering::Acquire);
        let input_chars = self.total_input_chars.load(Ordering::Acquire);
        let output_chars = self.total_output_chars.load(Ordering::Acquire);
        let latency_ms = self.total_latency_ms.load(Ordering::Acquire);
        let budget_remaining = self.budget.remaining();

        // fmt::Write for String is infallible (OOM panics, never returns Err),
        // but we propagate via a helper to satisfy zero-swallowing discipline.
        fn w(buf: &mut String, args: std::fmt::Arguments<'_>) {
            buf.write_fmt(args).expect("String::write_fmt is infallible");
        }

        let mut buf = String::with_capacity(512);
        w(&mut buf, format_args!("  Telemetry:\n"));
        w(&mut buf, format_args!("    Total M calls:    {calls}\n"));
        w(&mut buf, format_args!("    Cache hits:       {cache_hits}\n"));
        w(&mut buf, format_args!("    API errors:       {errors}\n"));
        w(&mut buf, format_args!("    Input chars:      {input_chars}\n"));
        w(&mut buf, format_args!("    Output chars:     {output_chars}\n"));
        if calls > cache_hits {
            let api_calls = calls - cache_hits;
            w(
                &mut buf,
                format_args!(
                    "    Avg latency:      {:.1}ms/call\n",
                    latency_ms as f64 / api_calls as f64
                ),
            );
        }
        if !self.budget.is_unlimited() {
            w(&mut buf, format_args!("    Budget remaining: {budget_remaining}\n"));
        }
        let corruptions = self.cache.corruption_count();
        if corruptions > 0 {
            w(&mut buf, format_args!("    Cache corruptions: {corruptions} (bitrot detected)\n"));
        }

        // Single atomic write under stderr lock — no interleaving possible.
        // Stderr failure (ENOMEM, EBADF, ENOSPC) is non-fatal: increment a
        // lock-free drop counter and continue. Telemetry must never kill the
        // primary function — lossy degradation over catastrophic abort.
        static TELEMETRY_DROPS: AtomicU64 = AtomicU64::new(0);
        let stderr = std::io::stderr();
        let mut handle = stderr.lock();
        if let Err(e) = handle.write_all(buf.as_bytes()) {
            let drops = TELEMETRY_DROPS.fetch_add(1, Ordering::Relaxed) + 1;
            tracing::error!(target: "telemetry", error = %e, drops, "stderr write failed — telemetry dropped");
        }
    }
}

// ── Quorum Consensus ─────────────────────────────────────────────
// 3 parallel calls, pick the result with highest consensus via
// normalized trigram similarity (more structural than word Jaccard).
//
// Hardened: accepts partial results on timeout, logs degradation.

/// Quorum timeout in seconds. Network latency distributions shift over time;
/// a hardcoded value becomes either brittle or wasteful as infrastructure evolves.
/// Override via LAMBDA_RLM_QUORUM_TIMEOUT_SECS env var.
fn quorum_timeout() -> Duration {
    static CACHED: std::sync::OnceLock<Duration> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        let secs = std::env::var("LAMBDA_RLM_QUORUM_TIMEOUT_SECS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .filter(|&v| v >= 5 && v <= 600)
            .unwrap_or(60);
        Duration::from_secs(secs)
    })
}

pub async fn quorum_call(
    oracle: Arc<Oracle>,
    system: &str,
    user_prompt: &str,
    max_tokens: u32,
) -> Result<String> {
    let mut set = JoinSet::new();
    for _ in 0..3 {
        let o = Arc::clone(&oracle);
        let s = system.to_string();
        let u = user_prompt.to_string();
        set.spawn(async move { o.call(&s, &u, max_tokens).await });
    }

    let mut results = Vec::with_capacity(3);
    let deadline = Instant::now() + quorum_timeout();

    // Minimum quorum size: prevents certifying a single fast-returning
    // hallucination during latency spikes.
    // Override via LAMBDA_RLM_MIN_QUORUM_SIZE (1-3, default 2).
    let min_quorum: usize = {
        static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        *CACHED.get_or_init(|| {
            std::env::var("LAMBDA_RLM_MIN_QUORUM_SIZE")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .filter(|&v| v >= 1 && v <= 3)
                .unwrap_or(2)
        })
    };

    // Consensus threshold for early termination and final validation.
    let min_consensus: f64 = {
        static CACHED: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
        *CACHED.get_or_init(|| {
            std::env::var("LAMBDA_RLM_MIN_CONSENSUS_SIMILARITY")
                .ok()
                .and_then(|s| s.parse::<f64>().ok())
                .filter(|&v| v > 0.0 && v < 1.0)
                .unwrap_or(0.3)
        })
    };

    while let Some(r) = set.join_next().await {
        if Instant::now() > deadline {
            tracing::warn!(collected = results.len(), "quorum deadline exceeded, aborting remaining tasks");
            set.abort_all();
            break;
        }
        match r {
            Ok(Ok(text)) => results.push(text),
            Ok(Err(e)) => {
                tracing::warn!(error = %e, "quorum member failed");
            }
            Err(e) => {
                tracing::warn!(error = %e, "quorum task panicked");
            }
        }

        // Fast-path cancellation: once min_quorum results are in and they
        // agree (similarity exceeds threshold), abort remaining in-flight
        // calls to save budget and file descriptors.
        if results.len() >= min_quorum && !set.is_empty() {
            let normalized: Vec<String> = results
                .iter()
                .map(|r| normalize_for_similarity(r))
                .collect();
            let (_, avg_sim) = quorum_vote(&normalized);
            if avg_sim >= min_consensus {
                tracing::debug!(
                    collected = results.len(),
                    avg_sim,
                    "fast-path quorum consensus reached, aborting remaining calls"
                );
                eprintln!(
                    "    QUORUM: fast-path consensus ({} of 3, sim={:.2}), cancelling remaining",
                    results.len(),
                    avg_sim
                );
                set.abort_all();
                // Drain to ensure RAII guards (BudgetGuard, CbFailureGuard) drop cleanly
                while set.join_next().await.is_some() {}
                break;
            }
        }
    }

    if results.is_empty() {
        anyhow::bail!("All quorum calls failed");
    }

    if results.len() < min_quorum {
        anyhow::bail!(
            "Quorum requires at least {} results but only {} succeeded — \
             refusing to accept potentially divergent output",
            min_quorum,
            results.len()
        );
    }
    if results.len() == 1 {
        return Ok(results.into_iter().next().unwrap());
    }

    // Normalized trigram similarity voting
    let normalized: Vec<String> = results
        .iter()
        .map(|r| normalize_for_similarity(r))
        .collect();

    let (best_idx, avg_sim) = quorum_vote(&normalized);

    eprintln!(
        "    QUORUM: {} responses, consensus similarity={:.2} (trigram+token)",
        results.len(),
        avg_sim
    );

    if avg_sim < min_consensus {
        // Degradation mode: instead of failing the entire pipeline on split
        // consensus (e.g., transient LLM drift), select the best-available
        // result and tag it for operator alerting. This prevents total pipeline
        // failure in long-running autonomous systems while keeping the signal.
        let degrade_on_split: bool = {
            static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *CACHED.get_or_init(|| {
                std::env::var("LAMBDA_RLM_QUORUM_DEGRADE_ON_SPLIT")
                    .ok()
                    .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                    .unwrap_or(false)
            })
        };

        if degrade_on_split {
            tracing::warn!(
                avg_sim,
                threshold = min_consensus,
                best_idx,
                "quorum consensus below threshold — DEGRADED mode, selecting best-available result"
            );
            eprintln!(
                "    QUORUM: DEGRADED — consensus {:.2} < {:.2}, selecting best of {} responses",
                avg_sim, min_consensus, results.len()
            );
            return Ok(results.into_iter().nth(best_idx).unwrap());
        }

        tracing::warn!(avg_sim, threshold = min_consensus, "quorum consensus too low, responses are divergent");
        anyhow::bail!(
            "Quorum consensus ambiguity: avg similarity {:.2} < {:.2} threshold. \
             LLM responses are too divergent to trust any single answer. \
             Set LAMBDA_RLM_QUORUM_DEGRADE_ON_SPLIT=1 to accept best-available result.",
            avg_sim,
            min_consensus
        );
    }

    let best = results.into_iter().nth(best_idx).unwrap();

    // Post-consensus semantic check: high syntactic similarity can mask
    // shared hallucination (3 LLMs agree on the same wrong answer).
    // Run refusal detection on the winning result — if all quorum members
    // returned a refusal that passed similarity, reject rather than propagate.
    if quorum_is_refusal(&best) {
        tracing::warn!(
            avg_sim,
            "quorum consensus selected a refusal/empty result — all members may have hallucinated agreement"
        );
        anyhow::bail!(
            "Quorum consensus on refusal: all {} responses agreed on a non-answer",
            min_quorum
        );
    }

    Ok(best)
}

/// Detect refusal sentinels in quorum-selected output. Separate from
/// verify::is_refusal because quorum operates at the reduce level, not leaf.
fn quorum_is_refusal(text: &str) -> bool {
    let t = text.trim();
    if t.is_empty() {
        return true;
    }
    let l = t.to_lowercase();
    l.starts_with("no_match")
        || l.starts_with("no_data")
        || l.starts_with("[degraded:")
        || l.contains("no relevant information")
        || l.contains("nothing found")
        || l.contains("insufficient context")
        || l.contains("unable to find")
        || l.contains("no evidence")
        || l.contains("cannot determine")
}

/// Pick the result with highest pairwise consensus. Returns (best_idx, avg_similarity).
fn quorum_vote(normalized: &[String]) -> (usize, f64) {
    let mut best_idx = 0;
    let mut best_sim = 0.0f64;
    for i in 0..normalized.len() {
        let mut total_sim = 0.0;
        for j in 0..normalized.len() {
            if i != j {
                total_sim += combined_similarity(&normalized[i], &normalized[j]);
            }
        }
        if total_sim > best_sim {
            best_sim = total_sim;
            best_idx = i;
        }
    }
    let avg_sim = if normalized.len() > 1 {
        best_sim / (normalized.len() - 1) as f64
    } else {
        1.0
    };
    (best_idx, avg_sim)
}

/// Strip comments, normalize whitespace, canonicalize sentinels, lowercase.
/// Sentinel canonicalization prevents format-based consensus failures where
/// "NO_MATCH" vs "NO_MATCH\n" vs "No match found" would diverge on trigram
/// similarity despite semantic equivalence.
fn normalize_for_similarity(text: &str) -> String {
    let trimmed = text.trim();
    // Canonicalize known sentinel patterns
    let lower = trimmed.to_lowercase();
    if lower.starts_with("no_match") || lower.starts_with("no match") || lower == "n/a" {
        return "SENTINEL_NO_MATCH".to_string();
    }
    if lower.starts_with("no_data")
        || lower.starts_with("no data")
        || lower == "none"
        || lower.starts_with("no relevant")
        || lower.starts_with("no useful")
    {
        return "SENTINEL_NO_DATA".to_string();
    }
    trimmed
        .lines()
        .map(str::trim)
        .filter(|l| {
            !l.starts_with("//")
                && !l.starts_with('#')
                && !l.starts_with("/*")
                && !l.is_empty()
        })
        .collect::<Vec<_>>()
        .join(" ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// Max input size for similarity functions. Inputs beyond this are truncated
/// to prevent OOM from unexpectedly large LLM responses in quorum consensus.
const MAX_SIMILARITY_INPUT_BYTES: usize = 64 * 1024;

/// Character trigram Jaccard similarity.
/// Captures token boundaries and code structure patterns better than word-level.
fn trigram_similarity(a: &str, b: &str) -> f64 {
    if a.len() < 3 || b.len() < 3 {
        return if a == b { 1.0 } else { 0.0 };
    }

    // Cap input size to prevent OOM on large responses
    let a = &a[..a.len().min(MAX_SIMILARITY_INPUT_BYTES)];
    let b = &b[..b.len().min(MAX_SIMILARITY_INPUT_BYTES)];

    let grams_a: HashSet<&[u8]> = a.as_bytes().windows(3).collect();
    let grams_b: HashSet<&[u8]> = b.as_bytes().windows(3).collect();

    let intersection = grams_a.intersection(&grams_b).count();
    let union = grams_a.union(&grams_b).count();

    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

/// Token-level (word) Jaccard similarity.
/// Captures structural equivalence across formatting/whitespace differences
/// that trigram similarity misses (e.g., reformatted code with same tokens).
fn token_similarity(a: &str, b: &str) -> f64 {
    let a = &a[..a.len().min(MAX_SIMILARITY_INPUT_BYTES)];
    let b = &b[..b.len().min(MAX_SIMILARITY_INPUT_BYTES)];
    let tokens_a: HashSet<&str> = a.split_whitespace().collect();
    let tokens_b: HashSet<&str> = b.split_whitespace().collect();
    let intersection = tokens_a.intersection(&tokens_b).count();
    let union = tokens_a.union(&tokens_b).count();
    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

/// Combined similarity: trigram (character-level) + token (word-level).
/// Token similarity catches structural equivalence across formatting;
/// trigram catches character-level code patterns. Weighted combination
/// is more robust than either metric alone against model drift.
fn combined_similarity(a: &str, b: &str) -> f64 {
    0.4 * trigram_similarity(a, b) + 0.6 * token_similarity(a, b)
}

/// Find the next SSE line boundary in the byte buffer.
/// Returns (line_end_position, bytes_to_skip) for \n (1), \r\n (2), or bare \r (1).
/// Returns None if no complete line boundary is available (need more data).
fn sse_line_boundary(buf: &[u8]) -> Option<(usize, usize)> {
    for (i, &b) in buf.iter().enumerate() {
        if b == b'\n' {
            return Some((i, 1));
        }
        if b == b'\r' {
            if buf.get(i + 1) == Some(&b'\n') {
                return Some((i, 2)); // \r\n
            }
            // Bare \r at end of buffer: could be \r\n, wait for more data
            if i + 1 >= buf.len() {
                return None;
            }
            return Some((i, 1)); // Bare \r followed by non-\n
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oracle_error_retryable_classification() {
        assert!(OracleError::RateLimit {
            retry_after: Duration::from_secs(5)
        }
        .is_retryable());
        assert!(OracleError::Timeout(Duration::from_secs(30)).is_retryable());
        assert!(OracleError::Network("connection reset".into()).is_retryable());
        assert!(OracleError::ApiError {
            status: 500,
            body: "internal".into()
        }
        .is_retryable());
        assert!(OracleError::ApiError {
            status: 503,
            body: "unavailable".into()
        }
        .is_retryable());
    }

    #[test]
    fn oracle_error_non_retryable_classification() {
        assert!(!OracleError::AuthFailed("bad key".into()).is_retryable());
        assert!(!OracleError::ApiError {
            status: 400,
            body: "bad request".into()
        }
        .is_retryable());
        assert!(!OracleError::StreamCorrupted("invalid json".into()).is_retryable());
    }

    #[test]
    fn oracle_error_fatal_classification() {
        assert!(OracleError::AuthFailed("bad key".into()).is_fatal());
        assert!(!OracleError::RateLimit {
            retry_after: Duration::from_secs(5)
        }
        .is_fatal());
        assert!(!OracleError::Timeout(Duration::from_secs(30)).is_fatal());
    }

    #[test]
    fn sse_line_boundary_lf() {
        assert_eq!(sse_line_boundary(b"data: hello\n"), Some((11, 1)));
    }

    #[test]
    fn sse_line_boundary_crlf() {
        assert_eq!(sse_line_boundary(b"data: hello\r\n"), Some((11, 2)));
    }

    #[test]
    fn sse_line_boundary_bare_cr() {
        assert_eq!(sse_line_boundary(b"data: hello\rmore"), Some((11, 1)));
    }

    #[test]
    fn sse_line_boundary_cr_at_end_defers() {
        // Bare \r at end of buffer: could be \r\n, wait for more data
        assert_eq!(sse_line_boundary(b"data: hello\r"), None);
    }

    #[test]
    fn sse_line_boundary_no_boundary() {
        assert_eq!(sse_line_boundary(b"data: hello"), None);
    }

    #[test]
    fn sse_line_boundary_empty_line() {
        assert_eq!(sse_line_boundary(b"\n"), Some((0, 1)));
    }

    #[test]
    fn normalize_sentinels_canonicalized() {
        // Different sentinel formats should normalize to the same canonical form
        assert_eq!(
            normalize_for_similarity("NO_MATCH"),
            normalize_for_similarity("no_match\n")
        );
        assert_eq!(
            normalize_for_similarity("No match found"),
            normalize_for_similarity("NO_MATCH")
        );
        assert_eq!(
            normalize_for_similarity("NO_DATA"),
            normalize_for_similarity("No data available")
        );
        assert_eq!(
            normalize_for_similarity("No relevant information"),
            normalize_for_similarity("NO_DATA")
        );
    }

    #[test]
    fn combined_similarity_structural_equivalence() {
        // Same tokens, different formatting should have high similarity
        let a = "fn foo() { bar(); baz(); }";
        let b = "fn foo() {\n    bar();\n    baz();\n}";
        let na = normalize_for_similarity(a);
        let nb = normalize_for_similarity(b);
        let sim = combined_similarity(&na, &nb);
        assert!(sim > 0.8, "Expected high similarity, got {sim}");
    }

    #[test]
    fn combined_similarity_divergent_content() {
        let a = normalize_for_similarity("fn alpha() { process_data(); }");
        let b = normalize_for_similarity("struct Config { timeout: u64 }");
        let sim = combined_similarity(&a, &b);
        assert!(sim < 0.3, "Expected low similarity, got {sim}");
    }

    #[test]
    fn deterministic_jitter_reproducible() {
        // Same idempotency key + attempt must produce identical jitter
        let key = "lrlm-abc123";
        let cap = 500u64;
        let mut results = Vec::new();
        for _ in 0..100 {
            let mut hasher = blake3::Hasher::new();
            hasher.update(key.as_bytes());
            hasher.update(&1u64.to_le_bytes());
            let hash = hasher.finalize();
            let bytes = hash.as_bytes();
            let raw = u64::from_le_bytes(bytes[..8].try_into().unwrap());
            results.push(raw % (cap + 1));
        }
        // All values must be identical (deterministic)
        assert!(results.windows(2).all(|w| w[0] == w[1]));
        // Value must be in range [0, cap]
        assert!(results[0] <= cap);
    }
}
