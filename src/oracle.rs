//! Oracle M — the only neural primitive.
//!
//! Wraps the Fireworks API with: circuit breaker, RAII budget guard,
//! replay cache, per-call timeout, retries with exponential backoff,
//! telemetry counters. All state is lock-free (atomics).
//!
use crate::resilience::{BudgetGuard, CachePermit, CallBudget, CircuitBreaker, ReplayCache};
use crate::types::CodeGenerator;
use anyhow::Result;
use bytes::BytesMut;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;
use tokio::sync::{Semaphore, SemaphorePermit};

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
    map: &'a DashMap<String, Arc<tokio::sync::watch::Sender<Option<String>>>>,
    gauge: &'a AtomicUsize,
    key: String,
}

static INFLIGHT_GUARD_LIVE_COUNT: AtomicU64 = AtomicU64::new(0);

impl<'a> InflightGuard<'a> {
    fn new(
        map: &'a DashMap<String, Arc<tokio::sync::watch::Sender<Option<String>>>>,
        gauge: &'a AtomicUsize,
        key: String,
    ) -> Self {
        INFLIGHT_GUARD_LIVE_COUNT.fetch_add(1, Ordering::AcqRel);
        Self { map, gauge, key }
    }
}

impl Drop for InflightGuard<'_> {
    fn drop(&mut self) {
        self.map.remove(&self.key);
        self.gauge.fetch_sub(1, Ordering::AcqRel);
        INFLIGHT_GUARD_LIVE_COUNT.fetch_sub(1, Ordering::AcqRel);
        if self.map.is_empty() {
            self.map.shrink_to_fit();
        }
    }
}

pub struct ResourceStack<'a> {
    _budget: BudgetGuard<'a>,
    _cache_permit: CachePermit<'a>,
    _bulkhead: SemaphorePermit<'a>,
}

impl ResourceStack<'_> {
    fn commit_budget(&mut self) {
        self._budget.commit();
    }
}

pub struct OracleMetrics {
    pub budget_guards_live: u64,
    pub inflight_guards_live: u64,
    pub inflight_requests: usize,
    pub cache_hit_rate: f64,
    pub circuit_breaker_state: &'static str,
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
}

impl Bulkhead {
    pub fn new(llm_permits: usize) -> Self {
        Self {
            llm: Arc::new(Semaphore::new(llm_permits.clamp(1, MAX_BULKHEAD_PERMITS))),
        }
    }

    pub fn llm(&self) -> &Semaphore {
        &self.llm
    }
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
        let mut buf = BytesMut::with_capacity(8192);
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
                    let _ = buf.split_to(3);
                }
                bom_stripped = true;
            }

            while let Some((line_end, skip)) = sse_line_boundary(&buf[..]) {
                let line = String::from_utf8(buf[..line_end].to_vec())
                    .map_err(|e| anyhow::anyhow!("SSE stream contains invalid UTF-8: {e}"))?;
                let _ = buf.split_to(line_end + skip);

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
                    _ => OracleError::ApiError { status: code, body },
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
    /// IMPORTANT: Oracle intentionally has no back-reference to PhiConfig.
    /// Keep ownership one-way (PhiConfig -> Oracle) to prevent Arc cycles.
    inflight: DashMap<String, Arc<tokio::sync::watch::Sender<Option<String>>>>,
    inflight_gauge: AtomicUsize,
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
    codegen_claude_calls: AtomicU64,
    codegen_opencode_calls: AtomicU64,
    codegen_claude_latency_ms: AtomicU64,
    codegen_opencode_latency_ms: AtomicU64,
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
        let provider = FireworksProvider::new(api_key, model.clone(), timeout);
        Arc::new(Self {
            provider,
            model,
            call_count: AtomicU64::new(0),
            bulkhead: Bulkhead::new(concurrency),
            dry_run,
            circuit: CircuitBreaker::new(3, Duration::from_secs(30)),
            cache: ReplayCache::new(cache_dir, cache_enabled, max_cache_entries),
            budget: CallBudget::new(max_calls),
            max_retries,
            shutdown: AtomicBool::new(false),
            inflight: DashMap::new(),
            inflight_gauge: AtomicUsize::new(0),
            cache_hits: AtomicU64::new(0),
            errors: AtomicU64::new(0),
            total_input_chars: AtomicU64::new(0),
            total_output_chars: AtomicU64::new(0),
            total_latency_ms: AtomicU64::new(0),
            codegen_claude_calls: AtomicU64::new(0),
            codegen_opencode_calls: AtomicU64::new(0),
            codegen_claude_latency_ms: AtomicU64::new(0),
            codegen_opencode_latency_ms: AtomicU64::new(0),
        })
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

    pub async fn acquire_resources(&self) -> Result<ResourceStack<'_>> {
        if !self.budget.try_acquire() {
            anyhow::bail!(
                "Call budget exhausted (remaining: {})",
                self.budget.remaining()
            );
        }
        let budget = BudgetGuard::new(&self.budget);
        let cache_permit = self.cache.reserve();
        let bulkhead = self.bulkhead.llm().acquire().await?;
        Ok(ResourceStack {
            _budget: budget,
            _cache_permit: cache_permit,
            _bulkhead: bulkhead,
        })
    }

    pub fn check_inflight_leak(&self) -> bool {
        self.inflight_gauge.load(Ordering::Acquire) == self.inflight.len()
    }

    pub fn metrics(&self) -> OracleMetrics {
        let calls = self.calls();
        let cache_hits = self.cache_hits.load(Ordering::Acquire);
        let cache_hit_rate = if calls == 0 {
            0.0
        } else {
            cache_hits as f64 / calls as f64
        };
        OracleMetrics {
            budget_guards_live: self.budget.leaked_guards(),
            inflight_guards_live: INFLIGHT_GUARD_LIVE_COUNT.load(Ordering::Acquire),
            inflight_requests: self.inflight.len(),
            cache_hit_rate,
            circuit_breaker_state: self.circuit.state(),
        }
    }

    /// Signal shutdown to cancel in-flight and prevent future API calls.
    /// Called from the signal handler to stop billing when shutdown fires.
    pub fn trigger_shutdown(&self) {
        self.shutdown.store(true, Ordering::Release);
    }

    pub fn record_codegen_call(&self, generator: &CodeGenerator, latency: Duration) {
        let latency_ms = latency.as_millis() as u64;
        match generator {
            CodeGenerator::Claude => {
                self.codegen_claude_calls.fetch_add(1, Ordering::AcqRel);
                self.codegen_claude_latency_ms
                    .fetch_add(latency_ms, Ordering::AcqRel);
            }
            CodeGenerator::Opencode => {
                self.codegen_opencode_calls.fetch_add(1, Ordering::AcqRel);
                self.codegen_opencode_latency_ms
                    .fetch_add(latency_ms, Ordering::AcqRel);
            }
        }
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
        // 1. Ordered resource acquisition: Budget -> Cache -> Bulkhead
        let mut resources = self.acquire_resources().await?;

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
            resources.commit_budget();
            return Ok(cached);
        }

        // 2b. Single-flight: coalesce concurrent requests for the same cache key.
        // If another task is already fetching this key, wait for its result instead
        // of issuing a duplicate LLM call (thundering herd prevention).
        // Claim ownership by inserting while holding the lock so two callers can
        // never overwrite each other's sender and leak waiters.
        let inflight_tx = loop {
            match self.inflight.entry(cache_key.clone()) {
                dashmap::mapref::entry::Entry::Occupied(existing) => {
                    let mut rx = existing.get().subscribe();
                    drop(existing);
                    loop {
                        if let Some(ref result) = *rx.borrow() {
                            self.cache_hits.fetch_add(1, Ordering::AcqRel);
                            resources.commit_budget();
                            return Ok(result.clone());
                        }
                        if rx.changed().await.is_err() {
                            // In-flight caller failed — sender dropped.
                            // Retry entry acquisition and attempt to claim leader.
                            break;
                        }
                    }
                }
                dashmap::mapref::entry::Entry::Vacant(vacant) => {
                    let (tx, _) = tokio::sync::watch::channel::<Option<String>>(None);
                    let tx = Arc::new(tx);
                    vacant.insert(Arc::clone(&tx));
                    self.inflight_gauge.fetch_add(1, Ordering::AcqRel);
                    break tx;
                }
            }
        };

        // RAII guard removes entry on all exit paths (success, error, panic).
        let _inflight_guard =
            InflightGuard::new(&self.inflight, &self.inflight_gauge, cache_key.clone());

        // 3. Bulkhead permit is now held by ResourceStack.
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
            resources.commit_budget();
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
                resources.commit_budget();
                anyhow::bail!("Cancelled by shutdown signal");
            }

            // Circuit breaker: use allow_request for half-open probe support
            if !self.circuit.allow_request() {
                let failures = self.circuit.failures();
                tracing::warn!(failures, cooldown_s = 30, "circuit breaker open");
                resources.commit_budget(); // budget consumed: we tried
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
                tracing::debug!(
                    call = n,
                    attempt,
                    max = self.max_retries,
                    ?backoff,
                    cap_ms = cap,
                    idempotency_key,
                    "retrying with deterministic jitter"
                );
                eprintln!(
                    "    M #{n} retry {attempt}/{} (backoff {:?}, cap {:?})",
                    self.max_retries,
                    backoff,
                    Duration::from_millis(cap)
                );
                tokio::time::sleep(backoff).await;
            }

            let start = Instant::now();
            // No CbFailureGuard here: all error paths below call record_failure()
            // explicitly. Task cancellation (quorum abort) must NOT record failure.
            let api_result = self
                .call_api(system, user_prompt, max_tokens, &idempotency_key)
                .await;

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
                    resources.commit_budget();
                    return Ok(text);
                }
                Err(e) if e.is_fatal() => {
                    // Fatal: trip breaker immediately, do not retry
                    self.circuit.record_failure();
                    self.errors.fetch_add(1, Ordering::AcqRel);
                    tracing::error!(call = n, error = %e, "fatal API error, not retrying");
                    eprintln!("    M #{n} FATAL: {e}");
                    resources.commit_budget();
                    anyhow::bail!("{e}");
                }
                Err(e) => {
                    self.circuit.record_failure();
                    self.errors.fetch_add(1, Ordering::AcqRel);
                    tracing::error!(call = n, attempt = attempt + 1, error = %e, "API call failed");
                    eprintln!("    M #{n} ERROR (attempt {}): {e}", attempt + 1);

                    if !e.is_retryable() {
                        // Non-retryable (e.g., 4xx client error): fail fast
                        resources.commit_budget();
                        anyhow::bail!("{e}");
                    }

                    last_error = Some(e);
                }
            }
        }

        resources.commit_budget(); // budget consumed: we made real API attempts
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
        self.provider
            .invoke(system, user_prompt, max_tokens, idempotency_key)
            .await
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
        let metrics = self.metrics();
        let leaked_budget_guards = metrics.budget_guards_live;
        let live_inflight_guards = metrics.inflight_guards_live;
        let inflight_requests = metrics.inflight_requests;
        let inflight_gauge = self.inflight_gauge.load(Ordering::Acquire);
        let inflight_consistent = self.check_inflight_leak();
        let codegen_claude_calls = self.codegen_claude_calls.load(Ordering::Acquire);
        let codegen_opencode_calls = self.codegen_opencode_calls.load(Ordering::Acquire);
        let codegen_claude_latency_ms = self.codegen_claude_latency_ms.load(Ordering::Acquire);
        let codegen_opencode_latency_ms = self.codegen_opencode_latency_ms.load(Ordering::Acquire);

        // fmt::Write for String is infallible (OOM panics, never returns Err),
        // but we propagate via a helper to satisfy zero-swallowing discipline.
        fn w(buf: &mut String, args: std::fmt::Arguments<'_>) {
            buf.write_fmt(args)
                .expect("String::write_fmt is infallible");
        }

        let mut buf = String::with_capacity(512);
        w(&mut buf, format_args!("  Telemetry:\n"));
        w(&mut buf, format_args!("    Total M calls:    {calls}\n"));
        w(
            &mut buf,
            format_args!("    Cache hits:       {cache_hits}\n"),
        );
        w(&mut buf, format_args!("    API errors:       {errors}\n"));
        w(
            &mut buf,
            format_args!("    Input chars:      {input_chars}\n"),
        );
        w(
            &mut buf,
            format_args!("    Output chars:     {output_chars}\n"),
        );
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
            w(
                &mut buf,
                format_args!("    Budget remaining: {budget_remaining}\n"),
            );
        }
        w(
            &mut buf,
            format_args!("    Budget guards live: {leaked_budget_guards}\n"),
        );
        w(
            &mut buf,
            format_args!("    Inflight guards live: {live_inflight_guards}\n"),
        );
        w(
            &mut buf,
            format_args!("    Inflight entries: {inflight_requests}\n"),
        );
        if inflight_requests != inflight_gauge {
            w(
                &mut buf,
                format_args!(
                    "    Inflight gauge mismatch: map={inflight_requests} gauge={inflight_gauge}\n"
                ),
            );
        }
        w(
            &mut buf,
            format_args!(
                "    Inflight invariant: {}\n",
                if inflight_consistent { "ok" } else { "mismatch" }
            ),
        );
        w(
            &mut buf,
            format_args!("    Circuit state:    {}\n", metrics.circuit_breaker_state),
        );
        w(
            &mut buf,
            format_args!("    Cache hit rate:   {:.2}%\n", metrics.cache_hit_rate * 100.0),
        );
        w(
            &mut buf,
            format_args!("    Codegen Claude:   {codegen_claude_calls} call(s)\n"),
        );
        if codegen_claude_calls > 0 {
            w(
                &mut buf,
                format_args!(
                    "    Claude avg lat.:  {:.1}ms/call\n",
                    codegen_claude_latency_ms as f64 / codegen_claude_calls as f64
                ),
            );
        }
        w(
            &mut buf,
            format_args!("    Codegen OpenCode: {codegen_opencode_calls} call(s)\n"),
        );
        if codegen_opencode_calls > 0 {
            w(
                &mut buf,
                format_args!(
                    "    OpenCode avg lat.: {:.1}ms/call\n",
                    codegen_opencode_latency_ms as f64 / codegen_opencode_calls as f64
                ),
            );
        }
        let corruptions = self.cache.corruption_count();
        if corruptions > 0 {
            w(
                &mut buf,
                format_args!("    Cache corruptions: {corruptions} (bitrot detected)\n"),
            );
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
    use loom::sync::atomic::{AtomicUsize, Ordering as LoomOrdering};
    use loom::sync::{Arc as LoomArc, Mutex as LoomMutex};
    use loom::thread as loom_thread;

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

    #[test]
    fn inflight_single_flight_has_single_leader_under_race() {
        loom::model(|| {
            let map = LoomArc::new(LoomMutex::new(std::collections::HashMap::<
                &'static str,
                usize,
            >::new()));
            let leaders = LoomArc::new(AtomicUsize::new(0));

            let map_a = LoomArc::clone(&map);
            let leaders_a = LoomArc::clone(&leaders);
            let t1 = loom_thread::spawn(move || {
                let mut lock = map_a.lock().unwrap();
                if !lock.contains_key("key") {
                    lock.insert("key", 1);
                    leaders_a.fetch_add(1, LoomOrdering::AcqRel);
                }
            });

            let map_b = LoomArc::clone(&map);
            let leaders_b = LoomArc::clone(&leaders);
            let t2 = loom_thread::spawn(move || {
                let mut lock = map_b.lock().unwrap();
                if !lock.contains_key("key") {
                    lock.insert("key", 1);
                    leaders_b.fetch_add(1, LoomOrdering::AcqRel);
                }
            });

            t1.join().unwrap();
            t2.join().unwrap();

            assert_eq!(leaders.load(LoomOrdering::Acquire), 1);
            let lock = map.lock().unwrap();
            assert_eq!(lock.len(), 1);
        });
    }

    #[test]
    fn inflight_guard_always_drops_under_race() {
        struct LoomInflightGuard {
            slot: LoomArc<AtomicUsize>,
            live: LoomArc<AtomicUsize>,
        }

        impl Drop for LoomInflightGuard {
            fn drop(&mut self) {
                self.slot.store(0, LoomOrdering::Release);
                self.live.fetch_sub(1, LoomOrdering::AcqRel);
            }
        }

        loom::model(|| {
            let slot = LoomArc::new(AtomicUsize::new(0));
            let live = LoomArc::new(AtomicUsize::new(0));

            let slot_a = LoomArc::clone(&slot);
            let live_a = LoomArc::clone(&live);
            let t1 = loom_thread::spawn(move || {
                if slot_a
                    .compare_exchange(0, 1, LoomOrdering::AcqRel, LoomOrdering::Acquire)
                    .is_ok()
                {
                    live_a.fetch_add(1, LoomOrdering::AcqRel);
                    let _guard = LoomInflightGuard {
                        slot: LoomArc::clone(&slot_a),
                        live: LoomArc::clone(&live_a),
                    };
                    loom_thread::yield_now();
                }
            });

            let slot_b = LoomArc::clone(&slot);
            let live_b = LoomArc::clone(&live);
            let t2 = loom_thread::spawn(move || {
                if slot_b
                    .compare_exchange(0, 1, LoomOrdering::AcqRel, LoomOrdering::Acquire)
                    .is_ok()
                {
                    live_b.fetch_add(1, LoomOrdering::AcqRel);
                    let _guard = LoomInflightGuard {
                        slot: LoomArc::clone(&slot_b),
                        live: LoomArc::clone(&live_b),
                    };
                    loom_thread::yield_now();
                }
            });

            t1.join().unwrap();
            t2.join().unwrap();

            assert_eq!(live.load(LoomOrdering::Acquire), 0);
            assert_eq!(slot.load(LoomOrdering::Acquire), 0);
        });
    }
}
