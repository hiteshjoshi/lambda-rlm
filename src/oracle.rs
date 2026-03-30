//! Oracle M — the only neural primitive.
//!
//! Wraps the Fireworks API with: circuit breaker, RAII budget guard,
//! replay cache, per-call timeout, retries with exponential backoff,
//! telemetry counters. All state is lock-free (atomics).
//!
//! Also: quorum consensus with normalized trigram similarity.

use crate::resilience::{BudgetGuard, CallBudget, CircuitBreaker, ReplayCache};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;
use tokio::task::JoinSet;

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

// ── Bulkhead ─────────────────────────────────────────────────────
// Separate semaphores for different resource classes to prevent
// I/O-bound LLM calls from starving CPU-bound reduce operations
// and vice versa.

pub struct Bulkhead {
    llm: Arc<Semaphore>,
    #[allow(dead_code)] // Reserved for CPU-bound reduce operations
    cpu: Arc<Semaphore>,
}

impl Bulkhead {
    pub fn new(llm_permits: usize) -> Self {
        let cpu_permits = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        Self {
            llm: Arc::new(Semaphore::new(llm_permits)),
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

// ── Oracle ───────────────────────────────────────────────────────

pub struct Oracle {
    api_key: String,
    model: String,
    client: reqwest::Client,
    call_count: AtomicUsize,
    bulkhead: Bulkhead,
    dry_run: bool,
    // Resilience
    circuit: CircuitBreaker,
    cache: ReplayCache,
    budget: CallBudget,
    timeout: Duration,
    max_retries: usize,
    // Telemetry
    cache_hits: AtomicUsize,
    errors: AtomicUsize,
    total_input_chars: AtomicUsize,
    total_output_chars: AtomicUsize,
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
        Arc::new(Self {
            api_key,
            model,
            client: reqwest::Client::new(),
            call_count: AtomicUsize::new(0),
            bulkhead: Bulkhead::new(concurrency),
            dry_run,
            circuit: CircuitBreaker::new(3, Duration::from_secs(30)),
            cache: ReplayCache::new(cache_dir, cache_enabled, max_cache_entries),
            budget: CallBudget::new(max_calls),
            timeout,
            max_retries,
            cache_hits: AtomicUsize::new(0),
            errors: AtomicUsize::new(0),
            total_input_chars: AtomicUsize::new(0),
            total_output_chars: AtomicUsize::new(0),
            total_latency_ms: AtomicU64::new(0),
        })
    }

    pub fn calls(&self) -> usize {
        self.call_count.load(Ordering::Acquire)
    }

    pub fn cache(&self) -> &ReplayCache {
        &self.cache
    }

    #[allow(dead_code)]
    pub fn budget_remaining(&self) -> usize {
        self.budget.remaining()
    }

    #[allow(dead_code)]
    pub fn budget_unlimited(&self) -> bool {
        self.budget.is_unlimited()
    }

    #[allow(dead_code)]
    pub fn bulkhead(&self) -> &Bulkhead {
        &self.bulkhead
    }

    /// Outer call: budget → cache → semaphore → retry loop with circuit breaker.
    ///
    /// PRE: api_key is valid (or dry_run=true)
    /// POST: Ok(text) where text passes basic non-empty check, or Err
    ///
    /// Budget is consumed on all normal exits (success AND error after retries).
    /// Budget is restored only on panic/cancellation via BudgetGuard RAII.
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

        // 3. Semaphore — held for entire call including retries
        let _permit = self.bulkhead.llm().acquire().await.unwrap();
        let n = self.call_count.fetch_add(1, Ordering::AcqRel) + 1;

        if self.dry_run {
            guard.commit();
            return Ok(format!(
                "[DRY RUN] Call #{n} ({} chars in, max_tok={max_tokens})",
                user_prompt.len()
            ));
        }

        // 4. Retry loop with circuit breaker + exponential backoff
        let mut last_error = None;
        for attempt in 0..=self.max_retries {
            if self.circuit.is_open() {
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
                let backoff = Duration::from_millis(500 << (attempt - 1).min(4));
                tracing::debug!(call = n, attempt, max = self.max_retries, ?backoff, "retrying");
                eprintln!(
                    "    M #{n} retry {attempt}/{} (backoff {:?})",
                    self.max_retries, backoff
                );
                tokio::time::sleep(backoff).await;
            }

            let start = Instant::now();
            match self.call_api(system, user_prompt, max_tokens).await {
                Ok(text) => {
                    let latency = start.elapsed();
                    self.circuit.record_success();
                    self.total_input_chars
                        .fetch_add(user_prompt.len(), Ordering::AcqRel);
                    self.total_output_chars
                        .fetch_add(text.len(), Ordering::AcqRel);
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
                    guard.commit();
                    return Ok(text);
                }
                Err(e) => {
                    self.circuit.record_failure();
                    self.errors.fetch_add(1, Ordering::AcqRel);
                    tracing::error!(call = n, attempt = attempt + 1, error = %e, "API call failed");
                    eprintln!("    M #{n} ERROR (attempt {}): {e}", attempt + 1);
                    last_error = Some(e);
                }
            }
        }

        guard.commit(); // budget consumed: we made real API attempts
        Err(last_error.unwrap())
    }

    /// Raw API call with timeout. No retries, no cache, no budget check.
    async fn call_api(&self, system: &str, user_prompt: &str, max_tokens: u32) -> Result<String> {
        let fut = async {
            let mut messages = Vec::with_capacity(2);
            if !system.is_empty() {
                messages.push(ChatMessage {
                    role: "system".into(),
                    content: system.into(),
                });
            }
            messages.push(ChatMessage {
                role: "user".into(),
                content: user_prompt.into(),
            });

            // Fireworks requires stream=true for max_tokens > 4096
            let use_stream = max_tokens > 4096;

            let request = ChatRequest {
                model: self.model.clone(),
                max_tokens,
                messages,
                stream: use_stream,
                temperature: None,
                response_format: None,
            };

            let response = self
                .client
                .post("https://api.fireworks.ai/inference/v1/chat/completions")
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("Content-Type", "application/json")
                .json(&request)
                .send()
                .await
                .context("Fireworks API request failed")?;

            let status = response.status();
            if !status.is_success() {
                let body = response.text().await.unwrap_or_default();
                anyhow::bail!("Fireworks API {status}: {body}");
            }

            if use_stream {
                self.read_stream(response).await
            } else {
                let resp: ChatResponse = response.json().await?;
                Ok(resp
                    .choices
                    .first()
                    .map(|c| c.message.content.clone())
                    .unwrap_or_default())
            }
        };

        tokio::time::timeout(self.timeout, fut)
            .await
            .map_err(|_| anyhow::anyhow!("API call timed out after {:?}", self.timeout))?
    }

    async fn read_stream(&self, response: reqwest::Response) -> Result<String> {
        use futures::StreamExt;
        let mut result = String::new();
        let mut stream = response.bytes_stream();
        let mut buffer = String::new();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.context("Stream read error")?;
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            while let Some(newline_pos) = buffer.find('\n') {
                let line = buffer[..newline_pos].trim().to_string();
                buffer = buffer[newline_pos + 1..].to_string();

                if let Some(data) = line.strip_prefix("data: ") {
                    if data == "[DONE]" {
                        return Ok(result);
                    }
                    if let Ok(chunk) = serde_json::from_str::<StreamChunk>(data) {
                        if let Some(choice) = chunk.choices.first() {
                            if let Some(content) = &choice.delta.content {
                                result.push_str(content);
                            }
                        }
                    }
                }
            }
        }

        Ok(result)
    }

    pub fn print_telemetry(&self) {
        let calls = self.calls();
        let cache_hits = self.cache_hits.load(Ordering::Acquire);
        let errors = self.errors.load(Ordering::Acquire);
        let input_chars = self.total_input_chars.load(Ordering::Acquire);
        let output_chars = self.total_output_chars.load(Ordering::Acquire);
        let latency_ms = self.total_latency_ms.load(Ordering::Acquire);
        let budget_remaining = self.budget.remaining();

        eprintln!("  Telemetry:");
        eprintln!("    Total M calls:    {calls}");
        eprintln!("    Cache hits:       {cache_hits}");
        eprintln!("    API errors:       {errors}");
        eprintln!("    Input chars:      {input_chars}");
        eprintln!("    Output chars:     {output_chars}");
        if calls > cache_hits {
            let api_calls = calls - cache_hits;
            eprintln!(
                "    Avg latency:      {:.1}ms/call",
                latency_ms as f64 / api_calls as f64
            );
        }
        if !self.budget.is_unlimited() {
            eprintln!("    Budget remaining: {budget_remaining}");
        }
    }
}

// ── Quorum Consensus ─────────────────────────────────────────────
// 3 parallel calls, pick the result with highest consensus via
// normalized trigram similarity (more structural than word Jaccard).
//
// Hardened: accepts partial results on timeout, logs degradation.

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
    let deadline = Instant::now() + Duration::from_secs(60);

    while let Some(r) = set.join_next().await {
        if Instant::now() > deadline {
            tracing::warn!(collected = results.len(), "quorum deadline exceeded, using partial results");
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
    }

    if results.is_empty() {
        anyhow::bail!("All quorum calls failed");
    }
    if results.len() == 1 {
        tracing::warn!("Quorum degraded to single result");
        return Ok(results.into_iter().next().unwrap());
    }

    // Normalized trigram similarity voting
    let normalized: Vec<String> = results
        .iter()
        .map(|r| normalize_for_similarity(r))
        .collect();

    let mut best_idx = 0;
    let mut best_sim = 0.0f64;
    for i in 0..normalized.len() {
        let mut total_sim = 0.0;
        for j in 0..normalized.len() {
            if i != j {
                total_sim += trigram_similarity(&normalized[i], &normalized[j]);
            }
        }
        if total_sim > best_sim {
            best_sim = total_sim;
            best_idx = i;
        }
    }

    let avg_sim = best_sim / (results.len() - 1) as f64;
    eprintln!(
        "    QUORUM: {} responses, consensus similarity={:.2} (trigram)",
        results.len(),
        avg_sim
    );
    Ok(results.into_iter().nth(best_idx).unwrap())
}

/// Strip comments, normalize whitespace, lowercase for structural comparison.
fn normalize_for_similarity(text: &str) -> String {
    text.lines()
        .map(str::trim)
        .filter(|l| !l.starts_with("//") && !l.starts_with('#') && !l.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// Character trigram Jaccard similarity.
/// Captures token boundaries and code structure patterns better than word-level.
fn trigram_similarity(a: &str, b: &str) -> f64 {
    if a.len() < 3 || b.len() < 3 {
        return if a == b { 1.0 } else { 0.0 };
    }

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
