use anyhow::{Context, Result};
use serde_json::Value;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const LOG_ROOT_DIR: &str = ".lambda-logs";

#[derive(Clone, Debug)]
pub struct RunLogger {
    run_dir: PathBuf,
}

impl RunLogger {
    pub fn initialize(input_path: &Path, run_id: Option<&str>) -> Option<Self> {
        match Self::new(input_path, run_id) {
            Ok(logger) => Some(logger),
            Err(error) => {
                tracing::warn!(error = %error, "run logger unavailable; continuing without local log files");
                None
            }
        }
    }

    fn new(input_path: &Path, run_id: Option<&str>) -> Result<Self> {
        let base = if input_path.is_file() {
            input_path.parent().unwrap_or(input_path)
        } else {
            input_path
        };
        let base = std::fs::canonicalize(base).unwrap_or_else(|_| base.to_path_buf());
        let logs_root = base.join(LOG_ROOT_DIR);
        create_dir_secure(&logs_root)?;

        let run_name = sanitize_run_name(run_id.unwrap_or("run"));
        let run_dir = logs_root.join(run_name);
        create_dir_secure(&run_dir)?;
        create_dir_secure(&run_dir.join("analysis"))?;
        create_dir_secure(&run_dir.join("codegen"))?;

        Ok(Self { run_dir })
    }

    pub fn write_run_start(&self, payload: &Value) {
        if let Err(error) = self.write_json("run.json", payload) {
            tracing::warn!(error = %error, "failed to write run metadata");
        }
    }

    pub fn write_run_finish(&self, payload: &Value) {
        if let Err(error) = self.write_json("run.json", payload) {
            tracing::warn!(error = %error, "failed to update run metadata");
        }
    }

    pub fn write_analysis(&self, iteration: usize, content: &str) {
        let path = self
            .run_dir
            .join("analysis")
            .join(format!("iteration-{iteration:03}.md"));
        if let Err(error) = write_file_atomically(&path, content.as_bytes()) {
            tracing::warn!(path = %path.display(), error = %error, "failed to write analysis log");
        }
    }

    pub fn write_codegen_disabled(&self) {
        let path = self.run_dir.join("codegen").join("disabled.txt");
        let message =
            "No code generator was enabled for this run (neither --claude nor --opencode).\n";
        if let Err(error) = write_file_atomically(&path, message.as_bytes()) {
            tracing::warn!(path = %path.display(), error = %error, "failed to write disabled codegen marker");
        }
    }

    pub fn write_codegen_prompt(&self, iteration: usize, generator: &str, prompt: &str) {
        let dir = self.run_dir.join("codegen");
        let generator_path = dir.join(format!("iteration-{iteration:03}.generator.txt"));
        let prompt_path = dir.join(format!("iteration-{iteration:03}.prompt.txt"));
        if let Err(error) = write_file_atomically(&generator_path, generator.as_bytes()) {
            tracing::warn!(path = %generator_path.display(), error = %error, "failed to write generator marker");
        }
        if let Err(error) = write_file_atomically(&prompt_path, prompt.as_bytes()) {
            tracing::warn!(path = %prompt_path.display(), error = %error, "failed to write generator prompt");
        }
    }

    pub fn write_codegen_result(
        &self,
        iteration: usize,
        stdout: Option<&str>,
        stderr: Option<&str>,
        summary: Option<&str>,
    ) {
        let dir = self.run_dir.join("codegen");
        if let Some(stdout) = stdout {
            let stdout_path = dir.join(format!("iteration-{iteration:03}.stdout.txt"));
            if let Err(error) = write_file_atomically(&stdout_path, stdout.as_bytes()) {
                tracing::warn!(path = %stdout_path.display(), error = %error, "failed to write generator stdout");
            }
        }
        if let Some(stderr) = stderr {
            let stderr_path = dir.join(format!("iteration-{iteration:03}.stderr.txt"));
            if let Err(error) = write_file_atomically(&stderr_path, stderr.as_bytes()) {
                tracing::warn!(path = %stderr_path.display(), error = %error, "failed to write generator stderr");
            }
        }
        if let Some(summary) = summary {
            let summary_path = dir.join(format!("iteration-{iteration:03}.summary.txt"));
            if let Err(error) = write_file_atomically(&summary_path, summary.as_bytes()) {
                tracing::warn!(path = %summary_path.display(), error = %error, "failed to write generator summary");
            }
        }
    }

    pub fn append_event(&self, kind: &str, status: &str, message: &str, extra: Option<Value>) {
        let path = self.run_dir.join("events.jsonl");
        let mut event = serde_json::json!({
            "ts_ms": unix_time_ms(),
            "kind": kind,
            "status": status,
            "message": message,
        });
        if let Some(extra) = extra {
            event["extra"] = extra;
        }

        let line = match serde_json::to_string(&event) {
            Ok(line) => line,
            Err(error) => {
                tracing::warn!(error = %error, "failed to serialize local run event");
                return;
            }
        };

        let write_result = (|| -> Result<()> {
            let mut file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .with_context(|| format!("failed to open {}", path.display()))?;
            file.write_all(line.as_bytes())?;
            file.write_all(b"\n")?;
            file.flush()?;
            Ok(())
        })();

        if let Err(error) = write_result {
            tracing::warn!(path = %path.display(), error = %error, "failed to append local run event");
        }
    }

    fn write_json(&self, rel: &str, payload: &Value) -> Result<()> {
        let path = self.run_dir.join(rel);
        let bytes =
            serde_json::to_vec_pretty(payload).context("failed to serialize run metadata")?;
        write_file_atomically(&path, &bytes)
    }
}

fn create_dir_secure(path: &Path) -> Result<()> {
    std::fs::create_dir_all(path)
        .with_context(|| format!("failed to create {}", path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700));
    }
    Ok(())
}

fn write_file_atomically(path: &Path, content: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("missing parent directory"))?;
    create_dir_secure(parent)?;
    let mut tmp = tempfile::NamedTempFile::new_in(parent)
        .with_context(|| format!("failed to create temp file in {}", parent.display()))?;
    tmp.write_all(content)?;
    tmp.as_file_mut().sync_all()?;
    let persisted = tmp
        .persist(path)
        .map_err(|e| anyhow::anyhow!("failed to persist {}: {}", path.display(), e.error))?;
    persisted.sync_all()?;
    if let Ok(dir) = std::fs::File::open(parent) {
        let _ = dir.sync_all();
    }
    Ok(())
}

fn sanitize_run_name(run_id: &str) -> String {
    let mut out = String::with_capacity(run_id.len());
    for ch in run_id.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        format!("run_{}", unix_time_ms())
    } else {
        out
    }
}

fn unix_time_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}
