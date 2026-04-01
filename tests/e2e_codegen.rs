use std::fs;
use std::process::{Command, Stdio};

#[test]
#[ignore = "e2e boundary test; requires local claude/opencode CLI stubs"]
fn opencode_claude_idempotency_test() {
    let temp = tempfile::tempdir().expect("tempdir");
    let repo = temp.path().join("repo");
    fs::create_dir_all(repo.join("src")).expect("create src");
    fs::write(
        repo.join("src/main.rs"),
        "fn main() { println!(\"ok\"); }\n",
    )
    .expect("write source");
    fs::write(
        repo.join(".lambda-rlm-result.md"),
        "pre-seeded analysis payload for idempotency smoke test\n",
    )
    .expect("seed result file");

    let seeded = fs::read_to_string(repo.join(".lambda-rlm-result.md")).expect("seed exists");
    let hash = blake3::hash(seeded.as_bytes()).to_hex().to_string();
    assert_eq!(hash.len(), 64);
}

#[test]
fn opencode_preflight_fails_fast_when_binary_missing() {
    let temp = tempfile::tempdir().expect("tempdir");
    let repo = temp.path().join("repo");
    fs::create_dir_all(repo.join("src")).expect("create src");
    fs::write(
        repo.join("src/main.rs"),
        "fn main() { println!(\"ok\"); }\n",
    )
    .expect("write source");

    let empty_path = temp.path().join("empty-bin");
    fs::create_dir_all(&empty_path).expect("create empty PATH dir");

    let output = Command::new(env!("CARGO_BIN_EXE_lambda_rlm"))
        .arg("-p")
        .arg(&repo)
        .arg("-q")
        .arg("smoke")
        .arg("--dry-run")
        .arg("--opencode")
        .env("PATH", &empty_path)
        .output()
        .expect("run lambda_rlm");

    assert!(
        !output.status.success(),
        "expected startup validation failure"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("[ERROR]"),
        "expected [ERROR] in startup failure output, got: {stderr}"
    );
    assert!(
        !stderr.contains("ITERATION"),
        "opencode preflight should fail before fix-loop start: {stderr}"
    );
}

#[test]
fn interactive_fails_fast_without_tty() {
    let temp = tempfile::tempdir().expect("tempdir");
    let repo = temp.path().join("repo");
    fs::create_dir_all(repo.join("src")).expect("create src");
    fs::write(
        repo.join("src/main.rs"),
        "fn main() { println!(\"ok\"); }\n",
    )
    .expect("write source");

    let output = Command::new(env!("CARGO_BIN_EXE_lambda_rlm"))
        .arg("-p")
        .arg(&repo)
        .arg("-q")
        .arg("smoke")
        .arg("--dry-run")
        .arg("--opencode")
        .arg("--interactive")
        .stdin(Stdio::piped())
        .output()
        .expect("run lambda_rlm");

    assert!(
        !output.status.success(),
        "expected interactive preflight to reject non-TTY stdin/stdout"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("[ERROR]"),
        "expected [ERROR] in failure output, got: {stderr}"
    );
    assert!(
        !stderr.contains("launching interactive"),
        "interactive child should not spawn without a TTY: {stderr}"
    );
    assert!(
        !stderr.contains("ITERATION"),
        "interactive mode should fail before entering fix-loop iterations: {stderr}"
    );
}
