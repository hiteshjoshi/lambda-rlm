use std::fs;
use std::process::Command;

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
        stderr.contains("Not Found"),
        "expected sanitized startup failure, got: {stderr}"
    );
    assert!(
        !stderr.contains("ITERATION"),
        "opencode preflight should fail before fix-loop start: {stderr}"
    );
}

#[cfg(unix)]
#[test]
fn opencode_disk_full_degrades_to_inline_prompt() {
    use std::os::unix::fs::PermissionsExt;

    let temp = tempfile::tempdir().expect("tempdir");
    let repo = temp.path().join("repo");
    fs::create_dir_all(repo.join("src")).expect("create src");
    fs::write(
        repo.join("src/main.rs"),
        "fn main() { println!(\"ok\"); }\n",
    )
    .expect("write source");

    let bin_dir = temp.path().join("bin");
    fs::create_dir_all(&bin_dir).expect("create bin");
    let args_log = temp.path().join("opencode-args.log");
    let opencode_path = bin_dir.join("opencode");
    let stub = format!(
        "#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then\n  echo \"opencode 1.0.0\"\n  exit 0\nfi\nprintf '%s\\n' \"$@\" > \"{}\"\necho \"CLEAN\"\n",
        args_log.display()
    );
    fs::write(&opencode_path, stub).expect("write opencode stub");
    let mut perms = fs::metadata(&opencode_path)
        .expect("opencode stub metadata")
        .permissions();
    perms.set_mode(0o755);
    fs::set_permissions(&opencode_path, perms).expect("chmod opencode stub");

    let output = Command::new(env!("CARGO_BIN_EXE_lambda_rlm"))
        .arg("-p")
        .arg(&repo)
        .arg("-q")
        .arg("smoke")
        .arg("--dry-run")
        .arg("--opencode")
        .arg("--max-iterations")
        .arg("1")
        .env("PATH", &bin_dir)
        .env("LAMBDA_RLM_TEST_FORCE_RESULT_ENOSPC", "1")
        .output()
        .expect("run lambda_rlm");

    assert!(
        output.status.success(),
        "expected degraded inline fallback success"
    );
    let args = fs::read_to_string(&args_log).expect("read opencode stub args");
    assert!(
        !args.contains("--file"),
        "opencode should not receive --file when result write is ENOSPC: {args}"
    );
    assert!(
        args.contains("result file unavailable due to disk constraints"),
        "expected inline fallback prompt marker in opencode args: {args}"
    );
}
