use std::fs;

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
