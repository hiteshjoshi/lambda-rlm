#!/usr/bin/env bash
set -euo pipefail

# lambda-rlm installer
# Builds from source and installs to ~/.cargo/bin/lambda_rlm

RED='\033[0;31m'
GREEN='\033[0;32m'
DIM='\033[2m'
RESET='\033[0m'

echo ""
echo "  lambda-RLM v3"
echo "  Recursive map-reduce runtime for LLMs"
echo ""

# Check dependencies
for cmd in cargo rustc; do
    if ! command -v "$cmd" &>/dev/null; then
        echo -e "${RED}Error: $cmd not found.${RESET}"
        echo "Install Rust: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
        exit 1
    fi
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${DIM}Building release binary...${RESET}"
cargo build --release

BINARY="$SCRIPT_DIR/target/release/lambda_rlm"
if [ ! -f "$BINARY" ]; then
    echo -e "${RED}Build failed — binary not found${RESET}"
    exit 1
fi

INSTALL_DIR="${CARGO_HOME:-$HOME/.cargo}/bin"
mkdir -p "$INSTALL_DIR"
cp "$BINARY" "$INSTALL_DIR/lambda_rlm"

echo ""
echo -e "${GREEN}Installed to $INSTALL_DIR/lambda_rlm${RESET}"
echo ""

# Verify it's on PATH
if command -v lambda_rlm &>/dev/null; then
    echo -e "  ${DIM}$(lambda_rlm --help | head -1)${RESET}"
else
    echo -e "  ${DIM}Note: $INSTALL_DIR is not on your PATH.${RESET}"
    echo -e "  ${DIM}Add this to your shell profile:${RESET}"
    echo ""
    echo "    export PATH=\"$INSTALL_DIR:\$PATH\""
fi

# Check for optional dependencies
echo ""
if command -v claude &>/dev/null; then
    echo -e "  ${GREEN}claude CLI found${RESET} — --claude loop mode available"
else
    echo -e "  ${DIM}claude CLI not found — --claude loop mode won't work${RESET}"
    echo -e "  ${DIM}Install: npm install -g @anthropic-ai/claude-code${RESET}"
fi

echo ""
echo "  Usage:"
echo "    lambda_rlm -p ./src -q \"Find security issues\" -t aggregate"
echo "    lambda_rlm -p ./src -q \"Summarize this codebase\""
echo "    lambda_rlm -p ./src -q \"Fix all bugs\" --claude"
echo ""
echo "  Set FIREWORKS_API env var before running."
echo ""
