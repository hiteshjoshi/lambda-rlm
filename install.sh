#!/usr/bin/env bash
set -euo pipefail

# lambda-RLM installer
# curl -sSf https://raw.githubusercontent.com/hiteshjoshi/lambda-rlm/main/install.sh | bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
DIM='\033[2m'
BOLD='\033[1m'
RESET='\033[0m'

REPO="https://github.com/hiteshjoshi/lambda-rlm.git"
INSTALL_DIR="${CARGO_HOME:-$HOME/.cargo}/bin"
CLONE_DIR="${LAMBDA_RLM_DIR:-$HOME/.lambda-rlm}"

echo ""
echo -e "${BOLD}  lambda-RLM v3${RESET}"
echo -e "  ${DIM}Recursive map-reduce runtime for LLMs${RESET}"
echo -e "  ${DIM}arXiv:2603.20105${RESET}"
echo ""

# ── Step 1: Check for Rust ──────────────────────────────────────

if ! command -v cargo &>/dev/null; then
    echo -e "${YELLOW}Rust not found. Installing via rustup...${RESET}"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "${CARGO_HOME:-$HOME/.cargo}/env"

    if ! command -v cargo &>/dev/null; then
        echo -e "${RED}Failed to install Rust. Install manually: https://rustup.rs${RESET}"
        exit 1
    fi
    echo -e "${GREEN}Rust installed.${RESET}"
    echo ""
fi

# ── Step 2: Clone or update ─────────────────────────────────────

if [ -d "$CLONE_DIR/.git" ]; then
    echo -e "${DIM}Updating $CLONE_DIR...${RESET}"
    git -C "$CLONE_DIR" pull --ff-only origin main 2>/dev/null || true
else
    echo -e "${DIM}Cloning to $CLONE_DIR...${RESET}"
    rm -rf "$CLONE_DIR"
    git clone --depth 1 "$REPO" "$CLONE_DIR"
fi

# ── Step 3: Build release binary ────────────────────────────────

echo -e "${DIM}Building release binary (this takes ~30s first time)...${RESET}"
cargo build --release --manifest-path "$CLONE_DIR/Cargo.toml"

BINARY="$CLONE_DIR/target/release/lambda_rlm"
if [ ! -f "$BINARY" ]; then
    echo -e "${RED}Build failed.${RESET}"
    exit 1
fi

# ── Step 4: Install to PATH ─────────────────────────────────────

mkdir -p "$INSTALL_DIR"
cp "$BINARY" "$INSTALL_DIR/lambda_rlm"
echo ""
echo -e "${GREEN}Installed to $INSTALL_DIR/lambda_rlm${RESET}"

if ! command -v lambda_rlm &>/dev/null; then
    echo ""
    echo -e "  ${YELLOW}$INSTALL_DIR is not on your PATH.${RESET}"
    echo -e "  Add to your shell profile (~/.zshrc or ~/.bashrc):"
    echo ""
    echo "    export PATH=\"$INSTALL_DIR:\$PATH\""
    echo ""
fi

# ── Step 5: Check dependencies ──────────────────────────────────

echo ""
echo -e "${BOLD}  Dependencies${RESET}"
echo ""

# Fireworks API
if [ -n "${FIREWORKS_API:-}" ]; then
    echo -e "  ${GREEN}FIREWORKS_API set${RESET}"
else
    echo -e "  ${YELLOW}FIREWORKS_API not set${RESET}"
    echo ""
    echo -e "  ${DIM}lambda-RLM uses Fireworks AI for LLM inference. The default model${RESET}"
    echo -e "  ${DIM}is Kimi K2.5 Turbo, available free through Fireworks' Fire Pass.${RESET}"
    echo ""
    echo -e "  ${DIM}1. Sign up at ${RESET}${BOLD}https://fireworks.ai${RESET}"
    echo -e "  ${DIM}2. Activate Fire Pass (free tier — includes Kimi K2.5 Turbo)${RESET}"
    echo -e "  ${DIM}3. Go to API Keys, create one${RESET}"
    echo -e "  ${DIM}4. Add to your shell profile:${RESET}"
    echo ""
    echo "     export FIREWORKS_API=\"your-key-here\""
    echo ""
fi

# Claude Code (optional)
if command -v claude &>/dev/null; then
    echo -e "  ${GREEN}claude CLI found${RESET} — --claude loop mode available"
else
    echo -e "  ${DIM}claude CLI not found (optional — needed for --claude loop mode)${RESET}"
    echo -e "  ${DIM}Install: npm install -g @anthropic-ai/claude-code${RESET}"
fi

# ── Done ────────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}  Quick start${RESET}"
echo ""
echo "  # Analyze a codebase"
echo "  lambda_rlm -p ./src -q \"Find security vulnerabilities\" -t aggregate"
echo ""
echo "  # Summarize"
echo "  lambda_rlm -p ./src -q \"What does this codebase do?\""
echo ""
echo "  # Autonomous fix loop (requires claude CLI)"
echo "  lambda_rlm -p ./src -q \"Find and fix all bugs\" --claude"
echo ""
echo "  # Dry run (no API calls, test your setup)"
echo "  lambda_rlm -p ./src -q \"test\" --dry-run"
echo ""
