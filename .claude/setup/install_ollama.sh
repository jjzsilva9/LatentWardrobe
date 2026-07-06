#!/bin/bash
# Ollama + local LLM for Method 5 (LLM emitter-mutation prompting -- see
# scripts/llm_prompt_mutation.py). Installed 2026-07-06. Run as root (the
# `ollama.com/install.sh` installer needs sudo, and this WSL user has no
# interactive sudo password set -- see .claude/setup/README.md's own root-vs-
# user gotchas):
#
#   wsl -u root bash -c 'bash "/mnt/u/My Documents/LatentWardrobe/.claude/setup/install_ollama.sh"'
#
# Idempotent: reinstalling is a no-op if already present; pulling an
# already-pulled model is a no-op.

set -e  # no `-u` -- see README.md's conda-activate gotcha; this script doesn't source conda anyway

if ! command -v ollama >/dev/null; then
    # The installer's own zstd check fails hard on a minimal Ubuntu image
    # (this distro doesn't ship zstd by default) -- install it first.
    command -v zstd >/dev/null || apt-get install -y zstd
    curl -fsSL https://ollama.com/install.sh | sh
fi

ollama pull llama3.1:8b

echo "ollama ready: $(ollama --version)"
curl -s http://127.0.0.1:11434/api/tags
