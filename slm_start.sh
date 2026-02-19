#!/usr/bin/env bash
set -euo pipefail

if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama CLI not found. Install Ollama first." >&2
  exit 1
fi

if ollama list >/dev/null 2>&1; then
  echo "Ollama server already running."
  exit 0
fi

nohup ollama serve >/tmp/ollama.log 2>&1 &
sleep 2

if ollama list >/dev/null 2>&1; then
  echo "Ollama server started."
  exit 0
fi

echo "Failed to start Ollama server. Check /tmp/ollama.log" >&2
exit 1
