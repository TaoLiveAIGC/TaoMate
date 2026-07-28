#!/usr/bin/env bash
set -euo pipefail

SESSIONS=(taomate_service taomate_worker taomate_dialogue)

for session in "${SESSIONS[@]}"; do
  tmux send-keys -t "${session}" C-c 2>/dev/null || true
done
sleep 2

for session in "${SESSIONS[@]}"; do
  tmux kill-session -t "${session}" 2>/dev/null || true
done

echo "TaoMate demo stopped."
