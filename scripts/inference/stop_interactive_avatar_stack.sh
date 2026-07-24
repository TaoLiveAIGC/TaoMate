#!/usr/bin/env bash
set -euo pipefail

for session in taomate_service taomate_worker taomate_dialogue; do
  tmux kill-session -t "${session}" 2>/dev/null || true
done

echo "TaoMate demo stopped."
