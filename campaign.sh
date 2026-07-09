#!/usr/bin/env bash
# campaign.sh — run N Vigiles experiment profiles CONCURRENTLY, the standard way.
#
# Each profile runs the ONE canonical command from the README:
#   auspexai-tenant experiment launch --profile <p>   (build -> submit -> approve -> drive)
# ...in its own detached tmux session, so every experiment has a persistent driver
# that survives a disconnected terminal, and all N drive at once. Logs land in
# runs/<profile>.log.
#
# This exists because `launch` BLOCKS driving for the whole run, so N concurrent
# experiments need N persistent processes — running them in one terminal drives
# only the first and leaves the rest "approved with no work units". This wrapper
# is the standard; don't hand-run launch in a loop.
#
# Usage:
#   ./campaign.sh overnight10_mistral overnight10_llama overnight10_qwen3
#
# Manage:
#   tmux ls                        # list running experiment drivers
#   tmux attach -t vig-<profile>   # watch one (Ctrl-b d to detach)
#   tmux kill-session -t vig-<profile>   # stop one (aborts that run)
set -euo pipefail
cd "$(dirname "$0")"

if [ "$#" -eq 0 ]; then
  echo "usage: $0 <profile> [<profile> ...]" >&2
  echo "  e.g. $0 overnight10_mistral overnight10_llama overnight10_qwen3" >&2
  exit 2
fi
command -v tmux >/dev/null || { echo "tmux not found — install it (brew install tmux)"; exit 1; }
command -v auspexai-tenant >/dev/null || { echo "auspexai-tenant not on PATH"; exit 1; }

mkdir -p runs
for p in "$@"; do
  s="vig-$p"
  if tmux has-session -t "$s" 2>/dev/null; then
    echo "SKIP $p — a session '$s' is already running (tmux kill-session -t $s to replace)"
    continue
  fi
  tmux new-session -d -s "$s" \
    "auspexai-tenant experiment launch --profile $p 2>&1 | tee runs/$p.log"
  echo "started $p  ->  tmux session '$s'   (log: runs/$p.log)"
done

echo
echo "all launched. each waits for your maintainer approval, then drives."
echo "  watch:  tmux attach -t vig-<profile>     list: tmux ls"
echo "  stop:   tmux kill-session -t vig-<profile>"
