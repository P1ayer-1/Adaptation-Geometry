#!/usr/bin/env bash
# Kill a hung `uag train` (flaky laptop GPU under WSL) when LOG has not changed for STALL seconds;
# keep_running.sh then restarts the resumable script. Stops when DONE appears in LOG.
#   nohup bash scripts/hang_watchdog.sh results/crossed_test.log "crossed_test: done" 900 &
LOG="$1"; DONE="$2"; STALL="${3:-900}"
while ! grep -q "$DONE" "$LOG" 2>/dev/null; do
  sleep 60
  age=$(( $(date +%s) - $(stat -c %Y "$LOG") ))
  if [ "$age" -gt "$STALL" ]; then
    for pid in $(pgrep -f "bin/uag train"); do
      echo "=== hang_watchdog: log idle ${age}s, killing uag train pid $pid $(date) ===" >> "$LOG"
      kill "$pid"
    done
  fi
done
