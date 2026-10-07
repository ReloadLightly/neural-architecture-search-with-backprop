#!/bin/sh
# Hold the container open while the compute queue runs, and trace progress.
#
# This is not decoration. Measured on this machine: with only a detached
# `nohup` queue and the session idle, the container is reclaimed within minutes
# and the suite advances about 4 runs an hour. With a *harness-tracked*
# background task running — one the session is waiting on — the container stays
# up and the same suite advances about 109 runs an hour, a factor of twenty-five.
#
# So this script is run through the Bash tool with `run_in_background` and a
# long timeout, never with nohup. It polls once a minute, prints a line the
# session can read back, and exits early if the shards die or the queue
# finishes, so that a failure is noticed in a minute rather than in two hours.
#
# Usage: ./bench/watch.sh [minutes]
set -eu
cd "$(dirname "$0")/.."
MINUTES=${1:-110}
START=$(ls results/backprop-neat-v6/raw/runs/*.json 2>/dev/null | wc -l | tr -d ' ')
echo "start=$START at $(date -u +%H:%M:%SZ), watching for ${MINUTES}m"

i=0
while [ "$i" -lt "$MINUTES" ]; do
  sleep 60
  i=$((i + 1))
  V6=$(ls results/backprop-neat-v6/raw/runs/*.json 2>/dev/null | wc -l | tr -d ' ')
  V8=$(ls results/backprop-neat-v8/raw/runs/*.json 2>/dev/null | wc -l | tr -d ' ')
  SHARDS=$(ps -eo args= | grep -cE '^[^ ]*python[0-9.]* .*bpneat\.v(6|8)\.run' || true)
  QUEUE=$(ps -eo args= | grep -cE '^/bin/sh \./bench/queue\.sh' || true)
  echo "$(date -u +%H:%M:%SZ) v6=$V6 (+$((V6 - START))) v8=$V8 shards=$SHARDS queue=$QUEUE"
  if grep -q "ALL COMPUTE COMPLETE" logs/queue.log 2>/dev/null; then
    echo "QUEUE DONE — both sealed tests are available and neither has been taken"
    exit 0
  fi
  if [ "$SHARDS" -eq 0 ]; then
    echo "SHARDS DIED — run ./bench/v6_checkpoint.sh && make v6-run"
    exit 0
  fi
done
echo "watch window elapsed at $(date -u +%H:%M:%SZ); re-arm it"
