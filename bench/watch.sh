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
IDLE=0
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
  # A zero here is not necessarily a failure. When one phase of the queue
  # finishes, its shards exit and the next phase's are not up until the queue's
  # next poll — up to two minutes of legitimate zero. Exiting on the first one
  # dropped the watcher at exactly the moment v6's ladder completed, and the
  # container would have been reclaimed during the handover. So a zero only
  # counts once the queue itself is gone, or once it has persisted long enough
  # to outlast a changeover.
  if [ "$SHARDS" -eq 0 ]; then
    IDLE=$((IDLE + 1))
    if [ "$QUEUE" -eq 0 ]; then
      echo "SHARDS AND QUEUE BOTH GONE — run ./bench/v6_checkpoint.sh && ./bench/v6_launch.sh"
      exit 0
    fi
    if [ "$IDLE" -ge 4 ]; then
      echo "SHARDS DIED — no shard for ${IDLE}m while the queue is still up"
      exit 0
    fi
    echo "  (no shards yet; the queue is between phases, ${IDLE}m)"
  else
    IDLE=0
  fi
done
echo "watch window elapsed at $(date -u +%H:%M:%SZ); re-arm it"
