#!/bin/sh
# One entry point for the whole v6 run: shards, the checkpoint timer, and the
# watcher that starts the declared extension rung once the confirmatory ladder
# is complete.
#
# Re-running this is the recovery procedure. The container these shards run in
# is ephemeral and has been reclaimed mid-run before; every finished run is
# written atomically and skipped on resume by fingerprint, and the checkpoint
# timer pushes them, so a reclaim costs one checkpoint interval and a relaunch,
# not a suite.
set -eu
cd "$(dirname "$0")/.."
OUT=${V6_OUT:-results/backprop-neat-v6}
SHARDS=${SHARDS:-4}
INTERVAL=${INTERVAL:-600}
mkdir -p logs

# Never two of anything: a second set of shards on the same cells would race on
# the same run files, and a second timer would race on the index.
#
# Matched on the command line's first word, not with `pgrep -f`: a shell
# running this script has the script's own text on its command line, so a
# substring match finds "a shard" that is really the launcher looking at
# itself, and the launcher then declines to launch anything.
running() {
  ps -eo args= | grep -E "^[^ ]*/python[0-9.]* .*$1" | grep -v grep | grep -c . || true
}

if [ "$(running 'bpneat\.v6\.run')" -gt 0 ]; then
  echo "shards are already running; nothing to do"
  exit 0
fi

i=0
while [ "$i" -lt "$SHARDS" ]; do
  nohup .venv/bin/python -u -m bpneat.v6.run --out "$OUT" \
      --shard-index "$i" --shard-total "$SHARDS" \
      > "logs/v6-shard$i.log" 2>&1 &
  echo "shard $i -> logs/v6-shard$i.log"
  i=$((i + 1))
done

if ! ps -eo args= | grep -q "[v]6_checkpoint.sh; done"; then
  nohup sh -c "while true; do sleep $INTERVAL; ./bench/v6_checkpoint.sh; done" \
      > logs/v6-checkpoint.log 2>&1 &
  echo "checkpoint every ${INTERVAL}s -> logs/v6-checkpoint.log"
fi

if ! ps -eo args= | grep -q "[v]6_chain_extension.sh"; then
  nohup ./bench/v6_chain_extension.sh > logs/v6-chain.log 2>&1 &
  echo "extension rung will start when the confirmatory ladder completes"
fi
