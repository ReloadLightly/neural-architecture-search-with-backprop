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
# Anchored on the command line's FIRST word in every case. A substring match
# finds this launcher's own invocation — and, worse, any shell whose command
# line happens to quote the script's text — so the guard then reports that
# everything is already running and starts nothing. That has now happened three
# times; it is an anchored match from here on.
running() {   # running <extended regex anchored at the start of argv>
  ps -eo args= | grep -cE "$1" || true
}

if [ "$(running '^[^ ]*python[0-9.]* .*bpneat\.v6\.run')" -gt 0 ]; then
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

if [ "$(running '^sh -c while true; do sleep')" -eq 0 ]; then
  nohup sh -c "while true; do sleep $INTERVAL; ./bench/v6_checkpoint.sh; done" \
      > logs/v6-checkpoint.log 2>&1 &
  echo "checkpoint every ${INTERVAL}s -> logs/v6-checkpoint.log"
fi

if [ "$(running '^/bin/sh \./bench/queue\.sh')" -eq 0 ]; then
  nohup ./bench/queue.sh > logs/queue.log 2>&1 &
  echo "queue armed: v8 confirmatory, then both extensions -> logs/queue.log"
fi

# The shards above will only run while the container is alive, and the
# container is reclaimed within minutes of this session going idle — measured
# at 4 runs an hour idle against 109 with a tracked task holding it open. So
# the last step is not something this script can do: the session has to run
# `./bench/watch.sh` through its Bash tool with run_in_background and a long
# timeout, and re-arm it each time it returns.
echo
echo "NOW ARM THE WATCHER, or none of the above will get any compute:"
echo "  ./bench/watch.sh 110   (Bash tool, run_in_background, timeout 7200000)"
