#!/bin/sh
# Run the compute phases in order, each one gated on the last.
#
# Order is a decision, not an accident. v6's declared extension rung is about
# seventeen wall-hours and scores no hypothesis; v8's confirmatory suite is
# about five and is a complete new result, the first in this project on real
# data. So v8 goes first and both extensions go last.
#
#   1. v6 confirmatory   - the budget ladder, 1,080 runs
#   2. v8 confirmatory   - four arms on three real datasets, 360 runs
#   3. v6 extension      - the 8x rung, 270 runs, scores nothing
#   4. v8 extension      - digits, 120 runs, scores nothing
#
# Sealed tests are deliberately NOT in this queue. Each is touched exactly once
# per release and cannot be undone; it is a decision, not a step, and it waits
# for a person. The queue stops when the compute is done and says so.
#
# Safe to re-run: every phase is skipped if its manifest already says complete,
# and the suites themselves skip finished runs by fingerprint.
set -eu
cd "$(dirname "$0")/.."
SHARDS=${SHARDS:-4}
POLL=${POLL:-120}
V6=results/backprop-neat-v6
V8=results/backprop-neat-v8

complete() {
  [ -f "$1" ] || return 1
  .venv/bin/python -c "
import json, sys
m = json.load(open('$1'))
sys.exit(0 if m.get('$2') else 1)
" 2>/dev/null
}

await() {
  echo "waiting for $3"
  while ! complete "$1" "$2"; do sleep "$POLL"; done
  echo "$3 complete at $(date -u +%H:%M:%SZ)"
}

launch() {
  module=$1; out=$2; shift 2
  mkdir -p logs
  tag=$(echo "$module" | tr '.' '-')
  i=0
  while [ "$i" -lt "$SHARDS" ]; do
    nohup .venv/bin/python -u -m "$module" --out "$out" "$@" \
        --shard-index "$i" --shard-total "$SHARDS" \
        > "logs/$tag-shard$i.log" 2>&1 &
    i=$((i + 1))
  done
  echo "launched $SHARDS shards of $module $* at $(date -u +%H:%M:%SZ)"
}

await "$V6/manifest.json" complete "v6 confirmatory ladder"
./bench/v6_checkpoint.sh || true

if ! complete "$V8/manifest.json" complete; then
  launch bpneat.v8.run "$V8"
  await "$V8/manifest.json" complete "v8 confirmatory suite"
fi
./bench/v6_checkpoint.sh || true

if ! complete "$V6/manifest.json" extension_complete; then
  launch bpneat.v6.run "$V6" --extension
  await "$V6/manifest.json" extension_complete "v6 extension rung"
fi
./bench/v6_checkpoint.sh || true

if ! complete "$V8/manifest.json" extension_complete; then
  launch bpneat.v8.run "$V8" --extension
  await "$V8/manifest.json" extension_complete "v8 extension dataset"
fi
./bench/v6_checkpoint.sh || true

echo "ALL COMPUTE COMPLETE at $(date -u +%H:%M:%SZ)."
echo "Two sealed tests are now available and neither has been taken."
echo "They are touched once each and wait for a person:"
echo "  make v6-finaltest && make v6-release"
echo "  make v8-finaltest && make v8-release"
