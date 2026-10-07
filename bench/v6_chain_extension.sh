#!/bin/sh
# Start the declared b16800 extension rung the moment the confirmatory ladder
# is complete, and not before.
#
# Order matters. The suite runs a cell's rungs cheapest-first, so launching with
# `--extension` from the start would spend half an hour on 16,800 candidates in
# cell 1 before touching cell 2 — and a run cut short would then hold neither a
# complete confirmatory ladder nor a complete extension. Waiting means the
# confirmatory release is finishable at every moment, and the extension is
# strictly additive.
set -eu
cd "$(dirname "$0")/.."
OUT=results/backprop-neat-v6
SHARDS=${SHARDS:-4}

while true; do
  if [ -f "$OUT/manifest.json" ] && .venv/bin/python -c "
import json, sys
m = json.load(open('$OUT/manifest.json'))
sys.exit(0 if m.get('complete') else 1)
"; then
    break
  fi
  sleep 120
done

echo "confirmatory ladder complete at $(date -u +%H:%M:%SZ); starting the extension rung"
./bench/v6_checkpoint.sh || true
mkdir -p logs
i=0
while [ "$i" -lt "$SHARDS" ]; do
  nohup .venv/bin/python -u -m bpneat.v6.run --out "$OUT" --extension \
      --shard-index "$i" --shard-total "$SHARDS" \
      > "logs/v6-ext-shard$i.log" 2>&1 &
  echo "extension shard $i -> logs/v6-ext-shard$i.log"
  i=$((i + 1))
done
