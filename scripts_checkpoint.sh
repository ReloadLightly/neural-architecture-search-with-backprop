#!/bin/sh
# Commit and push completed v3 run records at intervals while the suite runs.
# Records are written atomically, so any file visible on disk is complete.
# Derived tables and checksums are NOT produced here: they are sealed once, by
# `make v3-release`, after the suite finishes and the sealed test is opened.
cd /home/user/backprop-neat-summer || exit 1
while :; do
  sleep "${1:-1200}"
  N=$(ls results/backprop-neat-v3/raw/runs/*.json 2>/dev/null | wc -l)
  if ! git diff --quiet HEAD -- results/backprop-neat-v3 2>/dev/null || \
     [ -n "$(git ls-files --others --exclude-standard results/backprop-neat-v3)" ]; then
    git add results/backprop-neat-v3 >/dev/null 2>&1
    git -c user.email=roland.loechli@googlemail.com -c user.name="Roland Löchli" \
        commit -q -m "v3 checkpoint: $N/1920 runs

Raw records only; tables, figures and checksums are sealed once by
make v3-release after the sealed test.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GP1bnJXe8xy12aBhsum7ab" >/dev/null 2>&1
    git push origin main >/dev/null 2>&1 && echo "$(date -u +%H:%M) pushed $N/1920"
  fi
  [ "$N" -ge 1920 ] && break
done
