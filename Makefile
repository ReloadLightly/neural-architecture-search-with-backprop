# One entry point per phase. Long jobs run in the background, sharded and
# resumable; re-running a finished shard is a no-op.
PY      ?= .venv/bin/python
SHARDS  ?= 4
V3_OUT  ?= results/backprop-neat-v3

.PHONY: setup gates verify audit v3-run v3-status v3-finaltest v3-release clean-logs

setup:
	uv venv .venv && uv pip install --python $(PY) -e ".[dev]" scipy

gates:
	$(PY) -m pytest tests/ -q && .venv/bin/ruff check .

verify:
	.venv/bin/bpneat verify --dir results/backprop-neat-v2/track-a
	.venv/bin/bpneat verify --dir results/backprop-neat-v2/track-b

audit:
	$(PY) bench/audit_2026_10.py

## Launch the whole v3 confirmatory suite in the background, $(SHARDS) shards.
## Safe to re-run: completed cells are skipped by fingerprint.
v3-run:
	@mkdir -p logs
	@for i in $$(seq 0 $$(( $(SHARDS) - 1 )) ); do \
	  nohup $(PY) -u -m bpneat.v3.run --out $(V3_OUT) \
	      --shard-index $$i --shard-total $(SHARDS) \
	      > logs/v3-shard$$i.log 2>&1 & \
	  echo "shard $$i -> logs/v3-shard$$i.log"; \
	done

v3-status:
	@$(PY) -m bpneat.v3.run --out $(V3_OUT) --status

v3-finaltest:
	$(PY) -m bpneat.v3.run --out $(V3_OUT) --final-test

v3-release:
	$(PY) -m bpneat.v3.run --out $(V3_OUT) --release

clean-logs:
	rm -rf logs
