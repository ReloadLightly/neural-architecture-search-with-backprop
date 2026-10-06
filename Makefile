# One entry point per phase. Long jobs run in the background, sharded and
# resumable; re-running a finished shard is a no-op.
PY      ?= .venv/bin/python
SHARDS  ?= 4
V3_OUT  ?= results/backprop-neat-v3
V4_OUT  ?= results/backprop-neat-v4
V5_OUT  ?= results/backprop-neat-v5

.PHONY: setup gates verify audit v3-run v3-status v3-finaltest v3-release \
        v4-run v4-status v4-bridge v4-sensitivity v4-finaltest v4-release \
        v5-run v5-status v5-finaltest v5-release clean-logs

setup:
	uv venv .venv && uv pip install --python $(PY) -e ".[dev]"

gates:
	$(PY) -m pytest tests/ -q && .venv/bin/ruff check .

## Every committed release must regenerate from its own raw records.
verify:
	.venv/bin/bpneat verify --dir results/backprop-neat-v2/track-a
	.venv/bin/bpneat verify --dir results/backprop-neat-v2/track-b
	$(PY) -m bpneat.v3.verify --dir $(V3_OUT)
	@test -f $(V4_OUT)/sha256sums.txt \
	  && $(PY) -m bpneat.v4.verify --dir $(V4_OUT) \
	  || echo "v4 release not sealed yet, skipped"
	@test -f $(V5_OUT)/sha256sums.txt \
	  && $(PY) -m bpneat.v5.verify --dir $(V5_OUT) \
	  || echo "v5 release not sealed yet, skipped"

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

## Launch the whole v4 confirmatory suite in the background, $(SHARDS) shards.
## Safe to re-run: completed cells are skipped by fingerprint.
v4-run:
	@mkdir -p logs
	@for i in $$(seq 0 $$(( $(SHARDS) - 1 )) ); do \
	  nohup $(PY) -u -m bpneat.v4.run --out $(V4_OUT) \
	      --shard-index $$i --shard-total $(SHARDS) \
	      > logs/v4-shard$$i.log 2>&1 & \
	  echo "shard $$i -> logs/v4-shard$$i.log"; \
	done

v4-status:
	@$(PY) -m bpneat.v4.run --out $(V4_OUT) --status

## Reproducibility gate: v3's backprop_neat re-run on v3's own seeds.
v4-bridge:
	$(PY) -m bpneat.v4.run --out $(V4_OUT) --bridge

## Measure the protocol's sensitivity to numerically irrelevant perturbations.
v4-sensitivity:
	$(PY) bench/sensitivity_v4.py

v4-finaltest:
	$(PY) -m bpneat.v4.run --out $(V4_OUT) --final-test

v4-release:
	$(PY) -m bpneat.v4.run --out $(V4_OUT) --release

## Launch the whole v5 confirmatory suite in the background, $(SHARDS) shards.
v5-run:
	@mkdir -p logs
	@for i in $$(seq 0 $$(( $(SHARDS) - 1 )) ); do \
	  nohup $(PY) -u -m bpneat.v5.run --out $(V5_OUT) \
	      --shard-index $$i --shard-total $(SHARDS) \
	      > logs/v5-shard$$i.log 2>&1 & \
	  echo "shard $$i -> logs/v5-shard$$i.log"; \
	done

v5-status:
	@$(PY) -m bpneat.v5.run --out $(V5_OUT) --status

v5-finaltest:
	$(PY) -m bpneat.v5.run --out $(V5_OUT) --final-test

v5-release:
	$(PY) -m bpneat.v5.run --out $(V5_OUT) --release

clean-logs:
	rm -rf logs
