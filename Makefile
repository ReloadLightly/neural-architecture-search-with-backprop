# One entry point per phase. Long jobs run in the background, sharded and
# resumable; re-running a finished shard is a no-op.
PY      ?= .venv/bin/python
SHARDS  ?= 4
V3_OUT  ?= results/backprop-neat-v3
V4_OUT  ?= results/backprop-neat-v4
V5_OUT  ?= results/backprop-neat-v5
V6_OUT  ?= results/backprop-neat-v6

.PHONY: setup gates verify audit figures v3-run v3-status v3-finaltest v3-release \
        v4-run v4-status v4-bridge v4-sensitivity v4-finaltest v4-release \
        v5-run v5-status v5-finaltest v5-release \
        v6-run v6-extension v6-status v6-finaltest v6-release \
        vendor-data clean-logs

setup:
	uv venv .venv && uv pip install --python $(PY) -e ".[dev]"

## Exactly what CI runs. `python -m pytest` would also put the working
## directory on sys.path, which bare `pytest` does not — running a different
## command locally is how an import error reached main twice.
gates:
	.venv/bin/pytest -q && .venv/bin/ruff check .

## Re-extract the real tabular datasets from their bundled copies. The CSVs
## under data/tabular/ are committed, so a reader never needs this; it exists so
## that how they were made is a command rather than a description. It refuses to
## overwrite a file whose checksum has changed.
vendor-data:
	$(PY) bench/vendor_datasets.py

## Regenerate the figures the README shows, from the committed releases.
## The per-release figures come from `make vN-release`; these are the two
## scripts that read across releases.
figures:
	$(PY) bench/readme_figures.py
	$(PY) bench/portrait_figures.py

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
	@test -f $(V6_OUT)/sha256sums.txt \
	  && $(PY) -m bpneat.v6.verify --dir $(V6_OUT) \
	  || echo "v6 release not sealed yet, skipped"

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

## Launch the whole v6 confirmatory ladder in the background, $(SHARDS) shards,
## with the checkpoint timer that commits and pushes raw records and the
## watcher that starts the declared extension rung once the ladder completes.
##
## This is also the recovery command. The container is ephemeral and has been
## reclaimed mid-run; finished runs are skipped by fingerprint and the timer
## has already pushed them, so `make v6-run` after a restart picks up where the
## suite stopped. It declines to start a second set of shards.
v6-run:
	@./bench/v6_launch.sh

## The declared extension rung (16800 candidates, 8x the reference). Run only
## after `make v6-run` reports complete, and only before the sealed test. It
## costs about three times the confirmatory ladder and scores no hypothesis.
v6-extension:
	@mkdir -p logs
	@for i in $$(seq 0 $$(( $(SHARDS) - 1 )) ); do \
	  nohup $(PY) -u -m bpneat.v6.run --out $(V6_OUT) --extension \
	      --shard-index $$i --shard-total $(SHARDS) \
	      > logs/v6-ext-shard$$i.log 2>&1 & \
	  echo "shard $$i -> logs/v6-ext-shard$$i.log"; \
	done

v6-status:
	@$(PY) -m bpneat.v6.run --out $(V6_OUT) --status

v6-finaltest:
	$(PY) -m bpneat.v6.run --out $(V6_OUT) --final-test

v6-release:
	$(PY) -m bpneat.v6.run --out $(V6_OUT) --release

clean-logs:
	rm -rf logs
