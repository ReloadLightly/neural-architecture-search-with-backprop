"""Head-to-head: Ha's exact break rule vs bounded settling, same seeds."""
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bpneat.datasets import make_bundle
from bpneat.evolve import SearchConfig, search
from bpneat.baselines import make_mlp
from bpneat.learn import train, accuracy

print(f"{'task':8s} {'mode':10s} {'val_acc':>8} {'repr n/c':>10} {'causal n/c':>11} "
      f"{'cands':>6} {'min':>6} {'s/cand':>7}")
for task, gens in (("xor", 8), ("circle", 8), ("spiral", 12)):
    b = make_bundle(task, seed=7103)
    for mode, settle in (("ha-exact", False), ("settled", True)):
        cfg = SearchConfig(task=task, generations=gens, population=50, propagation=("settled" if settle else "ha2016"))
        t0 = time.time(); res = search(b, cfg, seed=17103); dt = time.time() - t0
        m = res.metrics
        print(f"{task:8s} {mode:10s} {m['validation_accuracy']:8.3f} "
              f"{str(m['represented_nodes'])+'/'+str(m['represented_connections']):>10} "
              f"{str(m['causal_hidden_nodes'])+'/'+str(m['causal_connections']):>11} "
              f"{res.candidates:6d} {dt/60:6.2f} {dt/res.candidates:7.3f}", flush=True)
    # Fixed-MLP control under each mode.
    for mode, settle in (("ha-exact", False), ("settled", True)):
        g = make_mlp((32, 32), np.random.default_rng(1))
        r = train(g, np.array(g.weight), b.train.X, b.train.y,
                  np.random.default_rng(2), n_cycles=600, propagation=("settled" if settle else "ha2016"))
        print(f"{task:8s} MLP-{mode:6s} {accuracy(g, r.weights, b.validation.X, b.validation.y, settle):8.3f}",
              flush=True)
