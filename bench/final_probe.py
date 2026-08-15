import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bpneat.datasets import make_bundle
from bpneat.evolve import SearchConfig, search
from bpneat.baselines import make_mlp
from bpneat.learn import train, accuracy
from bpneat.genome import settle_ticks

for task, gens in (("xor", 6), ("circle", 6), ("spiral", 10)):
    b = make_bundle(task, seed=7103)
    # Fixed-MLP control under the same learner.
    rng = np.random.default_rng(1)
    g = make_mlp((32, 32), rng)
    t0 = time.time()
    r = train(g, np.array(g.weight), b.train.X, b.train.y, np.random.default_rng(2), n_cycles=600)
    mlp_va = accuracy(g, r.weights, b.validation.X, b.validation.y)
    mlp_t = time.time() - t0

    cfg = SearchConfig(task=task, generations=gens, population=50)
    t0 = time.time()
    res = search(b, cfg, seed=17103)
    dt = time.time() - t0
    m = res.metrics
    print(f"{task:7s} | NEAT val {m['validation_accuracy']:.3f} "
          f"repr {m['represented_nodes']:3d}n/{m['represented_connections']:3d}c "
          f"causal {m['causal_hidden_nodes']:2d}n/{m['causal_connections']:3d}c "
          f"| {res.candidates:4d} cands {dt/60:5.2f}min ({dt/res.candidates:.3f}s/cand) "
          f"| MLP32x32 val {mlp_va:.3f} ({mlp_t:.1f}s)", flush=True)
    print(f"        ops={m['causal_operators']}", flush=True)
