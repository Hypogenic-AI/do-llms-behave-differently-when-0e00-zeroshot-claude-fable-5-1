"""Stage 3a: steering sweep. For each (direction type, layer, alpha) measure (i) the shift in the verbalised authorship
judgement P(AI) on fixed human-style prompts (manipulation check) and (ii) degradation (NLL of the unsteered responses).

usage: python src/run_sweep.py <model-key>
"""
import json
import os
import sys

import numpy as np

from lm import LM, ROOT, PAIR_Q
from run_behave import load_items

key = sys.argv[1]
lm = LM(key)
D = np.load(os.path.join(ROOT, "data", f"dirs_{key}.npz"))
items = [it for it in load_items() if it["dataset"] == "wildchat" and "ll" in it["variants"]][250:310]
beh = {(r["id"], r["var"]): r for r in map(json.loads, open(os.path.join(ROOT, "results", f"behave_{key}_gpt41.jsonl")))}
msgs = [it["variants"]["hc"] for it in items]
resp = [beh[(it["id"], "hc")]["response"] for it in items]
L = lm.n_layers
layers = sorted({max(2, round(f * L)) for f in [0.2, 0.35, 0.5, 0.65, 0.8]})
res = []
rng = np.random.default_rng(0)


def measure(tag, layer=None, vec=None, **kw):
    lm.clear()
    if vec is not None:
        lm.set_steer(layer, vec)
    r = dict(tag=tag, layer=layer, p_ai=float(lm.p_ai(msgs).mean()), nll=float(lm.nll(msgs, resp).mean()), **kw)
    lm.clear()
    res.append(r)
    print(r, flush=True)


# 2AFC authorship judgement with brief reasoning allowed (forced single-letter answers show strong response biases)
from lm_prompts import PAIR_Q_COT, parse_ab
allw = [it for it in load_items() if it["dataset"] == "wildchat" and "ll" in it["variants"]][250:]
PAIRS = [("hc", "ls"), ("hl", "ll"), ("orig", "ls"), ("hf", "ls"), ("hc", "hf")]
jobs = [(it, a, b, o) for it in allw for a, b in PAIRS for o in (0, 1)]
prompts = [PAIR_Q_COT.format(a=it["variants"][a if o == 0 else b], b=it["variants"][b if o == 0 else a]) for it, a, b, o in jobs]
outs = lm.generate(prompts, max_new_tokens=90, bs=32)
cot = {}
for a, b in PAIRS:
    v = []
    for (it, x, y, o), t in zip(jobs, outs):
        if (x, y) == (a, b) and parse_ab(t) is not None:
            v.append(float((parse_ab(t) == "B") == (o == 0)))
    cot[f"{a}_vs_{b}"] = dict(p_second_ai=float(np.mean(v)), n=len(v))
    print(f"CoT 2AFC {a} vs {b}: P({b} picked as AI)={np.mean(v):.3f} n={len(v)}", flush=True)
json.dump(cot, open(os.path.join(ROOT, "results", f"pair2afc_{key}_summary.json"), "w"), indent=1)
print("example:", outs[0][:300].replace("\n", " "), flush=True)

measure("base")
for pt in ["mean", "last"]:
    for layer in layers:
        d = D[f"style_{pt}"][layer]
        for a in [-8, -4, -2, 2, 4, 8]:
            measure(f"style_{pt}", layer, a * d, alpha=a, norm=float(np.linalg.norm(a * d)))
        for a in [4, 8]:
            for s in range(2):
                v = rng.standard_normal(d.shape).astype(np.float32)
                v = v / np.linalg.norm(v) * np.linalg.norm(a * d)
                measure(f"random_{pt}", layer, v, alpha=a, seed=s, norm=float(np.linalg.norm(v)))
json.dump(res, open(os.path.join(ROOT, "results", f"sweep_{key}.json"), "w"), indent=1)
