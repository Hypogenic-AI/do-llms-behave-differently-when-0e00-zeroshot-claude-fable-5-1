"""Stage 3b: activation steering with the prompt text held exactly fixed (original human prompts).

All steering vectors at a layer are expressed in units of s = ||style_last direction|| at that layer, i.e. the natural
distance between the mean LLM-style and mean casual-human-style activation. Controls: random directions, and the
formality / length / residual-LLM / eval-awareness / stated-authorship / refusal directions, all norm-matched.

usage: python src/run_steer.py <model-key> <layer> [A]
"""
import json
import os
import sys

import numpy as np

from lm import LM, ROOT
from run_behave import load_items, user_text

key, layer = sys.argv[1], int(sys.argv[2])
A = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
lm = LM(key)
D = np.load(os.path.join(ROOT, "data", f"dirs_{key}.npz"))
unit = lambda v: v / np.linalg.norm(v)
s = float(np.linalg.norm(D["style_last"][layer]))
rng = np.random.default_rng(0)

conds = [("base", None)]
for a in [-2 * A, -A, -A / 2, A / 2, A, 2 * A]:
    conds.append((f"style_last@{a:+g}", a * s * unit(D["style_last"][layer])))
# residual-LLM direction with the formality and length directions projected out
r = D["resid_last"][layer].copy()
for other in ["formal_last", "len_last"]:
    u = unit(D[other][layer])
    r = r - (r @ u) * u
fam = {nm: D[nm][layer] for nm in ["style_mean", "formal_last", "resid_last", "len_last", "eval_last", "stated_last", "refusal_last"]}
fam["resid_orth"] = r
for nm, v in fam.items():
    for a in [-A, A]:
        conds.append((f"{nm}@{a:+g}", a * s * unit(v)))
for a in [A, 2 * A]:
    for sd in range(3):
        v = rng.standard_normal(D["style_last"][layer].shape).astype(np.float32)
        conds.append((f"random{sd}@{a:+g}", a * s * unit(v)))

items = load_items()
gen_items = [it for it in items if it["dataset"].startswith(("jbb", "xstest")) or it["dataset"] == "tqa_sug"]
mwe = [it for it in items if it["dataset"] == "mwe_syco"]
wild = [it for it in items if it["dataset"] == "wildchat" and "ll" in it["variants"]][250:310]
beh = {(r_["id"], r_["var"]): r_ for r_ in map(json.loads, open(os.path.join(ROOT, "results", f"behave_{key}_gpt41.jsonl")))}
wmsgs = [it["variants"]["hc"] for it in wild]
wresp = [beh[(it["id"], "hc")]["response"] for it in wild]

path = os.path.join(ROOT, "results", f"steer_{key}.jsonl")
meta_path = os.path.join(ROOT, "results", f"steer_{key}_meta.json")
meta = dict(layer=layer, s=s, A=A, resid_norm=float(D["norm_last"][layer, 0]), conds={})
# internal manipulation check: how much of the natural casual-human -> LLM gap along the style direction is closed downstream?
down = sorted({min(lm.n_layers, layer + k) for k in (0, 2, 6)} | {lm.n_layers})
wls = [it["variants"]["ls"] for it in wild]
gap_dirs = {l: unit(D["style_last"][l]) for l in down}
h0, _ = lm.acts(wmsgs)
h1, _ = lm.acts(wls)
p0 = {l: float((h0[:, l].astype(np.float32) @ gap_dirs[l]).mean()) for l in down}
p1 = {l: float((h1[:, l].astype(np.float32) @ gap_dirs[l]).mean()) for l in down}
f = open(path, "w")
for name, vec in conds:
    lm.clear()
    if vec is not None:
        lm.set_steer(layer, vec)
    m = dict(p_ai=float(lm.p_ai(wmsgs).mean()), nll=float(lm.nll(wmsgs, wresp).mean()))
    hs, _ = lm.acts(wmsgs)
    m["gap_closed"] = {str(l): (float((hs[:, l].astype(np.float32) @ gap_dirs[l]).mean()) - p0[l]) / (p1[l] - p0[l]) for l in down}
    for var in ["orig", "hc"]:
        d = lm.choice_logits([user_text(it, var) for it in mwe])
        sl = [float(x if "B" in it["meta"]["match"] else -x) for it, x in zip(mwe, d)]
        m[f"mwe_{var}_syco"] = float(np.mean(np.array(sl) > 0))
        m[f"mwe_{var}_logit"] = float(np.mean(sl))
    outs = lm.generate([it["variants"]["orig"] for it in gen_items], max_new_tokens=120, bs=200)
    for it, o in zip(gen_items, outs):
        f.write(json.dumps(dict(id=it["id"], dataset=it["dataset"], var="orig", cond=name, response=o)) + "\n")
    f.flush()
    meta["conds"][name] = m
    json.dump(meta, open(meta_path, "w"), indent=1)
    print(name, m, flush=True)
lm.clear()
