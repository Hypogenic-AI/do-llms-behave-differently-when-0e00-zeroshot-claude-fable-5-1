"""Robustness: re-judge all TruthfulQA-suggestion responses of the local models with a second judge (Gemini 2.5 Flash)
and recompute the sycophancy contrasts."""
import json
import os
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

import judge as J
from analyze_behave import CONTRASTS, paired
from api import COST, ROOT, chat, pmap

SECOND = "google/gemini-2.5-flash"
base = {i["id"]: i for i in json.load(open(os.path.join(ROOT, "data", "base_items.json")))}
out = {}
rng = np.random.default_rng(0)
for m in ["qwen", "llama", "gemma"]:
    R = [json.loads(l) for l in open(os.path.join(ROOT, "results", f"behave_{m}_gpt41.jsonl"))]
    R = [r for r in R if r["dataset"] == "tqa_sug"]

    def job(r):
        it = base[r["id"]]
        p = J.SYCO.format(q=it["meta"]["question"], wrong=it["meta"]["wrong"], best=it["meta"]["best"], resp=r["response"])
        o = chat(SECOND, [{"role": "user", "content": p}], max_tokens=200, tag="jv1")
        return float("AGREE" in o.upper() and "DISAGREE" not in o.upper()) if o else np.nan

    labs = pmap(job, R, workers=16, desc=m)
    df = pd.DataFrame(R)
    df["y"] = labs
    W = df.pivot(index="id", columns="var", values="y")
    out[m] = {"rates": {v: float(W[v].mean()) for v in W.columns}, "contrasts": {c: paired(W, p, q, rng) for c, (p, q) in CONTRASTS.items()}}
    print(m, {k: round(v, 3) for k, v in out[m]["rates"].items()})
    for c, r in out[m]["contrasts"].items():
        if r:
            print(f"   {c:36s} diff={r['diff']:+.3f} [{r['lo']:+.3f},{r['hi']:+.3f}] p={r['p']:.3f}")
json.dump(out, open(os.path.join(ROOT, "results", "syco_second_judge.json"), "w"), indent=1)
print(COST)
