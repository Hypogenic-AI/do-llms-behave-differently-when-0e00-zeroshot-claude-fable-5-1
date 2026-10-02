"""Behavioural replication on API models (no activation access): same prompts, same schema as run_behave.py.

usage: python src/run_api.py <openrouter-model> <short-name>
"""
import json
import os
import sys

import numpy as np

from api import COST, ROOT, chat, pmap
from lm_prompts import PAIR_Q_COT, parse_ab
from run_behave_common import LABELS, MWE_TAIL, load_items, user_text

model, short = sys.argv[1], sys.argv[2]
items = load_items()
rows = []
for it in items:
    if it["dataset"] in ("wildchat", "tqa_plain"):
        continue
    for var in it["variants"]:
        rows.append(dict(id=it["id"], dataset=it["dataset"], var=var, user=user_text(it, var)))
    for lab in ["lab_ai", "lab_human"]:
        rows.append(dict(id=it["id"], dataset=it["dataset"], var=lab, user=LABELS[lab] + "\n\n" + user_text(it, "orig")))


def gen(r):
    mt = 6 if r["dataset"] == "mwe_syco" else 150
    return chat(model, [{"role": "user", "content": r["user"]}], max_tokens=mt, tag="beh1")


outs = pmap(gen, rows, workers=16, desc="gen")
byid = {it["id"]: it for it in items}
for r, o in zip(rows, outs):
    if r["dataset"] == "mwe_syco":
        m = byid[r["id"]]["meta"]
        o = o or ""
        r["mwe_raw"] = o
        if m["match"] in o and m["nomatch"] not in o:
            r["syco_logit"] = 1.0
        elif m["nomatch"] in o and m["match"] not in o:
            r["syco_logit"] = -1.0
    else:
        r["response"] = o if o is not None else ""
with open(os.path.join(ROOT, "results", f"behave_{short}_gpt41.jsonl"), "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

# --- 2AFC authorship judgement on WildChat (held-out 145 items) ---
wild = [it for it in items if it["dataset"] == "wildchat" and "ll" in it["variants"]][250:]
PAIRS = [("hc", "ls"), ("hl", "ll"), ("orig", "ls"), ("hf", "ls"), ("hc", "hf")]
jobs = [(it, a, b, order) for it in wild for a, b in PAIRS for order in (0, 1)]


def afc(job):
    it, a, b, order = job
    x, y = (it["variants"][a], it["variants"][b]) if order == 0 else (it["variants"][b], it["variants"][a])
    o = chat(model, [{"role": "user", "content": PAIR_Q_COT.format(a=x, b=y)}], max_tokens=150, tag="afc2") or ""
    pick = parse_ab(o)
    if pick is None:
        return None
    return float((pick == "B") == (order == 0))  # 1 if `b` (second-named variant) was picked as AI


res = pmap(afc, jobs, workers=16, desc="2afc")
out = {}
for a, b in PAIRS:
    v = [r for (it, x, y, o), r in zip(jobs, res) if (x, y) == (a, b) and r is not None]
    out[f"{a}_vs_{b}"] = dict(p_second_ai=float(np.mean(v)), n=len(v))
    print(f"2AFC {a} vs {b}: P({b} picked as AI)={np.mean(v):.3f} n={len(v)}")
json.dump(out, open(os.path.join(ROOT, "results", f"pair2afc_{short}_summary.json"), "w"), indent=1)
print(COST)
