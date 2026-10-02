"""Inter-judge agreement: re-label a random sample with a second judge model (and a regex baseline for refusals)."""
import json
import os
import random

from sklearn.metrics import cohen_kappa_score

import judge as J
from analyze_behave import REF_RE
from api import COST, ROOT, chat, pmap

SECOND = "google/gemini-2.5-flash"
random.seed(0)
base = {i["id"]: i for i in json.load(open(os.path.join(ROOT, "data", "base_items.json")))}
rows = []
for m in ["qwen", "llama", "gemma"]:
    R = [json.loads(l) for l in open(os.path.join(ROOT, "results", f"behave_{m}_gpt41.jsonl"))]
    R = [r for r in R if r.get("judge") in J.VALID and r["dataset"] != "tqa_plain"]
    rows += random.sample([r for r in R if r["dataset"].startswith(("jbb", "xst"))], 150) + random.sample([r for r in R if r["dataset"] == "tqa_sug"], 80)


def job(r):
    it = base[r["id"]]
    if r["dataset"] == "tqa_sug":
        p = J.SYCO.format(q=it["meta"]["question"], wrong=it["meta"]["wrong"], best=it["meta"]["best"], resp=r["response"])
    else:
        p = J.REFUSAL.format(req=it["text"], resp=r["response"])
    out = chat(SECOND, [{"role": "user", "content": p}], max_tokens=200, tag="jv1")
    for lab in ["REFUSE", "COMPLY", "AGREE", "CORRECT", "OTHER"]:
        if out and lab in out.upper():
            return lab
    return None


labs = pmap(job, rows, workers=16, desc="second judge")
out = {}
for name, sel in [("refusal", lambda r: r["dataset"] != "tqa_sug"), ("sycophancy", lambda r: r["dataset"] == "tqa_sug")]:
    pairs = [(r["judge"], l) for r, l in zip(rows, labs) if sel(r) and l is not None]
    a, b = zip(*pairs)
    out[name] = dict(n=len(pairs), agreement=sum(x == y for x, y in pairs) / len(pairs), kappa=float(cohen_kappa_score(a, b)))
    if name == "sycophancy":
        a2, b2 = [x == "AGREE" for x in a], [x == "AGREE" for x in b]
        out[name]["agree_vs_rest_agreement"] = sum(x == y for x, y in zip(a2, b2)) / len(a2)
        out[name]["agree_vs_rest_kappa"] = float(cohen_kappa_score(a2, b2))
ref = [(r["judge"], "REFUSE" if REF_RE.match(r["response"]) else "COMPLY") for r in rows if r["dataset"] != "tqa_sug"]
out["refusal_vs_regex"] = dict(n=len(ref), agreement=sum(x == y for x, y in ref) / len(ref), kappa=float(cohen_kappa_score(*zip(*ref))))
print(json.dumps(out, indent=1), COST)
json.dump(out, open(os.path.join(ROOT, "results", "judge_validation.json"), "w"), indent=1)
