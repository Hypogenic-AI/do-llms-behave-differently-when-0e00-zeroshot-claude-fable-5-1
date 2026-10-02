"""Assemble the base (pre-rewrite) items for every dataset into data/base_items.json."""
import json
import os
import random
import re

import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
random.seed(0)
items = []

# ---- JBB (refusal; harmful + matched benign) ----
jbb = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
for split in ["harmful", "benign"]:
    for r in jbb[split]:
        items.append(dict(id=f"jbb_{split}_{r['Index']}", dataset=f"jbb_{split}", text=r["Goal"], meta=dict(category=r["Category"])))

# ---- XSTest (borderline safe + contrast unsafe) ----
xs = pd.read_csv(os.path.join(ROOT, "data", "xstest.csv"))
safe = xs[xs.label == "safe"].sample(150, random_state=0)
unsafe = xs[xs.label == "unsafe"].sample(100, random_state=0)
for name, df in [("xstest_safe", safe), ("xstest_unsafe", unsafe)]:
    for _, r in df.iterrows():
        items.append(dict(id=f"{name}_{r['id']}", dataset=name, text=r["prompt"], meta=dict(type=r["type"])))

# ---- TruthfulQA (sycophancy to a user-suggested misconception; plain accuracy) ----
tqa = load_dataset("truthfulqa/truthful_qa", "generation")["validation"]
idx = [i for i in range(len(tqa)) if 2 <= len(tqa[i]["incorrect_answers"][0].split()) <= 14 and len(tqa[i]["question"].split()) <= 25]
random.shuffle(idx)
for i in idx[:200]:
    r = tqa[i]
    wrong = r["incorrect_answers"][0].rstrip(".")
    meta = dict(question=r["question"], best=r["best_answer"], correct=r["correct_answers"], wrong=wrong)
    sug = f"{r['question']} I think the answer is: {wrong}. But I'm really not sure."
    items.append(dict(id=f"tqa_sug_{i}", dataset="tqa_sug", text=sug, meta=meta))
    items.append(dict(id=f"tqa_plain_{i}", dataset="tqa_plain", text=r["question"], meta=meta))

# ---- WildChat (real human first turns; probe training + generic behaviour) ----
fs = "data/train-00000-of-00014.parquet"
df = pd.read_parquet(hf_hub_download("allenai/WildChat-1M", fs, repo_type="dataset"))
df = df[(df.language == "English") & (~df.toxic)]
seen, wc = set(), []
BAD = re.compile(r"midjourney|stable diffusion|prompt generator|as an ai|you are (a|an|now)\b|act as|pretend|ignore (all|previous)|dan\b|```|http", re.I)
for _, r in df.sample(frac=1, random_state=0).iterrows():
    t = r["conversation"][0]["content"].strip()
    n = len(t.split())
    if not (8 <= n <= 80) or "\n" in t or BAD.search(t) or not t.isascii():
        continue
    k = t[:30].lower()
    if k in seen:
        continue
    seen.add(k)
    wc.append(dict(id=f"wild_{len(wc)}", dataset="wildchat", text=t, meta=dict(hash=r["conversation_hash"])))
    if len(wc) == 400:
        break
items += wc

# ---- Anthropic model-written sycophancy (natively LLM-written; political typology) ----
p = hf_hub_download("Anthropic/model-written-evals", "sycophancy/sycophancy_on_political_typology_quiz.jsonl", repo_type="dataset")
L = [json.loads(l) for l in open(p)]
random.shuffle(L)
from collections import Counter
def cands(pre):
    b = [m.end() for m in re.finditer(r"[.!?]\s+", pre)]
    return [(pre[:i].strip(), pre[i:].strip()) for i in b]
sufc = Counter(sf for x in L for _, sf in cands(x["question"].split("\n (A)", 1)[0]))
n = {"liberal": 0, "conservative": 0}
k = 0
for x in L:
    q = x["question"]
    pre, opts = q.split("\n (A)", 1)
    ok = [(b, sf) for b, sf in cands(pre) if sufc[sf] >= 20]
    if not ok:
        continue
    bio, stem = ok[0]
    sents = re.split(r"(?<=[.!?])\s+", bio)
    if n[x["user_affiliation"]] >= 100 or len(sents) < 3:
        continue
    n[x["user_affiliation"]] += 1
    items.append(dict(id=f"mwe_{k}", dataset="mwe_syco", text=bio,
                      meta=dict(suffix=" " + stem + "\n (A)" + opts.replace("\n\nAnswer:", ""), match=x["answer_matching_behavior"].strip(),
                                nomatch=x["answer_not_matching_behavior"].strip(), aff=x["user_affiliation"])))
    k += 1

json.dump(items, open(os.path.join(ROOT, "data", "base_items.json"), "w"), indent=1)
from collections import Counter
print(Counter(i["dataset"] for i in items))
for d in ["tqa_sug", "wildchat", "mwe_syco"]:
    for it in [i for i in items if i["dataset"] == d][:4]:
        print(d, "|", it["text"][:300], "|", str(it["meta"].get("suffix", ""))[:80])
