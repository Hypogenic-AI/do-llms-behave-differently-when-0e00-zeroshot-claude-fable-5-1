"""Stage 1: activations, verbalised authorship judgement, and unsteered behaviour for every prompt variant.

usage: python src/run_behave.py <model-key> [variants-name]
"""
import json
import os
import sys

import numpy as np

from lm import LM, ROOT

from run_behave_common import AUTH_PAIRS, EVAL_PAIRS, LABELS, MAXTOK, MWE_TAIL, load_items, user_text  # noqa: E402,F401


def main():
    key = sys.argv[1]
    name = sys.argv[2] if len(sys.argv) > 2 else "gpt41"
    items = load_items(name)
    lm = LM(key)
    rows = []
    for it in items:
        for var in it["variants"]:
            rows.append(dict(id=it["id"], dataset=it["dataset"], var=var, user=user_text(it, var)))
        if it["dataset"] != "wildchat" and it["dataset"] != "tqa_plain" and name == "gpt41":
            for lab, pre in LABELS.items():
                rows.append(dict(id=it["id"], dataset=it["dataset"], var=lab, user=pre + "\n\n" + user_text(it, "orig")))
    users = [r["user"] for r in rows]
    tag = f"{key}_{name}"

    # --- verbalised authorship judgement (style variants only) ---
    sty = [i for i, r in enumerate(rows) if not r["var"].startswith("lab_")]
    pai_path = os.path.join(ROOT, "data", f"pai_{tag}.npy")
    if os.path.exists(pai_path):
        pai = np.load(pai_path)
    else:
        pai = lm.p_ai([it_text(rows[i], items) for i in sty])
        np.save(pai_path, pai)
    for i, p in zip(sty, pai):
        rows[i]["p_ai"] = float(p)
    print("p_ai by var:", {v: round(float(np.mean([rows[i]["p_ai"] for i in sty if rows[i]["var"] == v])), 3) for v in ["orig", "hc", "hf", "ls", "ll", "hl"]}, flush=True)

    # --- paired 2AFC authorship judgement ---
    PAIRS = [("hc", "ls"), ("hl", "ll"), ("orig", "ls"), ("hf", "ls"), ("hc", "hf"), ("hc", "hl")]
    prow = []
    for it in items:
        for a, b in PAIRS:
            if a in it["variants"] and b in it["variants"]:
                prow.append(dict(id=it["id"], dataset=it["dataset"], first=a, second=b))
    byid = {it["id"]: it for it in items}
    if name != "gpt41" or os.path.exists(os.path.join(ROOT, "results", f"pair2afc_{tag}.json")):
        prow = []  # forced-letter 2AFC only for the main variants, and not recomputed if already saved
    pp = lm.pair_ai([byid[r["id"]]["variants"][r["first"]] for r in prow], [byid[r["id"]]["variants"][r["second"]] for r in prow])
    for r, p in zip(prow, pp):
        r["p_second_ai"] = float(p)
    if prow:
        json.dump(prow, open(os.path.join(ROOT, "results", f"pair2afc_{tag}.json"), "w"))
    for a, b in PAIRS:
        x = [r["p_second_ai"] for r in prow if (r["first"], r["second"]) == (a, b)]
        if x:
            print(f"2AFC {a} vs {b}: mean P({b} is AI)={np.mean(x):.3f} acc={np.mean(np.array(x) > 0.5):.3f}", flush=True)

    # --- activations ---
    if name == "gpt41":
        last, mean = lm.acts([users[i] for i in sty])
        np.savez(os.path.join(ROOT, "data", f"acts_{tag}.npz"), last=last, mean=mean)
        json.dump([dict(id=rows[i]["id"], dataset=rows[i]["dataset"], var=rows[i]["var"]) for i in sty],
                  open(os.path.join(ROOT, "data", f"acts_{tag}_index.json"), "w"))
        wild = [it for it in items if it["dataset"] == "wildchat"][:100]
        cu, ci = [], []
        for fam, pairs in [("eval", EVAL_PAIRS), ("auth", AUTH_PAIRS)]:
            for k, (pos, neg) in enumerate(pairs):
                for it in wild:
                    for lab, pre in [(1, pos), (0, neg)]:
                        cu.append(pre + "\n\n" + it["variants"]["orig"])
                        ci.append(dict(fam=fam, pair=k, label=lab, id=it["id"]))
        cl, _ = lm.acts(cu)
        np.savez(os.path.join(ROOT, "data", f"acts_contrast_{key}.npz"), last=cl)
        json.dump(ci, open(os.path.join(ROOT, "data", f"acts_contrast_{key}_index.json"), "w"))
        print("acts saved", flush=True)

    # --- MWE sycophancy logits ---
    mi = [i for i, r in enumerate(rows) if r["dataset"] == "mwe_syco"]
    if mi:
        d = lm.choice_logits([users[i] for i in mi])  # logit(B) - logit(A)
        byid = {it["id"]: it for it in items}
        for i, x in zip(mi, d):
            m = byid[rows[i]["id"]]["meta"]["match"]
            rows[i]["syco_logit"] = float(x if "B" in m else -x)

    # --- generation ---
    for pre, mt in MAXTOK.items():
        gi = [i for i, r in enumerate(rows) if r["dataset"].startswith(pre)]
        if not gi:
            continue
        outs = lm.generate([users[i] for i in gi], max_new_tokens=mt)
        for i, o in zip(gi, outs):
            rows[i]["response"] = o
        print("generated", pre, len(gi), flush=True)
        with open(os.path.join(ROOT, "results", f"behave_{tag}.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    with open(os.path.join(ROOT, "results", f"behave_{tag}.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def it_text(row, items):
    """Text shown in the authorship question: the full user message (for MWE without the answer-format tail)."""
    return row["user"].replace(MWE_TAIL, "")


if __name__ == "__main__":
    main()
