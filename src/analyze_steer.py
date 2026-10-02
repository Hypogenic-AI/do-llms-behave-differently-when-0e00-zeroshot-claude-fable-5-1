"""Steering analysis: behaviour under each steering condition vs the unsteered baseline (paired item bootstrap), plus
a within-item dose-response check relating the shift in the internal 'LLM-style' projection to the behaviour shift.

usage: python src/analyze_steer.py [model ...]
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from analyze_behave import REF_RE, ROOT, load

GROUPS = {"harmful": ["jbb_harmful", "xstest_unsafe"], "borderline": ["jbb_benign", "xstest_safe"], "syco": ["tqa_sug"]}


def main():
    models = sys.argv[1:] or ["qwen", "llama", "gemma"]
    rng = np.random.default_rng(0)
    out = {}
    for m in models:
        R = pd.DataFrame([json.loads(l) for l in open(os.path.join(ROOT, "results", f"steer_{m}.jsonl"))])
        meta = json.load(open(os.path.join(ROOT, "results", f"steer_{m}_meta.json")))
        j = R.judge.where(R.judge != "INVALID", R.response.fillna("").map(lambda t: "REFUSE" if REF_RE.match(t) else "COMPLY"))
        R["y"] = np.where(R.dataset == "tqa_sug", j == "AGREE", j == "REFUSE").astype(float)
        R["other"] = (j == "OTHER").astype(float)
        res = {}
        for c in meta["conds"]:
            res[c] = dict(meta["conds"][c])
            for g, dss in GROUPS.items():
                W = R[R.dataset.isin(dss)].pivot(index="id", columns="cond", values="y")
                d = (W[c] - W["base"]).values
                boot = d[rng.integers(0, len(d), (4000, len(d)))].mean(1)
                res[c][g] = dict(rate=float(W[c].mean()), diff=float(d.mean()), lo=float(np.percentile(boot, 2.5)),
                                 hi=float(np.percentile(boot, 97.5)), n=int(len(d)))
        Dn = np.load(os.path.join(ROOT, "data", f"dirs_{m}.npz"))
        nat = {k: float(np.linalg.norm(Dn[k][meta["layer"]]) / meta["s"]) for k in
               ["style_last", "style_mean", "formal_last", "resid_last", "len_last", "eval_last", "stated_last", "refusal_last"]}
        out[m] = dict(layer=meta["layer"], s=meta["s"], resid_norm=meta["resid_norm"], A=meta["A"], conds=res, natural_norm_over_s=nat)
        print("  natural norms / s:", {k: round(v, 2) for k, v in nat.items()})
        print("=====", m, "layer", meta["layer"], "s/resid_norm", round(meta["s"] / meta["resid_norm"], 3))
        for c, r in res.items():
            print(f"  {c:18s} P(AI)={r['p_ai']:.3f} NLL={r['nll']:.2f} | harmful {r['harmful']['rate']:.3f} ({r['harmful']['diff']:+.3f}) "
                  f"| borderline {r['borderline']['rate']:.3f} ({r['borderline']['diff']:+.3f}) | syco {r['syco']['rate']:.3f} ({r['syco']['diff']:+.3f}) "
                  f"| mwe {r['mwe_orig_syco']:.3f}/{r['mwe_hc_syco']:.3f}")

        # ---- within-item dose-response: does a larger shift along the style direction predict a larger behaviour shift? ----
        B = load(m)
        A = np.load(os.path.join(ROOT, "data", f"acts_{m}_gpt41.npz"))["last"][:, meta["layer"]].astype(np.float32)
        idx = json.load(open(os.path.join(ROOT, "data", f"acts_{m}_gpt41_index.json")))
        D = np.load(os.path.join(ROOT, "data", f"dirs_{m}.npz"))
        d = D["style_last"][meta["layer"]]
        proj = {(r["id"], r["var"]): float(A[i] @ d / np.linalg.norm(d) ** 2) for i, r in enumerate(idx)}
        B["proj"] = [proj.get((i, v), np.nan) for i, v in zip(B.id, B["var"])]
        dr = {}
        for g, dss in {"refusal (all)": sum([GROUPS["harmful"], GROUPS["borderline"]], []), "sycophancy (TQA)": ["tqa_sug"]}.items():
            sub = B[B.dataset.isin(dss)]
            Y = sub.pivot(index="id", columns="var", values="y")
            P = sub.pivot(index="id", columns="var", values="proj")
            for a, b in [("hc", "ls"), ("hl", "ll"), ("hc", "hf"), ("hf", "ls"), ("orig", "ls")]:
                ok = Y[[a, b]].notna().all(axis=1)
                dy, dp = (Y[b] - Y[a])[ok].values, (P[b] - P[a])[ok].values
                r = float(np.corrcoef(dy, dp)[0, 1]) if dy.std() > 0 else float("nan")
                perm = [abs(np.corrcoef(rng.permutation(dy), dp)[0, 1]) for _ in range(2000)] if dy.std() > 0 else [1]
                dr[f"{g}|{b}-{a}"] = dict(r=r, p=float((np.sum(np.array(perm) >= abs(r)) + 1) / (len(perm) + 1)), n=int(ok.sum()),
                                          mean_dproj=float(dp.mean()))
                print(f"  dose-response {g:18s} {b}-{a}: mean dproj={dp.mean():+.2f} corr(dproj, dy)={r:+.3f} p={dr[f'{g}|{b}-{a}']['p']:.3f} n={ok.sum()}")
        out[m]["dose_response"] = dr
    prev = {}
    path = os.path.join(ROOT, "results", "steer_summary.json")
    if os.path.exists(path):
        prev = json.load(open(path))
    prev.update(out)
    json.dump(prev, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
