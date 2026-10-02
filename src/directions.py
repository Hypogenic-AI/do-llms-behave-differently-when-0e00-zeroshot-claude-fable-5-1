"""Stage 2 (CPU): build directions (style / length / formality / eval-awareness / stated-authorship / refusal), evaluate
probes for 'LLM-written' and their transfer, and compute cosines between directions.

usage: python src/directions.py <model-key>
"""
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_TRAIN = 250


def main():
    key = sys.argv[1]
    A = np.load(os.path.join(ROOT, "data", f"acts_{key}_gpt41.npz"))
    idx = json.load(open(os.path.join(ROOT, "data", f"acts_{key}_gpt41_index.json")))
    pos = {(r["id"], r["var"]): i for i, r in enumerate(idx)}
    ds_of = {r["id"]: r["dataset"] for r in idx}
    ids = lambda ds: sorted({r["id"] for r in idx if r["dataset"] == ds}, key=lambda s: int(s.split("_")[-1]))
    wild = [i for i in ids("wildchat") if (i, "ll") in pos]
    train, test = wild[:N_TRAIN], wild[N_TRAIN:]
    L = A["last"].shape[1]
    out = {"n_layers": L - 1, "probe": {}, "cos": {}}
    dirs = {}

    def rows(id_list, vars_):
        return [pos[(i, v)] for i in id_list for v in vars_ if (i, v) in pos]

    for pt in ["mean", "last"]:
        X = A[pt].astype(np.float32)
        mu = lambda id_list, vars_: X[rows(id_list, vars_)].mean(0)
        dirs[f"style_{pt}"] = mu(train, ["ls", "ll"]) - mu(train, ["hc", "hl"])
        dirs[f"len_{pt}"] = mu(train, ["ll", "hl"]) - mu(train, ["ls", "hc"])
        dirs[f"formal_{pt}"] = mu(train, ["hf"]) - mu(train, ["hc"])
        dirs[f"resid_{pt}"] = mu(train, ["ls"]) - mu(train, ["hf"])  # LLM vs careful-human: formality-matched contrast
        dirs[f"center_{pt}"] = mu(train, ["ls", "ll", "hc", "hl"])
        dirs[f"norm_{pt}"] = np.linalg.norm(X[rows(train, ["orig"])], axis=-1).mean(0)[:, None]

        # --- probe evaluation: diff-of-means projection and logistic regression, per layer ---
        evals = {
            "wild_test_2x2": (test, ["hc", "hl"], ["ls", "ll"]),
            "wild_test_short": (test, ["hc"], ["ls"]),
            "wild_test_long": (test, ["hl"], ["ll"]),
            "wild_test_orig_vs_ls": (test, ["orig"], ["ls"]),
            "wild_test_hf_vs_ls": (test, ["hf"], ["ls"]),
            "wild_test_hc_vs_hf": (test, ["hc"], ["hf"]),
            "wild_test_len(short_vs_long)": (test, ["hc", "ls"], ["hl", "ll"]),
        }
        for ds in ["jbb_harmful", "jbb_benign", "xstest_safe", "xstest_unsafe", "tqa_sug", "tqa_plain"]:
            evals[f"{ds}_orig_vs_ls"] = (ids(ds), ["orig"], ["ls"])
            evals[f"{ds}_hc_vs_ls"] = (ids(ds), ["hc"], ["ls"])
            evals[f"{ds}_hl_vs_ll"] = (ids(ds), ["hl"], ["ll"])
        evals["mwe_hc_vs_orig(native LLM)"] = (ids("mwe_syco"), ["hc"], ["orig"])
        evals["mwe_hc_vs_ls"] = (ids("mwe_syco"), ["hc"], ["ls"])
        res = {k: {"dom": [], "lr": []} for k in evals}
        tr_neg, tr_pos = rows(train, ["hc", "hl"]), rows(train, ["ls", "ll"])
        for l in range(L):
            d = dirs[f"style_{pt}"][l]
            clf = None
            if l % 4 == 0:
                Xtr = np.concatenate([X[tr_neg, l], X[tr_pos, l]])
                m, s = Xtr.mean(0), Xtr.std(0) + 1e-6
                clf = LogisticRegression(C=0.01, max_iter=300).fit((Xtr - m) / s, [0] * len(tr_neg) + [1] * len(tr_pos))
            for k, (id_list, nv, pv) in evals.items():
                rn, rp = rows(id_list, nv), rows(id_list, pv)
                y = [0] * len(rn) + [1] * len(rp)
                Xe = np.concatenate([X[rn, l], X[rp, l]])
                res[k]["dom"].append(float(roc_auc_score(y, Xe @ d)))
                res[k]["lr"].append(float(roc_auc_score(y, clf.decision_function((Xe - m) / s))) if clf is not None else None)
        out["probe"][pt] = res
        # --- dedicated probes trained on a specific contrast (train split) and tested on the held-out split ---
        ded = {}
        for nv, pv in [("hf", "ls"), ("orig", "ls"), ("hc", "hf"), ("hc", "ls")]:
            aucs, dom = [], []
            for l in range(L):
                rn, rp, tn, tp = rows(train, [nv]), rows(train, [pv]), rows(test, [nv]), rows(test, [pv])
                yte = [0] * len(tn) + [1] * len(tp)
                Xte = np.concatenate([X[tn, l], X[tp, l]])
                dom.append(float(roc_auc_score(yte, Xte @ (X[rp, l].mean(0) - X[rn, l].mean(0)))))
                if l % 4 == 0:
                    Xtr = np.concatenate([X[rn, l], X[rp, l]])
                    m, s = Xtr.mean(0), Xtr.std(0) + 1e-6
                    clf = LogisticRegression(C=0.01, max_iter=300).fit((Xtr - m) / s, [0] * len(rn) + [1] * len(rp))
                    aucs.append(float(roc_auc_score(yte, clf.decision_function((Xte - m) / s))))
                else:
                    aucs.append(None)
            ded[f"{nv}_vs_{pv}"] = {"lr": aucs, "dom": dom}
            print(pt, f"dedicated probe {nv} vs {pv}: LR", [round(a, 3) for a in aucs if a is not None], "DoM mid", round(dom[L // 2], 3), flush=True)
        out["dedicated"] = out.get("dedicated", {})
        out["dedicated"][pt] = ded

    # --- contrast directions (last token) ---
    C = np.load(os.path.join(ROOT, "data", f"acts_contrast_{key}.npz"))["last"].astype(np.float32)
    ci = json.load(open(os.path.join(ROOT, "data", f"acts_contrast_{key}_index.json")))
    for fam, nm in [("eval", "eval"), ("auth", "stated")]:
        p = [i for i, r in enumerate(ci) if r["fam"] == fam and r["label"] == 1]
        n = [i for i, r in enumerate(ci) if r["fam"] == fam and r["label"] == 0]
        dirs[f"{nm}_last"] = C[p].mean(0) - C[n].mean(0)
        # per-pair directions, to check consistency across phrasings
        pd_ = []
        for k in range(4):
            pk = [i for i, r in enumerate(ci) if r["fam"] == fam and r["label"] == 1 and r["pair"] == k]
            nk = [i for i, r in enumerate(ci) if r["fam"] == fam and r["label"] == 0 and r["pair"] == k]
            pd_.append(C[pk].mean(0) - C[nk].mean(0))
        cs = [[float(np.mean([cos(pd_[a][l], pd_[b][l]) for a in range(4) for b in range(a + 1, 4)])) for l in range(L)]]
        out["cos"][f"{nm}_pairwise_consistency"] = cs[0]
    Xl = A["last"].astype(np.float32)
    dirs["refusal_last"] = Xl[rows(ids("jbb_harmful"), ["orig"])].mean(0) - Xl[rows(ids("jbb_benign"), ["orig"])].mean(0)

    names = ["style_last", "style_mean", "resid_last", "len_last", "formal_last", "eval_last", "stated_last", "refusal_last"]
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            out["cos"][f"{names[a]}|{names[b]}"] = [float(cos(dirs[names[a]][l], dirs[names[b]][l])) for l in range(L)]

    # --- do the eval / stated-authorship / refusal directions "fire" on LLM-styled prompts? (paired, last token, held-out) ---
    proj = {}
    for nm in ["eval_last", "stated_last", "refusal_last", "style_last", "resid_last"]:
        proj[nm] = {}
        for lab, (id_list, nv, pv) in {"wild_test_hc_vs_ls": (test, "hc", "ls"), "wild_test_hl_vs_ll": (test, "hl", "ll"),
                                       "wild_test_hc_vs_hl(len)": (test, "hc", "hl"), "wild_test_hc_vs_hf(formal)": (test, "hc", "hf"), "wild_test_hf_vs_ls": (test, "hf", "ls"),
                                       "wild_test_orig_vs_ls": (test, "orig", "ls")}.items():
            aucs, dz = [], []
            for l in range(L):
                d = dirs[nm][l]
                a = Xl[[pos[(i, nv)] for i in id_list], l] @ d
                b = Xl[[pos[(i, pv)] for i in id_list], l] @ d
                aucs.append(float(roc_auc_score([0] * len(a) + [1] * len(b), np.concatenate([a, b]))))
                dz.append(float((b - a).mean() / ((b - a).std() + 1e-9)))
            proj[nm][lab] = {"auc": aucs, "paired_dz": dz}
    out["proj"] = proj
    out["style_norm_ratio"] = {pt: [float(np.linalg.norm(dirs[f"style_{pt}"][l]) / dirs[f"norm_{pt}"][l, 0]) for l in range(L)] for pt in ["mean", "last"]}
    np.savez(os.path.join(ROOT, "data", f"dirs_{key}.npz"), **dirs)
    json.dump(out, open(os.path.join(ROOT, "results", f"probes_{key}.json"), "w"), indent=1)
    best = int(np.argmax(out["probe"]["mean"]["wild_test_2x2"]["dom"]))
    for pt in ["mean", "last"]:
        r = out["probe"][pt]
        print(pt, "DoM AUC by layer (wild test 2x2):", np.round(r["wild_test_2x2"]["dom"], 3))
        for k in r:
            print(f"  {pt} {k}: DoM mid={r[k]['dom'][L // 2]:.3f} LR mid={r[k]['lr'][L // 2 - (L // 2) % 4]}")
    for k, v in out["cos"].items():
        print("cos", k, np.round(v[L // 2], 3), "max|.|", np.round(np.max(np.abs(v[2:])), 3))
    for nm in proj:
        for lab in proj[nm]:
            print("proj", nm, lab, "AUC mid", round(proj[nm][lab]["auc"][L // 2], 3), "dz", round(proj[nm][lab]["paired_dz"][L // 2], 2))
    print("style norm / resid norm:", np.round(out["style_norm_ratio"]["mean"], 3))


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


if __name__ == "__main__":
    main()
