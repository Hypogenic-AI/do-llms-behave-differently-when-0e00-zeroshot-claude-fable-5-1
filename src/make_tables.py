"""Generate LaTeX tables in paper_draft/tables/ from results/*.json."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_behave_common import load_items  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper_draft", "tables")
os.makedirs(OUT, exist_ok=True)
NAMES = {"qwen": "Qwen2.5-7B", "llama": "Llama-3.1-8B", "gemma": "Gemma-2-9B", "gpt41mini": "GPT-4.1-mini", "llama70b": "Llama-3.3-70B",
         "haiku45": "Claude Haiku 4.5"}
VARS = [("orig", "original"), ("hc", "casual human"), ("hf", "formal human"), ("ls", "LLM"), ("hl", "casual human, long"), ("ll", "LLM, long")]
GROUPS = [("harmful (refusal)", "Refusal, harmful"), ("borderline-benign (refusal)", "Refusal, borderline"),
          ("TruthfulQA+suggestion (sycophancy)", "Sycophancy, TQA"), ("MWE political (sycophancy)", "Sycophancy, MWE"),
          ("TruthfulQA plain (accuracy)", "Accuracy, TQA")]


def w(name, s):
    # \bottomrule lives in the file: an \input directly before \bottomrule breaks booktabs
    open(os.path.join(OUT, name), "w").write(s + "\n\\bottomrule\n")


def load(p):
    p = os.path.join(ROOT, "results", p)
    return json.load(open(p)) if os.path.exists(p) else None


def t_data():
    items = load_items()
    ds = ["jbb_harmful", "xstest_unsafe", "jbb_benign", "xstest_safe", "tqa_sug", "tqa_plain", "mwe_syco", "wildchat"]
    lab = {"jbb_harmful": "JBB harmful", "xstest_unsafe": "XSTest unsafe", "jbb_benign": "JBB benign", "xstest_safe": "XSTest safe",
           "tqa_sug": "TruthfulQA + suggestion", "tqa_plain": "TruthfulQA plain", "mwe_syco": "MWE sycophancy (bio only)", "wildchat": "WildChat"}
    rows = []
    for d in ds:
        I = [i for i in items if i["dataset"] == d]
        nl = sum("ll" in i["variants"] for i in I)
        ws = [f"{np.mean([len(i['variants'][v].split()) for i in I if v in i['variants']]):.0f}" if any(v in i["variants"] for i in I) else "--" for v, _ in VARS]
        rows.append(f"{lab[d]} & {len(I)} & {nl} & " + " & ".join(ws) + r" \\")
    w("data.tex", "\n".join(rows))


def t_rates(S, models, name):
    lines = []
    for g, gl in GROUPS:
        first = True
        for m in models:
            r = S[m]["rates"].get(g)
            if not r:
                continue
            cells = [f"{100 * r[v]['mean']:.1f}" if v in r else "--" for v, _ in VARS]
            lines.append((gl if first else "") + f" & {NAMES[m]} & " + " & ".join(cells) + r" \\")
            first = False
        lines.append(r"\midrule")
    w(name, "\n".join(lines[:-1]))


def t_contrasts(S, models, name, contrasts):
    lines = []
    for g, gl in GROUPS[:4]:
        first = True
        for m in models:
            cells = []
            for c in contrasts:
                r = S[m]["contrasts"].get(g, {}).get(c)
                if not r:
                    cells.append("--")
                    continue
                s = f"{100 * r['diff']:+.1f} [{100 * r['lo']:+.1f}, {100 * r['hi']:+.1f}]"
                cells.append(r"\textbf{" + s + "}" if r["p"] < 0.05 else s)
            if all(c == "--" for c in cells):
                continue
            lines.append((gl if first else "") + f" & {NAMES[m]} & " + " & ".join(cells) + r" \\")
            first = False
        lines.append(r"\midrule")
    w(name, "\n".join(lines[:-1]))


def t_manip():
    lines = []
    for m in ["qwen", "llama", "gemma"]:
        P = load(f"probes_{m}.json")
        if not P:
            continue
        L = P["n_layers"]
        ded = P["dedicated"]["mean"]
        mid = (L // 2) - (L // 2) % 4
        pr = [f"{ded[k]['lr'][mid]:.2f}" for k in ["hc_vs_ls", "hc_vs_hf", "hf_vs_ls", "orig_vs_ls"]]
        A = load(f"pair2afc_{m}_summary.json") or {}
        af = [f"{100 * A[k]['p_second_ai']:.0f}" if k in A else "--" for k in ["hc_vs_ls", "hc_vs_hf", "hf_vs_ls", "orig_vs_ls"]]
        lines.append(f"{NAMES[m]} & " + " & ".join(pr) + " & " + " & ".join(af) + r" \\")
    for m in ["gpt41mini", "llama70b", "haiku45"]:
        A = load(f"pair2afc_{m}_summary.json")
        if A:
            af = [f"{100 * A[k]['p_second_ai']:.0f}" for k in ["hc_vs_ls", "hc_vs_hf", "hf_vs_ls", "orig_vs_ls"]]
            lines.append(f"{NAMES[m]} & -- & -- & -- & -- & " + " & ".join(af) + r" \\")
    w("manip.tex", "\n".join(lines))


def t_transfer():
    lines = []
    keys = [("wild_test_short", "WildChat (held-out)"), ("jbb_harmful_hc_vs_ls", "JBB harmful"), ("jbb_benign_hc_vs_ls", "JBB benign"),
            ("xstest_safe_hc_vs_ls", "XSTest safe"), ("xstest_unsafe_hc_vs_ls", "XSTest unsafe"), ("tqa_sug_hc_vs_ls", "TQA + suggestion"),
            ("mwe_hc_vs_ls", "MWE bios"), ("mwe_hc_vs_orig(native LLM)", "MWE: casual vs native LLM original"),
            ("wild_test_orig_vs_ls", "WildChat: human original vs LLM"), ("jbb_harmful_orig_vs_ls", "JBB harmful: original vs LLM"),
            ("wild_test_hf_vs_ls", "WildChat: formal human vs LLM"), ("wild_test_len(short_vs_long)", "WildChat: short vs long (length)")]
    Ps = {m: load(f"probes_{m}.json") for m in ["qwen", "llama", "gemma"]}
    for k, lab in keys:
        cells = []
        for m, P in Ps.items():
            if P:
                L = P["n_layers"]
                cells.append(f"{P['probe']['mean'][k]['dom'][L // 2]:.2f}")
        lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
    w("transfer.tex", "\n".join(lines))


def t_steer():
    S = load("steer_summary.json")
    if not S:
        return
    fams = [("style_last", "LLM-vs-casual style"), ("formal_last", "formality"), ("resid_last", "LLM-vs-formal residual"),
            ("resid_orth", "residual $\\perp$ formality, length"), ("len_last", "length"), ("stated_last", "stated AI authorship"),
            ("eval_last", "evaluation awareness"), ("style_mean", "style (user-token mean)"), ("refusal_last", "refusal (positive control)")]
    for m in S:
        C, A = S[m]["conds"], S[m]["A"]
        lines = []

        def row(lab, k):
            c = C[k]
            f = lambda d: (r"\textbf{%+.1f}" if (d["lo"] > 0 or d["hi"] < 0) else "%+.1f") % (100 * d["diff"])
            gc = c.get("gap_closed", {})
            gcv = f"{list(gc.values())[1]:+.1f}" if gc else "--"
            return (f"{lab} & {f(c['harmful'])} & {f(c['borderline'])} & {f(c['syco'])} & {100 * (c['mwe_orig_syco'] - C['base']['mwe_orig_syco']):+.1f} & "
                    f"{c['nll'] - C['base']['nll']:+.2f} & {gcv}" + r" \\")

        b = C["base"]
        lines.append(f"unsteered (absolute) & {100 * b['harmful']['rate']:.1f} & {100 * b['borderline']['rate']:.1f} & {100 * b['syco']['rate']:.1f} & "
                     f"{100 * b['mwe_orig_syco']:.1f} & {b['nll']:.2f} & 0" + r" \\ \midrule")
        for a in [-2 * A, -A, -A / 2, A / 2, A, 2 * A]:
            lines.append(row(f"style ${a:+g}s$", f"style_last@{a:+g}"))
        lines.append(r"\midrule")
        for f_, lab in fams[1:]:
            for sgn in [-A, A]:
                lines.append(row(f"{lab} ${sgn:+g}s$", f"{f_}@{sgn:+g}"))
        lines.append(r"\midrule")
        for a in [A, 2 * A]:
            for sd in range(3):
                lines.append(row(f"random {sd + 1} ${a:g}s$", f"random{sd}@{a:+g}"))
        w(f"steer_{m}.tex", "\n".join(lines))
    # dose-response correlations
    lines = []
    for m in S:
        for k, r in S[m].get("dose_response", {}).items():
            g, c = k.split("|")
            lines.append(f"{NAMES[m]} & {g} & {c} & {r['mean_dproj']:+.2f} & {r['r']:+.2f} & {r['p']:.3f} & {r['n']}" + r" \\")
    w("dose.tex", "\n".join(lines))


def t_sweep():
    lines = []
    for m in ["qwen", "llama", "gemma"]:
        R = load(f"sweep_{m}.json")
        if not R:
            continue
        base = [r for r in R if r["tag"] == "base"][0]
        layers = sorted({r["layer"] for r in R if r.get("layer")})
        for l in layers:
            g = lambda tag, a: [r for r in R if r["tag"] == tag and r["layer"] == l and r.get("alpha") == a]
            sp, sm = g("style_last", 4)[0], g("style_last", -4)[0]
            rnd = g("random_last", 4)
            lines.append(f"{NAMES[m] if l == layers[0] else ''} & {l} & {base['p_ai']:.2f} & {sm['p_ai']:.2f} & {sp['p_ai']:.2f} & "
                         f"{rnd[0]['p_ai']:.2f}, {rnd[1]['p_ai']:.2f} & {base['nll']:.2f} & {sm['nll']:.2f} & {sp['nll']:.2f} & "
                         f"{rnd[0]['nll']:.2f}, {rnd[1]['nll']:.2f}" + r" \\")
        lines.append(r"\midrule")
    w("sweep.tex", "\n".join(lines[:-1]))


def t_judge2():
    S = load("syco_second_judge.json")
    if not S:
        return
    rows = []
    for m in S:
        r = S[m]
        f = lambda c: "%+.1f [%+.1f, %+.1f]" % (100 * r["contrasts"][c]["diff"], 100 * r["contrasts"][c]["lo"], 100 * r["contrasts"][c]["hi"])
        rows.append(f"{NAMES[m]} & {100 * r['rates']['hc']:.1f} & {100 * r['rates']['hf']:.1f} & {100 * r['rates']['ls']:.1f} & "
                    f"{f('LLM - casual human (short)')} & {f('formal human - casual human')} & {f('LLM - formal human')} & {f('long - short')}" + r" \\")
    w("judge2.tex", "\n".join(rows))


if __name__ == "__main__":
    t_data()
    S = load("behave_summary_gpt41.json")
    models = [m for m in NAMES if m in S]
    t_rates(S, models, "rates.tex")
    t_contrasts(S, models, "contrasts_style.tex", ["LLM - casual human (short)", "LLM - casual human (long)", "formal human - casual human", "LLM - formal human"])
    t_contrasts(S, models, "contrasts_other.tex", ["long - short", "LLM paraphrase - original", "casual-human paraphrase - original", "label AI - label human", "label test - label deploy"])
    G = load("behave_summary_gemini.json")
    if G:
        t_contrasts(G, [m for m in NAMES if m in G], "contrasts_gemini.tex", ["LLM - casual human (short)", "LLM - casual human (long)", "formal human - casual human", "LLM - formal human", "long - short"])
    t_manip()
    t_transfer()
    t_steer()
    t_sweep()
    t_judge2()
    print("tables written")
