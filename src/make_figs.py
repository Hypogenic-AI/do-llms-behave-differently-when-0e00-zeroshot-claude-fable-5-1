"""Build all paper figures from results/*.json into paper_draft/figs/."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "paper_draft", "figs")
os.makedirs(FIG, exist_ok=True)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MODELS = {"qwen": ("Qwen2.5-7B", "#2a78d6", "o"), "llama": ("Llama-3.1-8B", "#eb6834", "s"), "gemma": ("Gemma-2-9B", "#1baf7a", "^"),
          "gpt41mini": ("GPT-4.1-mini", "#eda100", "D"), "llama70b": ("Llama-3.3-70B", "#e87ba4", "v"), "haiku45": ("Claude Haiku 4.5", "#4a3aa7", "P")}
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "figure.facecolor": "white", "axes.facecolor": "white", "legend.frameon": False, "axes.titlesize": 9})


def fig_behaviour():
    S = json.load(open(os.path.join(ROOT, "results", "behave_summary_gpt41.json")))
    groups = [("harmful (refusal)", "Refusal, harmful"), ("borderline-benign (refusal)", "Refusal, borderline-benign"),
              ("TruthfulQA+suggestion (sycophancy)", "Sycophancy, TruthfulQA"), ("MWE political (sycophancy)", "Sycophancy, MWE political")]
    contrasts = [("LLM - casual human (pooled)", "LLM vs casual human\n(bundled style)"), ("formal human - casual human", "formal vs casual human\n(formality)"),
                 ("LLM - formal human", "LLM vs formal human\n(authorship, formality-matched)"), ("long - short", "elaborated vs short\n(length/explicitness)"),
                 ("label AI - label human", "stated 'AI-written' vs\n'human-written' label")]
    models = [m for m in MODELS if m in S]
    fig, axes = plt.subplots(1, 4, figsize=(11, 4.6), sharey=True)
    for ax, (g, title) in zip(axes, groups):
        for ci, (c, _) in enumerate(contrasts):
            for mi, m in enumerate(models):
                r = S[m]["contrasts"].get(g, {}).get(c)
                if c.startswith("LLM - casual") and r is None:
                    r = S[m]["contrasts"].get(g, {}).get("LLM - casual human (short)")
                if not r:
                    continue
                y = ci + (mi - (len(models) - 1) / 2) * 0.13
                name, col, mk = MODELS[m]
                ax.plot([100 * r["lo"], 100 * r["hi"]], [y, y], color=col, lw=1.4, solid_capstyle="round")
                ax.plot(100 * r["diff"], y, mk, color=col, ms=4.5, mec="white", mew=0.6, label=name if ci == 0 else None)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_title(title)
        ax.set_xlabel("difference (percentage points)")
        ax.set_yticks(range(len(contrasts)))
        ax.set_yticklabels([c[1] for c in contrasts])
        ax.set_ylim(len(contrasts) - 0.5, -0.5)
        ax.grid(axis="y", visible=False)
    axes[0].legend(loc="upper center", bbox_to_anchor=(2.3, 1.2), ncol=6)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "behaviour.pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_probes():
    ms = [m for m in ["qwen", "llama", "gemma"] if os.path.exists(os.path.join(ROOT, "results", f"probes_{m}.json"))]
    fig, axes = plt.subplots(1, len(ms), figsize=(3.6 * len(ms), 2.9), sharey=True)
    lines = [("hc_vs_ls", "LLM vs casual human", "#2a78d6", "-"), ("hc_vs_hf", "formal vs casual human", "#eb6834", "--"),
             ("hf_vs_ls", "LLM vs formal human", "#1baf7a", "-"), ("orig_vs_ls", "LLM vs original human", "#eda100", "-.")]
    for ax, m in zip(np.atleast_1d(axes), ms):
        P = json.load(open(os.path.join(ROOT, "results", f"probes_{m}.json")))
        ded = P["dedicated"]["mean"]
        for k, lab, col, ls in lines:
            ys = [(l, v) for l, v in enumerate(ded[k]["lr"]) if v is not None and l > 0]
            ax.plot([l for l, _ in ys], [v for _, v in ys], ls, color=col, lw=2, marker="o", ms=3.5, label=lab)
        ax.axhline(0.5, color=MUTED, lw=0.8, ls=":")
        ax.set_title(MODELS[m][0])
        ax.set_xlabel("layer")
        ax.set_ylim(0.45, 1.02)
    np.atleast_1d(axes)[0].set_ylabel("held-out probe AUC")
    np.atleast_1d(axes)[0].legend(loc="lower right", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "probes.pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_cos():
    S = json.load(open(os.path.join(ROOT, "results", "steer_summary.json"))) if os.path.exists(os.path.join(ROOT, "results", "steer_summary.json")) else {}
    ms = [m for m in ["qwen", "llama", "gemma"] if os.path.exists(os.path.join(ROOT, "results", f"probes_{m}.json"))]
    names = ["style_last", "formal_last", "resid_last", "len_last", "stated_last", "eval_last", "refusal_last"]
    labels = ["LLM-vs-casual\n(style)", "formality", "LLM-vs-formal\n(residual)", "length", "stated\nAI author", "eval\nawareness", "refusal"]
    fig, axes = plt.subplots(1, len(ms), figsize=(3.9 * len(ms), 3.6))
    for ax, m in zip(np.atleast_1d(axes), ms):
        P = json.load(open(os.path.join(ROOT, "results", f"probes_{m}.json")))
        layer = S.get(m, {}).get("layer", P["n_layers"] // 2)
        M = np.eye(len(names))
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                if i < j:
                    v = P["cos"].get(f"{a}|{b}") or P["cos"].get(f"{b}|{a}")
                    M[i, j] = M[j, i] = v[layer]
        im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{M[i, j]:.2f}".replace("0.", ".").replace("1.00", "1"), ha="center", va="center", fontsize=6.5,
                        color="white" if abs(M[i, j]) > 0.6 else INK)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(labels if ax is np.atleast_1d(axes)[0] else [], fontsize=7)
        ax.set_title(f"{MODELS[m][0]} (layer {layer})")
        ax.grid(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "cosines.pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_steer():
    path = os.path.join(ROOT, "results", "steer_summary.json")
    if not os.path.exists(path):
        return
    S = json.load(open(path))
    ms = [m for m in ["qwen", "llama", "gemma"] if m in S]
    fams = [("style_last", "LLM-vs-casual (style)"), ("formal_last", "formality"), ("resid_last", "LLM-vs-formal (residual)"),
            ("resid_orth", "residual, orthogonalised"), ("len_last", "length"), ("stated_last", "stated AI author"),
            ("eval_last", "eval awareness")]
    outs = [("harmful", "refusal, harmful"), ("borderline", "refusal, borderline-benign"), ("syco", "sycophancy, TruthfulQA")]
    fig, axes = plt.subplots(len(ms), 3, figsize=(11, 2.5 * len(ms)), sharey=True, squeeze=False, constrained_layout=True)
    for r, m in enumerate(ms):
        C = S[m]["conds"]
        A = S[m]["A"]
        for c, (g, title) in enumerate(outs):
            ax = axes[r, c]
            rnd = [100 * C[k][g]["diff"] for k in C if k.startswith("random") and k.endswith(f"@{A:+g}")] + [0.0]
            ax.axvspan(min(rnd), max(rnd), color="#d9d8d3", alpha=0.7, lw=0, label="3 random directions at the same norm (range)" if (r == 0 and c == 0) else None)
            ax.axvline(0, color=MUTED, lw=0.8)
            for i, (f, lab) in enumerate(fams):
                for sign, col, mk, off in [(+1, "#eb6834", "o", -0.14), (-1, "#2a78d6", "s", 0.14)]:
                    k = f"{f}@{sign * A:+g}"
                    if k not in C:
                        continue
                    d = C[k][g]
                    ax.plot([100 * d["lo"], 100 * d["hi"]], [i + off, i + off], color=col, lw=1.4, solid_capstyle="round")
                    ax.plot(100 * d["diff"], i + off, mk, color=col, ms=4.5, mec="white", mew=0.6,
                            label=(f"+{A:g}s (towards the named pole)" if sign > 0 else f"-{A:g}s (away from it)") if (r == 0 and c == 0 and i == 0) else None)
            ax.set_yticks(range(len(fams)))
            ax.set_yticklabels([f[1] for f in fams])
            ax.set_ylim(len(fams) - 0.5, -0.5)
            ax.grid(axis="y", visible=False)
            ax.set_xlim(-22, 28)
            ax.set_title(f"{MODELS[m][0]}: {title}")
            if r == len(ms) - 1:
                ax.set_xlabel("change vs. unsteered (percentage points)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=3)
    fig.savefig(os.path.join(FIG, "steering.pdf"), bbox_inches="tight")
    plt.close(fig)

    # dose-response along the style direction
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.6))
    keys = [("harmful", "refusal, harmful (%)"), ("borderline", "refusal, borderline-benign (%)"), ("syco", "sycophancy, TruthfulQA (%)")]
    for m in ms:
        C = S[m]["conds"]
        name, col, mk = MODELS[m]
        pts = sorted([(0.0, C["base"])] + [(float(k.split("@")[1]), v) for k, v in C.items() if k.startswith("style_last@")])
        for ax, (g, yl) in zip(axes, keys):
            ax.plot([a for a, _ in pts], [100 * v[g]["rate"] for _, v in pts], "-", color=col, marker=mk, ms=4.5, lw=2, mec="white", mew=0.6, label=name)
            ax.set_ylabel(yl)
        axes[3].plot([a for a, _ in pts], [v["nll"] for _, v in pts], "-", color=col, marker=mk, ms=4.5, lw=2, mec="white", mew=0.6)
    axes[3].set_ylabel("NLL of unsteered replies (nats/token)")
    for ax in axes:
        ax.set_xlabel("dose along style direction (units of $s$)")
    axes[0].legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "dose.pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_behaviour()
    fig_probes()
    fig_cos()
    fig_steer()
    print("figures written to", FIG)
