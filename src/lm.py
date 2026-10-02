"""Local-model utilities: chat formatting, batched generation, activation capture, steering / ablation hooks."""
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "models", "hf"))
MODELS = {
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "gemma": "google/gemma-2-9b-it",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
}

from lm_prompts import AUTH_Q, PAIR_Q  # noqa: E402,F401


class LM:
    def __init__(self, key):
        name = MODELS[key]
        self.key = key
        self.tok = AutoTokenizer.from_pretrained(name)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map="cuda")
        self.model.eval()
        self.layers = self.model.model.layers
        self.n_layers = len(self.layers)
        self.hooks = []

    # ---------- formatting ----------
    def fmt(self, user, prefill=""):
        s = self.tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
        return s + prefill

    def encode(self, texts):
        return self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False, return_offsets_mapping=True)

    # ---------- interventions ----------
    def clear(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def set_steer(self, layer, vec):
        """Add `vec` to the residual stream at hidden_states[layer] (output of block layer-1), all positions."""
        v = torch.as_tensor(vec, dtype=torch.bfloat16, device="cuda")

        def hook(mod, inp, out):
            if isinstance(out, tuple):
                return (out[0] + v,) + tuple(out[1:])
            return out + v

        self.hooks.append(self.layers[layer - 1].register_forward_hook(hook))

    def set_ablate(self, unit):
        """Project the unit direction out of the residual stream after every block (and the embeddings)."""
        u = torch.as_tensor(unit, dtype=torch.float32, device="cuda")
        u = (u / u.norm()).to(torch.bfloat16)

        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h - (h @ u).unsqueeze(-1) * u
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

        for l in self.layers:
            self.hooks.append(l.register_forward_hook(hook))
        self.hooks.append(self.model.model.embed_tokens.register_forward_hook(hook))

    # ---------- runs ----------
    @torch.no_grad()
    def generate(self, users, max_new_tokens=160, bs=48):
        texts = [self.fmt(u) for u in users]
        order = np.argsort([-len(t) for t in texts])
        out = [None] * len(texts)
        for i in range(0, len(texts), bs):
            idx = order[i:i + bs]
            enc = self.encode([texts[j] for j in idx])
            enc.pop("offset_mapping")
            enc = {k: v.cuda() for k, v in enc.items()}
            g = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None,
                                    top_k=None, pad_token_id=self.tok.pad_token_id)
            dec = self.tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for j, d in zip(idx, dec):
                out[j] = d
        return out

    @torch.no_grad()
    def acts(self, users, bs=32):
        """Residual activations (all layers incl. embeddings) at the last prompt token and averaged over user-content tokens."""
        texts = [self.fmt(u) for u in users]
        order = np.argsort([-len(t) for t in texts])
        last = np.zeros((len(texts), self.n_layers + 1, self.model.config.hidden_size), dtype=np.float16)
        mean = np.zeros_like(last)
        for i in range(0, len(texts), bs):
            idx = order[i:i + bs]
            enc = self.encode([texts[j] for j in idx])
            off = enc.pop("offset_mapping")
            mask = torch.zeros(enc["input_ids"].shape, dtype=torch.bool)
            for b, j in enumerate(idx):
                s = texts[j].find(users[j].strip())
                e = s + len(users[j].strip())
                mask[b] = (off[b, :, 0] >= s) & (off[b, :, 1] <= e) & (off[b, :, 1] > off[b, :, 0]) & enc["attention_mask"][b].bool()
            enc = {k: v.cuda() for k, v in enc.items()}
            hs = self.model.model(**enc, output_hidden_states=True).hidden_states
            m = mask.cuda().unsqueeze(-1).float()
            for l, h in enumerate(hs):
                h = h.float()
                last[idx, l] = h[:, -1].cpu().numpy()
                mean[idx, l] = ((h * m).sum(1) / m.sum(1).clamp(min=1)).cpu().numpy()
        return last, mean

    @torch.no_grad()
    def choice_logits(self, users, a="A", b="B", prefill="(", bs=32):
        """logit(b) - logit(a) for the next token after an assistant prefill."""
        ia = self.tok.encode(a, add_special_tokens=False)[0]
        ib = self.tok.encode(b, add_special_tokens=False)[0]
        texts = [self.fmt(u, prefill) for u in users]
        order = np.argsort([-len(t) for t in texts])
        out = np.zeros(len(texts))
        for i in range(0, len(texts), bs):
            idx = order[i:i + bs]
            enc = self.encode([texts[j] for j in idx])
            enc.pop("offset_mapping")
            enc = {k: v.cuda() for k, v in enc.items()}
            h = self.model.model(**enc).last_hidden_state[:, -1]  # avoid materialising full-vocab logits at every position
            lg = self.model.lm_head(h).float()
            cap = getattr(self.model.config, "final_logit_softcapping", None)
            if cap:
                lg = torch.tanh(lg / cap) * cap
            out[idx] = (lg[:, ib] - lg[:, ia]).cpu().numpy()
        return out

    def p_ai(self, msgs, bs=32):
        """Verbalised authorship judgement P('AI'), averaged over both option orders."""
        d1 = self.choice_logits([AUTH_Q.format(msg=m, a="Human", b="AI") for m in msgs], bs=bs)  # B = AI
        d2 = self.choice_logits([AUTH_Q.format(msg=m, a="AI", b="Human") for m in msgs], bs=bs)  # A = AI
        sig = lambda x: 1 / (1 + np.exp(-x))
        return (sig(d1) + sig(-d2)) / 2

    def pair_ai(self, first, second, bs=16):
        """2AFC: probability that `second` (vs `first`) is picked as the AI-generated version, averaged over both orders."""
        d1 = self.choice_logits([PAIR_Q.format(a=x, b=y) for x, y in zip(first, second)], bs=bs)  # B = second
        d2 = self.choice_logits([PAIR_Q.format(a=y, b=x) for x, y in zip(first, second)], bs=bs)  # A = second
        sig = lambda x: 1 / (1 + np.exp(-x))
        return (sig(d1) + sig(-d2)) / 2

    @torch.no_grad()
    def nll(self, users, responses, bs=8, max_resp_tokens=60):
        """Mean teacher-forced NLL (nats/token) of fixed responses, used as a degradation measure under interventions."""
        out = []
        self.tok.padding_side = "right"
        try:
            for i in range(0, len(users), bs):
                pre = [self.fmt(u) for u in users[i:i + bs]]
                rs = [self.tok.decode(self.tok.encode(r, add_special_tokens=False)[:max_resp_tokens]) for r in responses[i:i + bs]]
                n_pre = [len(self.tok.encode(p, add_special_tokens=False)) for p in pre]
                enc = self.tok([p + r for p, r in zip(pre, rs)], return_tensors="pt", padding=True, add_special_tokens=False)
                enc = {k: v.cuda() for k, v in enc.items()}
                lp = torch.log_softmax(self.model(**enc).logits.float(), -1)
                ids, am = enc["input_ids"], enc["attention_mask"]
                for b in range(len(pre)):
                    T = int(am[b].sum())
                    tgt = ids[b, n_pre[b]:T]
                    l = lp[b, n_pre[b] - 1:T - 1].gather(-1, tgt[:, None]).squeeze(-1)
                    out.append(float(-l.mean()))
        finally:
            self.tok.padding_side = "left"
        return np.array(out)
