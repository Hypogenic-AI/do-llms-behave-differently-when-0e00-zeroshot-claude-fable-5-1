"""Thin OpenRouter client with an on-disk cache, retries and cost tracking."""
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "api_cache.jsonl")
_client = None
_lock = threading.Lock()
_cache = None
COST = {"usd": 0.0, "calls": 0}


def client():
    global _client
    if _client is None:
        _client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_KEY"])
    return _client


def _load():
    global _cache
    if _cache is None:
        _cache = {}
        if os.path.exists(CACHE):
            for line in open(CACHE):
                try:
                    r = json.loads(line)
                    _cache[r["k"]] = r["v"]
                except Exception:
                    pass
    return _cache


def chat(model, messages, max_tokens=512, temperature=0.0, tag=""):
    key = hashlib.sha256(json.dumps([model, messages, max_tokens, temperature, tag]).encode()).hexdigest()
    with _lock:
        c = _load()
        if key in c:
            return c[key]
    out = None
    for attempt in range(6):
        try:
            r = client().chat.completions.create(model=model, messages=messages, max_tokens=max_tokens,
                                                 temperature=temperature)
            out = r.choices[0].message.content or ""
            cost = getattr(r.usage, "cost", 0) or 0
            with _lock:
                COST["usd"] += cost
                COST["calls"] += 1
            break
        except Exception as e:  # noqa
            time.sleep(2 + 3 * attempt)
    if out is None:
        return None
    with _lock:
        _cache[key] = out
        with open(CACHE, "a") as f:
            f.write(json.dumps({"k": key, "v": out}) + "\n")
    return out


def pmap(fn, items, workers=24, desc=""):
    with ThreadPoolExecutor(workers) as ex:
        return list(tqdm(ex.map(fn, items), total=len(items), desc=desc, mininterval=10))
