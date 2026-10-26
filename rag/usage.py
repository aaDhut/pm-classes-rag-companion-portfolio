"""Token counts and cost (USD and INR) for one question: search + transcription + answer,
and separately for reading the answer aloud."""
import json
import threading
import time
import urllib.request
from datetime import date

from . import config


def price(model: str, on: date | None = None) -> dict | None:
    """USD per 1M tokens for model on a date. Matches "gemini-3.5-flash-001" etc. to its
    family; the longest family name wins."""
    day = (on or date.today()).isoformat()
    for name in sorted(config.PRICES, key=len, reverse=True):
        if model.startswith(name):
            for start, p in config.PRICES[name]:
                if day >= start:
                    return p
    return None


_fx = {"rate": None, "at": 0.0}
_fx_lock = threading.Lock()


def usd_to_inr() -> float:
    """Today's rate, fetched at most once a day; the fallback if the fetch fails."""
    with _fx_lock:
        if _fx["rate"] and time.time() - _fx["at"] < 86400:
            return _fx["rate"]
        try:
            with urllib.request.urlopen(config.FX_URL, timeout=5) as r:
                rate = float(json.load(r)["rates"]["INR"])
            _fx.update(rate=rate, at=time.time())
            return rate
        except Exception:
            # Try again in an hour rather than on every page load.
            _fx.update(rate=_fx["rate"] or config.USD_INR_FALLBACK, at=time.time() - 86400 + 3600)
            return _fx["rate"]


class Usage:
    """Safe to add to from several threads (the picture check runs beside the answer)."""

    def __init__(self):
        self._lock = threading.Lock()
        self.input = 0      # prompt tokens sent to Gemini (text + audio)
        self.cached = 0     # part of input billed at the cached rate
        self.output = 0     # answer + thinking tokens
        self.thinking = 0   # the thinking part of output
        self.embed = 0      # tokens embedded for the search
        self.cost = 0.0
        self.priced = True  # False if some model had no entry in config.PRICES
        self.models: list[str] = []

    def _model(self, model: str) -> dict | None:
        if model not in self.models:
            self.models.append(model)
        p = price(model)
        if p is None:
            self.priced = False
        return p

    def add_generation(self, model: str, meta) -> None:
        """meta: the usage_metadata from a Gemini response."""
        if meta is None:
            return
        with self._lock:
            self._add_generation(model, meta)

    def _add_generation(self, model: str, meta) -> None:
        prompt = meta.prompt_token_count or 0
        cached = meta.cached_content_token_count or 0
        thoughts = meta.thoughts_token_count or 0
        out = (meta.candidates_token_count or 0) + thoughts
        audio = sum(d.token_count or 0 for d in (meta.prompt_tokens_details or [])
                    if str(d.modality).upper().endswith("AUDIO"))
        self.input += prompt
        self.cached += cached
        self.output += out
        self.thinking += thoughts
        p = self._model(model)
        if p is None:
            return
        text = max(prompt - audio - cached, 0)
        self.cost += (text * p["in"] + cached * p.get("cached_in", p["in"])
                      + audio * p.get("audio_in", p["in"]) + out * p["out"]) / 1e6

    def add_embedding(self, tokens: int) -> None:
        with self._lock:
            self._add_embedding(tokens)

    def _add_embedding(self, tokens: int) -> None:
        self.embed += tokens
        p = self._model(config.EMBED_MODEL)
        if p is not None:
            self.cost += tokens * p["in"] / 1e6

    def to_dict(self) -> dict:
        rate = usd_to_inr()
        cost = round(self.cost, 6) if self.priced else None
        return {"input": self.input, "cached": self.cached, "output": self.output,
                "thinking": self.thinking, "embed": self.embed, "cost": cost,
                "inr_rate": round(rate, 4),
                "cost_inr": round(self.cost * rate, 4) if self.priced else None,
                "models": [m for m in self.models if m != config.EMBED_MODEL]}


def inr(u: dict) -> float | None:
    """Rupee cost saved with the usage, or worked out at today's rate for older entries."""
    if u.get("cost_inr") is not None:
        return u["cost_inr"]
    return None if u.get("cost") is None else u["cost"] * usd_to_inr()


def fmt_money(usd: float | None, rupees: float | None) -> str:
    if usd is None:
        return "cost n/a"
    d = "$0" if usd == 0 else f"${usd:.4f}" if usd < 0.01 else f"${usd:.3f}"
    return f"{d} (₹{rupees:.2f})" if rupees is not None else d


def describe(u: dict) -> str:
    """One-line summary shown under an answer."""
    thinking = f" (incl. {u['thinking']:,} thinking)" if u.get("thinking") else ""
    models = " · " + ", ".join(u["models"]) if u.get("models") else ""
    return (f"🪙 Tokens: {u['input'] + u.get('embed', 0):,} input · {u['output']:,} output{thinking}"
            f" · Cost: {fmt_money(u.get('cost'), inr(u))}{models}")


def describe_voice(t: dict) -> str:
    secs = round(t["audio_tokens"] / 25)
    plays = t.get("plays", 1)
    what = f"{plays} plays" if plays > 1 else "stopped early" if not t.get("complete") else ""
    what = f", {what}" if what else ""
    return (f"🔊 Voice: {t['audio_tokens']:,} audio tokens ({secs // 60}:{secs % 60:02d} of speech{what})"
            f" · Cost: {fmt_money(t['cost'], t.get('cost_inr'))}")


TOKENS_HELP = (
    "**Tokens** are the small pieces of text the AI reads and writes (1 token ≈ ¾ of a word). "
    "Google bills per token.\n\n"
    "- **Input**: your question, the class slides it read, and the chat so far.\n"
    "- **Output**: the answer it wrote.\n"
    "- **Thinking**: the AI's private reasoning before answering. Billed as output.\n"
    "- **Audio tokens** (voice): 25 tokens per second of speech.\n\n"
    "Rupees use the exchange rate on the day the answer was given.")
