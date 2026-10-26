"""Which class a question's topic is taught in, and which excerpts to answer from.

A word like "matrix" or "retention" comes up in many classes. Taking the top 8 matches
alone lets a class that only shares the word crowd out the class that teaches the topic.
So:
1. Heat: a wide search (about 40 chunks per query), scored per class by rank. Only a
   file's best few chunks count, so overlapping neighbour chunks can't inflate a class.
2. Route: a quick Flash-Lite call reads the best passages of the hottest classes and says
   which classes teach the topic ("main"), and whether they teach the same idea or the
   word means different things there. Embedding similarity can't tell these apart: all
   PM classes look alike to it (0.8-0.93 between any two).
3. Select: most excerpt slots go to the main classes, each with the teacher speaking in
   the lecture when there is such a piece; the rest stay open for other classes, so a
   topic also taught elsewhere still gets in.
"""
import re
from collections import defaultdict

from . import llm, store

WIDE = 40           # chunks per search
HEAT_PER_FILE = 3   # a file's best chunks that count toward its class's heat
SHORTLIST = 5       # hottest classes shown to the router
SNIPPETS = 3        # passages per class shown to the router
SAME_TEXT = 0.6     # word overlap at which two chunks count as the same passage
HOLD_HEAT = 1.5     # a class this many times hotter than the router's first pick is kept


def key(h: dict) -> tuple:
    return (h["file_id"], h.get("start"), h.get("page"), h.get("part"), h["text"][:80])


def merge(lists: list[list[dict]]) -> list[dict]:
    """Chunks from several searches, best first. A chunk's score adds up 1/(rank+5) over
    the searches that found it, so ones both searches rank high come first."""
    score, first = defaultdict(float), {}
    for hits in lists:
        for rank, h in enumerate(hits):
            k = key(h)
            score[k] += 1 / (rank + 5)
            first.setdefault(k, h)
    return [{**first[k], "_score": s} for k, s in sorted(score.items(), key=lambda x: -x[1])]


TITLE_BOOST = 1.5   # heat multiplier for a class whose title names the question's topic
# Words in many class titles or questions that say nothing about the topic.
_GENERIC = set("""product products management manager managers class classes what which when where
why how does about important learn need know explain tell give model models type types using
with from into this that their there them they your have the and for are was were frameworks
framework""".split())


def _topic_words(text: str) -> set[str]:
    return {w[:5] for w in _words(text) if len(w) > 2 and w not in _GENERIC}


def heat(cands: list[dict], question: str = "") -> dict[str, float]:
    """{class label: share of the heat, 0-1}. With the question, a class whose title shares
    a topic word with it ("metrics" -> "Class 26 - Metrics and Priortization Frameworks")
    counts TITLE_BOOST times: that class is about the topic, not just mentioning it."""
    per_file = defaultdict(list)
    for h in cands:
        per_file[(h["class_label"], h["file_id"])].append(h["_score"])
    out = defaultdict(float)
    for (label, _), scores in per_file.items():
        out[label] += sum(sorted(scores, reverse=True)[:HEAT_PER_FILE])
    if q := _topic_words(question):
        for label in out:
            title = re.sub(r"^(Class \d+|Sunday Workshop[^-]*)\s*-\s*", "", label)
            if q & _topic_words(title):
                out[label] *= TITLE_BOOST
    total = sum(out.values()) or 1
    return {c: v / total for c, v in sorted(out.items(), key=lambda x: -x[1])}


def _kind(h: dict) -> str:
    return {"lecture": "lecture video", "slide": "slide in lecture", "summary": "class summary"}.get(
        h.get("source"), "notes")


def _snippets(cands: list[dict], label: str) -> list[dict]:
    """The class's best passages, from different files where it can."""
    mine = [h for h in cands if h["class_label"] == label]
    out, files = [], set()
    for h in mine:
        if h["file_id"] not in files:
            out.append(h)
            files.add(h["file_id"])
        if len(out) == SNIPPETS:
            return out
    return out + [h for h in mine if h not in out][:SNIPPETS - len(out)]


def _known(labels, short: list[str]) -> list[str]:
    """The model's class labels mapped onto the real ones. It sometimes shortens a label
    ("Class 26") or fixes its spelling ("Prioritization")."""
    out = []
    for lb in labels or []:
        lb = lb.strip()
        hit = lb if lb in short else None
        if not hit and (m := re.match(r"Class (\d+)\b", lb)):
            hit = next((s for s in short if re.match(rf"Class {m.group(1)}\b", s)), None)
        if not hit and short:
            w = _words(lb)
            best = max(short, key=lambda s: len(w & _words(s)))
            hit = best if len(w & _words(best)) >= max(2, len(w) // 2) else None
        if hit and hit not in out:
            out.append(hit)
    return out


def _norm(text: str) -> str:
    """Words cut to 5 letters, so a plural or other word form counts as the same word."""
    return " ".join(sorted({w[:5] for w in _words(text)}))


def route(question: str, cands: list[dict], usage=None, search=None) -> dict | None:
    """{"kind", "groups": [{"label", "meaning", "pool", "query"}], "also": [labels], "heat"}
    or None when the router can't be reached. A group is one class (and meaning) that
    teaches the topic; its excerpts are picked from its pool.
    search(text) -> candidates: used when the question looks voice-typed wrong
    ("matrix" for "metrics"); the corrected question is searched and becomes a group."""
    shares = heat(cands, question)
    short = list(shares)[:SHORTLIST]
    if not short:
        return None
    blocks = []
    for label in short:
        lines = [f"## {label} (heat {100 * shares[label]:.0f}%)"]
        for h in _snippets(cands, label):
            text = re.sub(r"\s+", " ", h["text"])[:450]
            lines.append(f"- [{_kind(h)}] {text}")
        blocks.append("\n".join(lines))
    try:
        r = llm.route_topic(question, "\n\n".join(blocks), usage)
    except Exception as e:  # the answer still works without it
        print(f"topic routing failed: {e}")
        return None

    main = _known(r.get("main"), short)[:3]
    kind = r.get("kind") if main else "not_taught"
    if kind == "not_taught":
        main = []
    if kind == "focused":
        main = main[:1]
    meanings = {}
    for m in r.get("meanings") or []:
        if (lb := _known([m.get("class", "")], short)) and m.get("meaning"):
            meanings[lb[0]] = m["meaning"].strip()
    groups = [{"label": c, "meaning": meanings.get(c, ""), "pool": cands, "query": None} for c in main]
    # The router only sees a few passages per class, so it can miss what the hottest class
    # has deeper down. A class far hotter than its picks stays in.
    hottest = short[0]
    if groups and hottest not in main and shares[hottest] >= HOLD_HEAT * shares[main[0]]:
        groups.insert(0, {"label": hottest, "meaning": "", "pool": cands, "query": None})
        groups = groups[:3]
        if kind == "focused":
            kind = "same_topic"

    fixed = (r.get("corrected") or "").strip()
    if search and fixed and _norm(fixed) != _norm(question):
        try:
            found = search(fixed)
        except Exception as e:
            print(f"search for the corrected question failed: {e}")
            found = []
        if found and (top := next(iter(heat(found, fixed)))):
            meaning = (r.get("corrected_meaning") or fixed).strip()
            new = {"label": top, "meaning": meaning, "pool": found, "query": fixed}
            # Same class and the same idea as a group the router gave: search it the fixed way.
            mw = {w for w in _words(meaning) if len(w) > 3}
            groups = [g for g in groups if not (g["label"] == top and mw & _words(g["meaning"]))]
            groups = [new] + groups[:2]
            kind = "different_meanings" if len(groups) > 1 else "focused"
    if kind in ("same_topic", "different_meanings") and len(groups) < 2:
        kind = "focused"
    labels = {g["label"] for g in groups}
    also = [c for c in _known(r.get("also"), short) if c not in labels]
    return {"kind": kind if groups else "not_taught", "groups": groups, "also": also, "heat": shares,
            "corrected": fixed}


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _same(a: dict, b: dict) -> bool:
    wa, wb = _words(a["text"]), _words(b["text"])
    return bool(wa and wb) and len(wa & wb) / len(wa | wb) >= SAME_TEXT


def _spoken(h: dict) -> bool:
    return h.get("source") == "lecture"


def _said(h: dict) -> str:
    """A chunk's own words: for a lecture piece, what was said, not the on-screen text."""
    return h["text"].split("\n\n[On screen]")[0] if _spoken(h) else h["text"]


def _mentions(h: dict, words: set[str], need: int = 1) -> bool:
    """Whether a chunk says `need` of the topic words (True when there are none to look for)."""
    return not words or len(words & {w[:5] for w in _words(_said(h))}) >= min(need, len(words))


def select(cands: list[dict], plan: dict | None, k: int, query: str, class_labels, sources: str,
           usage=None) -> list[dict]:
    """The k excerpts to answer from: the groups' first, then other classes. Each excerpt
    of a group gets "_g" (its group number) and "main": True."""
    chosen, keys = [], set()

    def take(pool, n, group=None, per_class=None):
        count, got = defaultdict(int), 0
        for h in pool:
            if got >= n:
                break
            if key(h) in keys or any(_same(h, o) for o in chosen):
                continue
            if per_class and count[h["class_label"]] >= per_class:
                continue
            chosen.append({**h, "_g": group, "main": group is not None})
            keys.add(key(h))
            count[h["class_label"]] += 1
            got += 1

    if not plan or not plan["groups"]:
        take(cands, k)
        return chosen
    groups = plan["groups"]
    labels = {g["label"] for g in groups}
    others_exist = any(h["class_label"] not in labels for h in cands)
    reserve = (max(2, round(k * (0.35 if len(groups) == 1 else 0.25))) if others_exist else 0)
    # Even shares (any spare slot to the first): with different meanings we can't be sure
    # which one the student wants.
    slots = [(k - reserve) // len(groups)] * len(groups)
    for i in range(k - reserve - sum(slots)):
        slots[i % len(slots)] += 1

    # The question's own words (and its corrected form's): what "Also comes up in" looks for.
    plan["words"] = _topic_words(query) | _topic_words(plan.get("corrected") or "")
    for gi, (g, n) in enumerate(zip(groups, slots)):
        words = g["words"] = _topic_words(g["query"] or query) | _topic_words(g["meaning"])
        need = 2 if len(words) >= 3 else 1
        mine = [h for h in g["pool"] if h["class_label"] == g["label"]]
        start = len(chosen)
        take(mine, n, gi)
        if sources != "notes" and not any(_spoken(h) for h in chosen[start:]):
            # The teacher explaining it, not only the slide or notes; only a moment where the
            # topic is said, not just any piece of that lecture.
            spoken = next((h for h in mine if _spoken(h) and key(h) not in keys and _mentions(h, words, need)),
                          None)
            if spoken is None:
                try:
                    found = store.search(g["query"] or query, class_labels=[g["label"]], limit=3,
                                         usage=usage, sources="lectures")
                except Exception:
                    found = []
                spoken = next((h for h in found if _spoken(h) and key(h) not in keys
                               and _mentions(h, words, need)), None)
            if spoken is not None:
                if len(chosen) - start >= n:
                    keys.discard(key(chosen.pop()))
                take([{**spoken, "_forced": True}], 1, gi)

    rest = [h for h in cands if h["class_label"] not in labels]
    # One from each class the router named as adding something, then the best of the rest.
    for label in plan["also"]:
        if len(chosen) < k:
            take([h for h in rest if h["class_label"] == label], 1)
    take(rest, k - len(chosen), per_class=2)
    take(cands, k - len(chosen))  # not enough elsewhere: more from the groups' classes
    return sorted(chosen, key=lambda h: len(groups) if h["_g"] is None else h["_g"])


def main_classes(hits: list[dict]) -> list[str]:
    """The classes routing found teach the topic, from an answer's hits ([] if none)."""
    return list(dict.fromkeys(h["class_label"] for h in hits if h.get("main")))


# ---------- what the answer model is told ----------

_LINE = ("Start your answer with this one line (only if the excerpts do cover the question): "
         "{line} Each [...](link) is one class: its number and title, the teacher if an excerpt says "
         "who taught it, and the time; use the link and time of the excerpt where that class actually "
         "explains the topic (a lecture video moment if one explains it, written \"▶ 37:10\"; otherwise "
         "its notes page, written \"📄 file, page N\").")


def footer(plan: dict | None, hits: list[dict]) -> str:
    """A tip under an answer that covers several meanings."""
    if plan and plan["kind"] == "different_meanings" and len({h.get("_g") for h in hits} - {None}) > 1:
        return '*Want just one? Pick that class under "Limit to classes" in the sidebar and ask again.*'
    return ""


def prompt_note(plan: dict | None, hits: list[dict]) -> str:
    """What the answer model is told about where the topic is taught (hits: the excerpts
    as numbered in the prompt)."""
    if not plan or not plan["groups"]:
        return ""
    groups = [(g, [i for i, h in enumerate(hits, 1) if h.get("_g") == gi])
              for gi, g in enumerate(plan["groups"])]
    groups = [(g, nums) for g, nums in groups if nums]
    if not groups:
        return ""
    if plan["kind"] == "different_meanings" and len(groups) > 1:
        means = "; ".join(f"\"{g['meaning']}\" in {g['label']} (excerpts {', '.join(map(str, nums))})"
                          for g, nums in groups)
        slip = (f" The question may be voice-typed wrong: it may mean \"{plan['corrected']}\"."
                if plan.get("corrected") else "")
        line = ("\"📍 **This means different things in different classes:** *meaning* → [Class N – "
                "title · teacher · ▶ time](link); *meaning* → [...](link)\", one entry per meaning in that order.")
        return ("Where this is taught: the student's wording fits different topics in different classes: "
                f"{means}.{slip} Give each meaning its own heading, in that order, and answer each "
                "from its own excerpts.\n" + _LINE.format(line=line))
    labels = list(dict.fromkeys(g["label"] for g, _ in groups))
    line = "\"📍 **Taught mainly in:** [Class N – title · teacher · ▶ time](link)\"" + (
        ", one [...] per class, separated by \"; \"." if len(labels) > 1 else ".")
    if plan["kind"] == "same_topic" and len(labels) > 1:
        return (f"Where this is taught: the same topic is taught in {', '.join(labels)}. Give one combined "
                "answer and say which class each point comes from.\n" + _LINE.format(line=line))
    return (f"Where this is taught: mainly {labels[0]}. Build the answer mainly from its excerpts; use the "
            "other classes' excerpts only for points they add, saying which class.\n" + _LINE.format(line=line))
