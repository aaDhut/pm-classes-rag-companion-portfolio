"""Accuracy eval: every case is answered by the same steps the app runs (routing, search,
answer, fact-check), then graded claim by claim against the real class material.

Run:  uv run python -m eval.accuracy                  # all cases
      uv run python -m eval.accuracy table where     # only these types (or case ids)
      uv run python -m eval.accuracy --no-verify     # without the fact-check step
      uv run python -m eval.accuracy --workers 6

Grading, per answer:
- Gemini 3.1 Pro splits the answer into its factual claims and checks each one against
  the case's own evidence (transcript windows / notes pages chosen by hand) plus the
  excerpts the app retrieved (lecture pieces widened to the transcript around them).
  Each claim is correct, wrong (contradicted) or unsupported (said as class material,
  but nothing shows it). It also checks the case's "must" facts were covered and that
  follow-up requests (shorter, simpler, …) were done.
- Exact checks: every lecture time or video link in the answer must fall inside an
  excerpt or evidence window of that class (±90 s); "mention" / "must_not_mention" words; not-covered replies.
Scores: accuracy = correct ÷ (correct + wrong + unsupported) over class claims; a clean
answer has no wrong or unsupported claim and no bad time. Results go to
eval/accuracy.md (readable) and eval/accuracy.json.
"""
import argparse
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from google.genai import types

from rag import config, images, llm, modes, netfix, quiz, store, topic, verify
from rag.answer import ask, format_context, search_query
from rag.usage import Usage

from . import sources
from .run import NOT_COVERED, grade_pictures, norm

netfix.install()  # DNS fallback: lookups of the Qdrant host fail now and then on this machine
HERE = Path(__file__).parent
JUDGE_MODEL = "gemini-3.1-pro-preview"
TIME_SLACK = 90
_lock = threading.Lock()


# ---------- answering, the way app.py does it ----------

def _labels() -> list[str]:
    if not hasattr(_labels, "v"):
        _labels.v = store.indexed_classes()
    return _labels.v


def respond(question: str, history: list[dict], classes: list[str] | None, verify_on: bool,
            usage: Usage, allow_general: bool = True) -> dict:
    """One chat turn: {"answer", "raw", "fixes", "hits", "mode"}."""
    intent = modes.detect(question, _labels())
    if intent["mode"] == "class_summary":
        found = [t for lb in intent["classes"] if (t := store.class_summary(lb))]
        if found:
            text = "\n\n---\n\n".join(found)
            return {"answer": text, "raw": text, "fixes": [], "hits": [], "mode": "class_summary",
                    "classes": intent["classes"]}
    topic, restyle = search_query(question, history)
    summary_mode = intent["mode"] in ("class_summary", "topic_summary") and not restyle
    filt = intent["classes"] if intent["mode"] == "class_summary" else classes
    # Each step gets its own Usage so the report can show what the answer and the
    # fact-check cost on their own (the grader's cost is not part of either).
    u_ans, u_chk = Usage(), Usage()
    t0 = time.time()
    hits, stream = ask(question, history, class_labels=filt or None, allow_general=allow_general, k=8,
                       usage=u_ans, topic_summary=summary_mode)
    first, parts = None, []
    for piece in stream:
        if first is None and piece.strip():
            first = time.time() - t0
        parts.append(piece)
    raw = "".join(parts)
    t1 = time.time()
    answer, fixes = verify.check_hits(raw, hits, question, u_chk) if verify_on else (raw, [])
    t2 = time.time()
    for u in (u_ans, u_chk):
        _merge(usage, u)
    mode = "restyle" if restyle else ("topic_summary" if summary_mode else "ask")
    steps = {"answer": _step(u_ans), "check": _step(u_chk),
             "first_word_s": round(first or (t1 - t0), 2), "answer_s": round(t1 - t0, 2),
             "check_s": round(t2 - t1, 2)}
    return {"answer": answer, "raw": raw, "fixes": fixes, "hits": hits, "mode": mode, "topic": topic,
            "steps": steps}


def _step(u: Usage) -> dict:
    return {"input": u.input, "cached": u.cached, "output": u.output, "thinking": u.thinking,
            "cost_usd": round(u.cost, 6)}


def _merge(into: Usage, u: Usage) -> None:
    with into._lock:
        for f in ("input", "cached", "output", "thinking", "embed", "cost"):
            setattr(into, f, getattr(into, f) + getattr(u, f))
        into.priced = into.priced and u.priced
        into.models += [m for m in u.models if m not in into.models]


# ---------- grading ----------

JUDGE_PROMPT = """You are grading a study assistant for a product management course for factual
accuracy. A student asked something; the assistant answered from the course material.
You have the ground truth: lecture transcripts (machine transcribed, so names and words may
be misheard), notes pages and the excerpts the assistant was given.

Split the ANSWER into its individual factual claims (one per fact: a definition, a number,
a table value, an example and what it shows, who said it, which class, a lecture time, a
file/page) and judge each:
- "correct": the material supports it (a fair paraphrase counts; small transcript
  mishearings fixed sensibly count; a lecture time within ~90 seconds of where it's said
  counts).
- "wrong": the material contradicts it (wrong value, wrong row/column, wrong class or
  teacher, wrong example, wrong time, mixed-up facts).
- "unsupported": presented as course material but nothing in the material shows it.
- "general": in a part clearly labelled as general knowledge / not from class. Only mark
  these "wrong" if they are plainly false.
- "skip": not a factual claim (greetings, structure, advice to revise, offers to help).
Judge strictly: a single wrong number or High/Low is "wrong". Don't invent problems:
if the material supports the claim, it is correct even if phrased differently.

Also:
- "must": for each REQUIRED FACT listed, is it in the answer (in substance)? And is the
  required fact itself supported by the ground truth ("fact_ok")? Report every listed fact.
- "request_ok": did the answer do what the student asked (e.g. shorter or simpler than the
  previous answer, summarise the previous answer, answer in Hinglish, list where it was
  taught)? true if nothing special was asked. "request_note": why, briefly. The app shows
  matching class slides/pictures separately under the answer (checked elsewhere), so never
  fail an answer for not showing a picture, and ignore REQUIRED FACTS about pictures.
{extra}
STUDENT'S REQUEST: {question}
{previous}
REQUIRED FACTS:
{must}

GROUND TRUTH (chosen for this question):
{evidence}

EXCERPTS THE ASSISTANT WAS GIVEN (lecture ones widened to the transcript around them):
{excerpts}

ANSWER:
<answer>
{answer}
</answer>"""

_JUDGE_SCHEMA = {"type": "OBJECT", "properties": {
    "claims": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
        "claim": {"type": "STRING"},
        "verdict": {"type": "STRING", "enum": ["correct", "wrong", "unsupported", "general", "skip"]},
        "why": {"type": "STRING"}}, "required": ["claim", "verdict", "why"]}},
    "must": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
        "fact": {"type": "STRING"}, "covered": {"type": "BOOLEAN"}, "fact_ok": {"type": "BOOLEAN"}},
        "required": ["fact", "covered", "fact_ok"]}},
    "request_ok": {"type": "BOOLEAN"}, "request_note": {"type": "STRING"}},
    "required": ["claims", "must", "request_ok", "request_note"]}


@llm._retry
def _judge_call(prompt: str):
    return llm.gen_client().models.generate_content(
        model=JUDGE_MODEL, contents=prompt,
        config=types.GenerateContentConfig(temperature=0, max_output_tokens=32768,
                                           response_mime_type="application/json",
                                           response_schema=_JUDGE_SCHEMA,
                                           thinking_config=types.ThinkingConfig(thinking_level="high")))


def widened(hits: list[dict]) -> str:
    """The retrieved excerpts, with each lecture piece widened to the transcript around it."""
    out = []
    for h in hits:
        if h.get("source") in ("lecture", "slide") and h.get("start") is not None:
            around = sources.lecture_window(h["file_id"], h["start"], h.get("end", h["start"]))
            if around:
                h = {**h, "text": h["text"] + "\n\n[Transcript around it]\n" + around}
        out.append(h)
    return format_context(out, links=True) if out else "(none)"


def judge(case: dict, answer: str, hits: list[dict], previous: str, usage: Usage, extra: str = "") -> dict:
    ev = [sources.evidence(e) for e in case.get("evidence", [])]
    evidence = "\n\n---\n\n".join(f"## {t}\n{x}" for t, x in ev) or "(none chosen; use the excerpts)"
    prompt = JUDGE_PROMPT.format(
        question=case["turns"][-1], extra=extra,
        previous=f"\nTHE ASSISTANT'S PREVIOUS ANSWER (for follow-up requests):\n{previous[:8000]}\n" if previous else "",
        must="\n".join(f"- {m}" for m in case.get("must", [])) or "(none)",
        evidence=evidence[:400_000], excerpts=widened(hits)[:200_000], answer=answer)
    resp = _judge_call(prompt)
    usage.add_generation(JUDGE_MODEL, resp.usage_metadata)
    return json.loads(resp.text)


_TIME = r"(\d{1,2}:\d{2}(?::\d{2})?)"


def answer_times(answer: str) -> list[tuple[object, int, str]]:
    """(class key, seconds, the text) for each lecture time the answer states."""
    out = []
    for m in re.finditer(r"/d/([\w-]+)/view\?t=(\d+)", answer):
        key = sources.class_key_of_video(m.group(1))
        if key is not None:
            out.append((key, int(m.group(2)), m.group(0)))
    for m in re.finditer(r"(Class (\d{1,2})(?![\d-])|(?:27[- ]Sep\w*|4[- ]Oct\w*|17(?:th)?[- ]Oct\w*)(?:[- ]2026)? workshop)"
                         r"[^\n\[\]]{0,60}?(?:at|from|around|@|,|\()\s*" + _TIME
                         + r"(?:\s*[–-]\s*" + _TIME + r")?", answer, re.I):
        key = int(m.group(2)) if m.group(2) else next(
            (k for k, d in sources.WORKSHOPS.items() if d.split()[0] in m.group(1)), None)
        if key is None:
            continue
        for t in (m.group(3), m.group(4)):
            if t:
                out.append((key, sources.secs(t), m.group(0)))
    return out


def time_problems(case: dict, answer: str, hits: list[dict]) -> list[str]:
    """Lecture times in the answer that no excerpt or evidence window of that class backs."""
    anchors = []  # (class key, from, to)
    for h in hits:
        if h.get("source") in ("lecture", "slide") and h.get("start") is not None:
            key = sources.class_key_of_video(h["file_id"])
            anchors.append((key, h["start"] - TIME_SLACK, h.get("end", h["start"]) + TIME_SLACK))
        if h.get("source") == "summary":
            key = sources.class_key_of_video(h["file_id"])
            for t in re.findall(r"\[\[?" + _TIME, h["text"]):
                anchors.append((key, sources.secs(t) - TIME_SLACK, sources.secs(t) + 300))
    for e in case.get("evidence", []):
        if "class" in e and not e.get("full"):
            if "section" in e:
                a, b = sources.section_window(e["class"], e["section"])
            else:
                a, b = sources.secs(e["from"]), sources.secs(e["to"])
            anchors.append((e["class"], a - TIME_SLACK, b + TIME_SLACK))
    bad = []
    for key, sec, text in answer_times(answer):
        try:
            if sec > sources._segments(key)[-1]["end"] + 60:
                bad.append(f"time past the end of the video: {text}")
                continue
        except KeyError:
            continue
        if not any(k == key and a <= sec <= b for k, a, b in anchors):
            bad.append(f"time not backed by any excerpt: {text}")
    return bad


def _matches(hit: dict, ev: dict) -> bool:
    """Does this retrieved chunk come from this hand-picked evidence item?"""
    if "file" in ev:
        if norm(ev["file"]) not in norm(hit.get("file_name", "")):
            return False
        return not ev.get("page") or hit.get("page") == ev["page"]
    if "summary" in ev:
        return hit.get("source") == "summary" and sources.class_key_of_video(hit["file_id"]) == ev["summary"]
    if "class" not in ev or ev.get("full") or hit.get("start") is None:
        return False
    if sources.class_key_of_video(hit["file_id"]) != ev["class"]:
        return False
    a, b = (sources.section_window(ev["class"], ev["section"]) if "section" in ev
            else (sources.secs(ev["from"]), sources.secs(ev["to"])))
    return hit["start"] <= b + TIME_SLACK and hit.get("end", hit["start"]) >= a - TIME_SLACK


def context_scores(case: dict, hits: list[dict]) -> dict | None:
    """Search quality, no AI needed: matches retrieved chunks to the case's evidence.
    precision: share of chunks that are evidence, weighted so evidence near the top counts
    more (the RAGAS formula). recall: share of evidence items at least one chunk found.
    None when the case has no specific evidence (whole-class cases, quizzes, not-covered)."""
    ev = [e for e in case.get("evidence", []) if not e.get("full")]
    if not ev or not hits:
        return None
    good = [any(_matches(h, e) for e in ev) for h in hits]
    found = [any(_matches(h, e) for h in hits) for e in ev]
    ranked = [sum(good[:k + 1]) / (k + 1) for k, g in enumerate(good) if g]
    return {"precision": round(sum(ranked) / len(ranked), 3) if ranked else 0.0,
            "recall": round(sum(found) / len(found), 3),
            "relevant": sum(good), "of": len(hits),
            "first": good.index(True) + 1 if any(good) else None}


def exact_checks(case: dict, answer: str, hits: list[dict], history: list[dict]) -> list[str]:
    problems = []
    plain = answer.replace("’", "'")
    missing = [m for m in case.get("mention", []) if norm(m) not in norm(answer)]
    if missing:
        problems.append(f"doesn't mention: {', '.join(missing)}")
    # A new topic after an old one must not drag the old topic in.
    leaked = [m for m in case.get("must_not_mention", []) if norm(m) in norm(answer)]
    if leaked:
        problems.append(f"mentions the earlier topic: {', '.join(leaked)}")
    if case.get("not_covered") and NOT_COVERED not in plain:
        problems.append("didn't say it wasn't covered")
    if case.get("covered") and NOT_COVERED in plain:
        problems.append("wrongly said it wasn't covered")
    if case.get("source") and not any(s.lower() in h["file_name"].lower() for s in case["source"] for h in hits):
        problems.append(f"expected file not retrieved ({' / '.join(case['source'])})")
    if case.get("classes"):
        stray = {h["class_label"] for h in hits} - set(case["classes"])
        if stray:
            problems.append(f"hits from outside the class filter: {', '.join(sorted(stray))}")
    if case.get("shorter_than_last") and history:
        last = next(m["content"] for m in reversed(history) if m["role"] == "assistant")
        if len(answer) > len(last) * case["shorter_than_last"]:
            problems.append(f"not short enough ({len(answer)} chars vs {len(last)} before)")
    return problems


# ---------- quizzes ----------

QUIZ_EXTRA = """This ANSWER is a quiz the assistant made. Treat each question as ONE claim: the
question with its marked answer. "correct" if the marked answer is right per the material
and (for multiple choice) exactly one option is right; "wrong" if the marked answer is
wrong or two options are right; "unsupported" if the material doesn't cover it."""


def quiz_text(state: dict) -> str:
    out = []
    for i, q in enumerate(state["questions"], 1):
        opts = "".join(f"\n   {l}) {o}" for l, o in zip("ABCD", q.get("options") or []))
        out.append(f"Q{i} ({q['type']}): {q['question']}{opts}\n   Marked answer: {q['answer']}"
                   f"\n   Explanation: {q['explanation']}")
    return "\n\n".join(out)


# ---------- one case ----------

def run_case(case: dict, verify_on: bool) -> dict:
    """run_once, tried again (up to 3 times) when the network or an API call fails."""
    for attempt in range(3):
        out = run_once(dict(case), verify_on)
        if not out["error"]:
            return out
        time.sleep(5 * (attempt + 1))
    return out


def run_once(case: dict, verify_on: bool) -> dict:
    usage = Usage()
    case.setdefault("turns", [case["question"]] if "question" in case else [])
    history, res, turn_steps = [], {}, []
    t0 = time.time()
    try:
        if case["type"] == "quiz":
            n, level = modes.quiz_options(case["turns"][-1])
            state, text = quiz.start(case["turns"][-1], case.get("classes") or [], n, level, usage)
            hits, _ = quiz._material(case["turns"][-1], case.get("classes") or [], usage)
            answer = quiz_text(state) if state else text
            res = {"answer": answer, "raw": answer, "fixes": [], "hits": hits, "mode": "quiz"}
            verdict = judge(case, answer, hits, "", usage, extra=QUIZ_EXTRA)
        elif case["type"] == "class_summary":
            key = case["summary_of"]
            answer = sources.summary_md(key)
            if not case.get("evidence"):
                case["evidence"] = [{"class": key, "full": True}]
            res = {"answer": answer, "raw": answer, "fixes": [], "hits": [], "mode": "class_summary"}
            verdict = judge(case, answer, [], "", usage, extra=(
                "\nThis ANSWER is the stored study-notes summary of one whole class. Check every "
                "point and every [time] against the full transcript. Ignore the lists of notes "
                "files at the end (\"This class's notes\", \"Closest notes\") and the "
                "\"Watch the lecture\" link (it is the class's own video link).\n"))
        else:
            history = [m if isinstance(m, dict) else {"role": m[0], "content": m[1]}
                       for m in case.get("history", [])]
            for q in case["turns"]:
                res = respond(q, history, case.get("classes"), verify_on, usage,
                              allow_general=case.get("allow_general", True))
                history += [{"role": "user", "content": q}, {"role": "assistant", "content": res["answer"]}]
                turn_steps.append(res.get("steps"))
            if res["mode"] == "class_summary" and not case.get("evidence"):
                # A stored summary is graded against the whole lecture, like the class_summary cases.
                keys = [k for lb in res["classes"] for k in [sources.key_of_label(lb)] if k is not None]
                case["evidence"] = [{"class": k, "full": True} for k in keys]
            prev = history[-4]["content"] if len(history) >= 4 else ""
            note = ("\nThe ANSWER is the stored study-notes summary of a class. Ignore the lists of notes "
                    "files at the end and the \"Watch the lecture\" link.\n") if res["mode"] == "class_summary" else ""
            verdict = judge(case, res["answer"], res["hits"], prev, usage, extra=note)
        hist_before = history[:-2]
        problems = exact_checks(case, res["answer"], res["hits"], hist_before)
        if res.get("mode") != "class_summary":  # summary times are checked by the grader against the whole transcript
            problems += time_problems(case, res["answer"], res["hits"])
        if case.get("expect_mode") and res.get("mode") != case["expect_mode"]:
            problems.append(f"routed as {res.get('mode')}, expected {case['expect_mode']}")
        if case.get("pictures"):
            pics = images.find(res.get("topic") or case["turns"][-1],
                               topic.main_classes(res.get("hits") or []) or case.get("classes"))
            problems += grade_pictures(case, pics)
        if not verdict.get("request_ok", True):
            problems.append(f"request not followed: {verdict.get('request_note', '')}")
        error = ""
    except Exception as e:  # keep going; the case shows as an error
        import traceback
        verdict, problems = {"claims": [], "must": []}, []
        error = f"{type(e).__name__}: {e} @ {traceback.extract_tb(e.__traceback__)[-1].line}"
    claims = verdict.get("claims", [])
    count = {v: sum(c["verdict"] == v for c in claims) for v in ("correct", "wrong", "unsupported", "general", "skip")}
    must = verdict.get("must", [])
    out = {"id": case["id"], "type": case["type"], "turns": case["turns"], "answer": res.get("answer", ""),
           "raw": res.get("raw", ""), "fixes": res.get("fixes", []), "mode": res.get("mode"),
           "hits": [f"{h['class_label']} — {h['file_name']}" + (f", p{h['page']}" if h.get("page") else "")
                    + (f" @ {h['time']}" if h.get("time") else "") for h in res.get("hits", [])],
           "claims": claims, "count": count, "must": must, "problems": problems, "error": error,
           "request_note": verdict.get("request_note", ""), "seconds": round(time.time() - t0, 1),
           "cost_usd": round(usage.cost, 4), "steps": [x for x in turn_steps if x]}
    try:
        out["context"] = context_scores(case, res.get("hits") or [])
    except Exception as e:  # a scoring slip must not lose the case
        out["context"] = None
        print(f"context score failed for {case['id']}: {e}", flush=True)
    out["clean"] = not error and count["wrong"] == 0 and count["unsupported"] == 0 and not problems
    with _lock:
        mark = "PASS" if out["clean"] else "FAIL"
        print(f"{mark}  {case['id']:<34} correct {count['correct']:>2} wrong {count['wrong']} "
              f"unsupported {count['unsupported']}  {'; '.join(problems)[:150]} {error[:150]}", flush=True)
    return out


# ---------- report ----------

def report(results: list[dict]) -> str:
    def acc(rs):
        c = sum(r["count"]["correct"] for r in rs)
        bad = sum(r["count"]["wrong"] + r["count"]["unsupported"] for r in rs)
        return c, bad, (100.0 * c / (c + bad) if c + bad else 100.0)

    def ctx(rs):
        cs = [r["context"] for r in rs if r.get("context")]
        if not cs:
            return "–", "–"
        return (f"{100 * sum(x['precision'] for x in cs) / len(cs):.0f}%",
                f"{100 * sum(x['recall'] for x in cs) / len(cs):.0f}%")

    c, bad, a = acc(results)
    clean = sum(r["clean"] for r in results)
    cp, cr = ctx(results)
    scored = sum(bool(r.get("context")) for r in results)
    must = [m for r in results for m in r["must"]]
    covered = sum(m["covered"] for m in must)
    lines = [f"# Accuracy eval: {a:.2f}% of claims correct · {clean}/{len(results)} answers clean", "",
             f"- Claims checked: {c + bad} ({c} correct, {bad} wrong or unsupported)",
             f"- Required facts covered: {covered}/{len(must)}"
             + (f" ({100 * covered / len(must):.1f}%)" if must else ""),
             f"- Fact-check fixes applied: {sum(len(r['fixes']) for r in results)}",
             f"- Search (context precision / recall, {scored} cases with hand-picked evidence): "
             f"{cp} / {cr}",
             f"- Errors: {sum(bool(r['error']) for r in results)}",
             f"- Cost of this run (answers + grading): ${sum(r['cost_usd'] or 0 for r in results):.2f}", "",
             "| Type | Cases | Clean | Claims | Accuracy | Ctx precision | Ctx recall |",
             "|---|---|---|---|---|---|---|"]
    for t in sorted({r["type"] for r in results}):
        rs = [r for r in results if r["type"] == t]
        tc, tb, ta = acc(rs)
        p, rc = ctx(rs)
        lines.append(f"| {t} | {len(rs)} | {sum(r['clean'] for r in rs)} | {tc + tb} | {ta:.1f}% | {p} | {rc} |")
    lines.append("")
    for r in sorted(results, key=lambda r: (r["clean"], r["type"], r["id"])):
        lines.append(f"## {'✅' if r['clean'] else '❌'} {r['id']} ({r['type']})")
        lines.append("")
        for i, t in enumerate(r["turns"], 1):
            lines.append(f"**Q{i if len(r['turns']) > 1 else ''}:** {t}  ")
        if r["error"]:
            lines.append(f"\n**Error:** {r['error']}")
        bad = [x for x in r["claims"] if x["verdict"] in ("wrong", "unsupported")]
        if bad:
            lines.append("\n**Wrong / unsupported claims:**")
            lines += [f"- [{x['verdict']}] {x['claim']} — *{x['why']}*" for x in bad]
        if r["problems"]:
            lines.append("\n**Other problems:** " + "; ".join(r["problems"]))
        missed = [m["fact"] for m in r["must"] if not m["covered"]]
        if missed:
            lines.append("\n**Required facts missing:** " + "; ".join(missed))
        badgold = [m["fact"] for m in r["must"] if not m.get("fact_ok", True)]
        if badgold:
            lines.append("\n**⚠ Required facts the grader could not confirm (check the case):** " + "; ".join(badgold))
        if r.get("context"):
            x = r["context"]
            lines.append(f"\n**Search:** {x['relevant']}/{x['of']} chunks are evidence, first at #{x['first'] or '–'} "
                         f"· precision {100 * x['precision']:.0f}% · recall {100 * x['recall']:.0f}%")
        if r["fixes"]:
            lines.append("\n**Fact-check fixed:** " + "; ".join(f"“{f['wrong']}” → “{f['right']}”" for f in r["fixes"]))
        lines.append(f"\n<details><summary>Answer ({r['mode']}, {r['count']['correct']} correct claims)</summary>\n\n"
                     f"{r['answer']}\n\n**Retrieved:** " + "; ".join(r["hits"]) + "\n</details>\n")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("only", nargs="*", help="case ids or types to run")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--out", default="accuracy")
    args = ap.parse_args()
    cases = json.loads((HERE / "cases.json").read_text())
    if args.only:
        cases = [c for c in cases if c["id"] in args.only or c["type"] in args.only]
    print(f"{len(cases)} case(s)", flush=True)
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(lambda c: run_case(c, not args.no_verify), cases))
    (HERE / f"{args.out}.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
    md = report(results)
    (HERE / f"{args.out}.md").write_text(md)
    print("\n" + "\n".join(md.splitlines()[:12 + len({r['type'] for r in results})]))
    print(f"\nFull report: {HERE / (args.out + '.md')}")


if __name__ == "__main__":
    main()
