"""Quiz mode: questions made from the class lectures and notes, asked one at a time,
each answer marked with an explanation and where it was taught.

The quiz state is a plain dict saved on the chat's messages, so a quiz carries on after
a page reload or when the chat is opened again."""
import json
import random
import re

from google.genai import types

from . import config, llm, store
from .answer import format_context, is_restyle, is_where_followup, where

_QUIZ_WORDS = re.compile(
    r"\b(quiz|test|mcqs?|practice|questions?|qs?|ask|me|on|about|from|of|the|a|an|some|please|pls|"
    r"give|take|do|can|you|my|easy|medium|hard|tough|difficult|advanced|simple|basic|level|\d+|"
    r"class|session|lecture|workshop|sunday|okay|ok|like|multiple|choice|type|types|kind|"
    r"domain|topic|subject|this|that|it|these|above|same|now|i|want|mcq|options?)\b", re.I)
_STOP_WORD = re.compile(r"\b(stop|end|quit|exit|finish|cancel|close|enough|done|no more)\b", re.I)
_QUIZ_WORD = re.compile(r"\b(quiz|test|questions?|mcqs?)\b", re.I)
# Asking where something was taught, rather than answering.
_WHERE = re.compile(r"\b(links?|time ?stamps?|which (lecture|class|video|session)|lecture video|video lecture|"
                    r"where (is|was|it|in|can|do|did)|source|reference|watch)\b", re.I)
# A request or question to the bot, not an answer to the quiz question.
_REQUEST = re.compile(r"^\s*(can|could|would|will) (you|u)\b|^\s*(please|pls|kindly)\b|"
                      r"^\s*(give|show|tell|explain|summari[sz]e|help) me\b|^\s*i (want|need|don'?t want)\b",
                      re.I)
_DONT_KNOW = re.compile(r"^\s*(i\s+)?(don'?t know|dont know|idk|no idea|not sure|pass|skip)\b", re.I)


def is_stop(text: str) -> bool:
    """'stop quiz', 'please stop with the quiz', 'end it', 'no more questions'…"""
    if not _STOP_WORD.search(text):
        return False
    return bool(_QUIZ_WORD.search(text)) or len(text.split()) <= 4


def kind(state: dict, text: str) -> str:
    """What a message sent during a quiz is: "stop", "answer", "where" (asks where the
    question was taught) or "aside" (another question, answered normally; the quiz waits)."""
    if is_stop(text):
        return "stop"
    q = state["questions"][state["i"]]
    if q["type"] == "mcq" and _mcq_choice(text, q["options"]):
        return "answer"
    if _WHERE.search(text):
        return "where"
    if _DONT_KNOW.match(text):
        return "answer"
    if is_restyle(text):
        return "aside"  # "summarise this", "simpler please": not an answer
    if q["type"] == "mcq":
        return "answer" if len(text.split()) <= 3 else "aside"
    return "aside" if _REQUEST.search(text) else "answer"

GEN_PROMPT = """Write a {level} quiz of {n} questions for a student of a product management
course, using ONLY the class material excerpts below (lecture transcripts, class summaries
and notes). Quiz topic: {topic}.

- {mix}
- Test understanding of what was taught (concepts, frameworks, steps, how to apply them,
  examples), not trivia such as timestamps, file names or which student asked something.
- easy = recall a definition or step; medium = understand or compare; hard = apply it to a
  new situation or spot the mistake.
- mcq: exactly 4 options, one clearly correct, the others plausible; "answer" is the letter.
- short: "answer" is a model answer of 1-3 sentences.
- "explanation": 1-2 sentences on why, in the words the class used.
- "source": the number of the excerpt the question is based on.
- "topic": the concept tested, in 2-5 words.
- Cover different parts of the material; no two questions on the same point.

Excerpts:
{excerpts}"""

_GEN_SCHEMA = {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
    "type": {"type": "STRING", "enum": ["mcq", "short"]}, "question": {"type": "STRING"},
    "options": {"type": "ARRAY", "items": {"type": "STRING"}}, "answer": {"type": "STRING"},
    "explanation": {"type": "STRING"}, "source": {"type": "INTEGER"}, "topic": {"type": "STRING"}},
    "required": ["type", "question", "answer", "explanation", "source", "topic"]}}

GRADE_PROMPT = """Mark a student's short answer in a product management class quiz.
Question: {question}
Model answer: {answer}
Student's answer: {reply}

"verdict": "correct" if it has the key idea (wording can differ, spelling mistakes are fine),
"partly" if it has some of it, "wrong" otherwise. "feedback": one or two friendly sentences
saying what was right and what was missing."""

CHECK_PROMPT = """Check a quiz made from class material before a student takes it. For each
question, using ONLY the excerpts:
- "problem": "none" if it is answerable from the excerpts, its marked answer is correct
  and (for mcq) exactly one option is correct; "wrong_key" if another single option (or a
  different model answer) is the correct one; "ambiguous" if more than one option could be
  right or the question is unclear; "unsupported" if the excerpts don't back it up.
- "answer": the correct option letter for mcq (or the correct model answer for short).
- "why": a short note.

Excerpts:
{excerpts}

Quiz:
{quiz}"""

_CHECK_SCHEMA = {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
    "id": {"type": "INTEGER"},
    "problem": {"type": "STRING", "enum": ["none", "wrong_key", "ambiguous", "unsupported"]},
    "answer": {"type": "STRING"}, "why": {"type": "STRING"}}, "required": ["id", "problem", "answer", "why"]}}


def check_keys(questions: list[dict], hits: list[dict], usage=None) -> list[dict]:
    """The questions whose marked answer the excerpts confirm. A question whose answer
    letter is wrong but whose correct option is clear gets the letter fixed; the rest go."""
    if not questions:
        return questions
    listing = "\n\n".join(
        f"[{i}] ({q['type']}) {q['question']}\n"
        + "".join(f"{l}) {o}\n" for l, o in zip("ABCD", q.get("options") or []))
        + f"Marked answer: {q['answer']}" for i, q in enumerate(questions, 1))
    try:
        res = _json_call(config.VERIFY_MODEL, CHECK_PROMPT.format(excerpts=format_context(hits), quiz=listing),
                         _CHECK_SCHEMA, usage, temp=0)
    except Exception as e:
        print(f"quiz check failed: {e}")
        return questions
    verdict = {r["id"]: r for r in res}
    kept = []
    for i, q in enumerate(questions, 1):
        r = verdict.get(i)
        if r is None:
            continue
        if r["problem"] == "none":
            kept.append(q)
        elif r["problem"] == "wrong_key":
            if q["type"] == "mcq":
                letter = (r.get("answer") or "").strip()[:1].upper()
                if letter and letter in "ABCD" and letter != q["answer"]:
                    q["answer"] = letter  # the right option is clear, only the mark was wrong
                    kept.append(q)
            elif r.get("answer", "").strip():
                q["answer"] = r["answer"].strip()
                kept.append(q)
    return kept


_GRADE_SCHEMA = {"type": "OBJECT", "properties": {
    "verdict": {"type": "STRING", "enum": ["correct", "partly", "wrong"]},
    "feedback": {"type": "STRING"}}, "required": ["verdict", "feedback"]}


def _json_call(model: str, prompt: str, schema: dict, usage, max_tokens: int = 8192, temp: float = 0.4):
    thinking = (types.ThinkingConfig(thinking_level="low") if model.startswith("gemini-3")
                else types.ThinkingConfig(thinking_budget=1024))
    resp = llm._retry(llm.gen_client().models.generate_content)(
        model=model, contents=prompt,
        config=types.GenerateContentConfig(temperature=temp, max_output_tokens=max_tokens,
                                           response_mime_type="application/json", response_schema=schema,
                                           thinking_config=thinking))
    if usage is not None:
        usage.add_generation(model, resp.usage_metadata)
    return json.loads(resp.text)


def topic_of(text: str) -> str:
    return re.sub(r"\s+", " ", _QUIZ_WORDS.sub(" ", text)).strip(" ,.?!-")


# Question words that carry no topic: "and what it is", "when was this explained".
_ASK_WORDS = re.compile(
    r"\b(what|which|when|where|who|whom|why|how|is|was|were|are|am|be|does|did|do|and|or|but|at|in|to|"
    r"for|by|with|its|it's|time|times|explained|explain|taught|teach|covered|discussed|mean|means|"
    r"meaning|define|definition|tell|more|again|also|then|so|sir|ma'?am|exactly|really|one|"
    # "what are the core concepts I should understand?"
    r"core|key|main|important|essential|basics?|fundamentals?|concepts?|topics?|things?|should|"
    r"need|must|know|learn|understand|study|revise|remember|we|have|been)\b", re.I)


def topic_from_history(history: list[dict]) -> str:
    """The topic of the last question that wasn't part of a quiz, for "quiz me on this".
    A follow-up like "and what it is" has no topic of its own, so earlier questions are
    tried until one names it ("which class taught the RACI matrix?" -> "RACI matrix")."""
    for i in range(len(history) - 1, 0, -1):
        m, prev = history[i], history[i - 1]
        if m["role"] == "assistant" and "quiz" not in m and prev["role"] == "user":
            q = prev["content"]
            if is_restyle(q) or is_where_followup(q):
                continue  # "summarise this", "give me the link": about the answer before
            words = re.sub(r"\s+", " ", _ASK_WORDS.sub(" ", topic_of(q))).strip(" ,.?!-")
            if len(words) >= 3:
                return words
    return ""


def _summary_hit(label: str, chars: int) -> dict | None:
    p = store.summary_payload(label)
    return {**p, "text": p["full_text"][:chars]} if p else None


def _material(text: str, classes: list[str], usage) -> tuple[list[dict], str]:
    """(excerpts to make questions from, topic name shown to the student)."""
    topic = topic_of(text)
    if classes:
        hits = store.search(topic or "key concepts frameworks examples taught", class_labels=classes,
                            limit=16, usage=usage)
        summaries = [_summary_hit(c, 6000) for c in classes]
        summaries = [x for x in summaries if x]
        name = ", ".join(classes) + (f" ({topic})" if topic else "")
        return summaries + hits, name
    if len(topic) >= 3 and not re.fullmatch(r"(everything|all|anything|whole course|so far|course)( so far)?",
                                            topic, re.I):
        return store.search(topic, limit=18, usage=usage), topic
    # A quiz on everything: summaries of a few random classes.
    labels = random.sample(store.indexed_classes(), k=min(5, len(store.indexed_classes())))
    hits = [x for c in labels if (x := _summary_hit(c, 5000))]
    if not hits:
        hits = store.search("key PM frameworks and concepts", limit=18, usage=usage)
    return hits, "the whole course so far"


def start(text: str, classes: list[str], n: int, level: str, usage=None,
          prev_topic: str = "") -> tuple[dict | None, str]:
    """New quiz -> (state, first message). state is None if there was nothing to quiz on.
    prev_topic: the question asked before, used for "quiz me on this"."""
    # "Quiz me on this" / "take quiz" right after a question is about that question.
    # "Quiz me on everything" names a topic of its own, so it isn't replaced.
    if prev_topic and not classes and len(topic_of(text)) < 3:
        text = prev_topic
    mcq_only = bool(re.search(r"\b(multiple|mcqs?|options?|objective)\b", text, re.I))
    hits, topic = _material(text, classes, usage)
    if not hits:
        return None, "I couldn't find class material on that to quiz you on. Try another topic or class."
    qs = _json_call(config.ANSWER_MODEL,
                    GEN_PROMPT.format(level=level, n=n, topic=topic, excerpts=format_context(hits),
                                      mix=('All multiple choice ("mcq").' if mcq_only else
                                           'About two thirds multiple choice ("mcq"), the rest short answer ("short").')),
                    _GEN_SCHEMA, usage)
    questions = []
    for q in qs[:n]:
        h = hits[q["source"] - 1] if 1 <= q.get("source", 0) <= len(hits) else None
        q["options"] = [re.sub(r"^\s*[A-Da-d][\)\.:]\s*", "", o) for o in q.get("options") or []]
        q["answer"] = q["answer"].strip()
        letter = _mcq_choice(q["answer"], q["options"]) if q["options"] else None
        # Trust the shape over the label: 4 options + an answer naming one of them is multiple choice.
        if len(q["options"]) == 4 and letter:
            q["type"], q["answer"] = "mcq", letter
        elif re.fullmatch(r"\(?[A-Da-d]\)?[\.\)]?", q["answer"]):
            continue  # a letter answer without its options can't be marked
        else:
            q["type"] = "short"
        q["cite"] = (f"{h['class_label']} — {where(h)}" if h else "")
        q["link"] = h.get("link", "") if h else ""
        q["class"] = h["class_label"] if h else ""
        questions.append(q)
    questions = check_keys(questions, hits, usage)
    if not questions:
        return None, "I couldn't make questions I'm sure are right from that material. Try another topic or class."
    state = {"active": True, "topic": topic, "level": level, "questions": questions,
             "i": 0, "score": 0.0, "results": []}
    intro = (f"🧠 **Quiz: {topic}** · {len(questions)} questions · {level}\n\n"
             "Reply with your answer. Type **stop quiz** to end early.\n\n---\n\n")
    return state, intro + ask_text(state)


def ask_text(state: dict) -> str:
    q = state["questions"][state["i"]]
    head = f"**Question {state['i'] + 1} of {len(state['questions'])}**\n\n{q['question']}"
    if q["type"] == "mcq":
        opts = "\n".join(f"- **{l})** {o}" for l, o in zip("ABCD", q["options"]))
        return f"{head}\n\n{opts}\n\n*Reply with A, B, C or D.*"
    return f"{head}\n\n*Reply in a sentence or two.*"


def _mcq_choice(reply: str, options: list[str]) -> str | None:
    m = re.match(r"^\s*(?:option\s*)?\(?([a-dA-D])(?:\s*$|\s*[\)\.:,]|\s+option\b)", reply)
    if not m and len(reply.split()) <= 4:
        m = re.match(r"^\s*(?:option\s*)?\(?([a-dA-D])\b", reply)
    if not m:
        m = re.search(r"\b(?:answer|option|it'?s|is)\s*(?:is\s*)?:?\s*\(?([a-dA-D])\b(?!['’])", reply)
    if m:
        return m.group(1).upper()
    low = reply.strip().lower()
    for l, o in zip("ABCD", options):
        if low and (low == o.lower() or (len(low) > 8 and low in o.lower())):
            return l
    return None


def _source_line(q: dict) -> str:
    if not q.get("cite"):
        return ""
    return f"\n\n📍 Taught in: [{q['cite']}]({q['link']})" if q.get("link") else f"\n\n📍 Taught in: {q['cite']}"


def reply(state: dict, text: str, usage=None) -> str:
    """Mark the answer to the current question; returns feedback plus the next question
    (or the final score). Updates state in place."""
    if is_stop(text):
        return finish(state, stopped=True)
    q = state["questions"][state["i"]]
    if q["type"] == "mcq":
        choice = _mcq_choice(text, q["options"])
        if choice is None and _DONT_KNOW.match(text):
            choice = "?"
        if choice is None:
            return "Please reply with **A**, **B**, **C** or **D** (or type **stop quiz**).\n\n" + ask_text(state)
        right = q["answer"][:1].upper()
        verdict = "correct" if choice == right else "wrong"
        right_text = f"**{right}) {q['options']['ABCD'.index(right)].rstrip('.')}**"
        fb = ("✅ **Correct!** " if verdict == "correct" else f"❌ **Not quite.** The answer is {right_text}. ")
        fb += q["explanation"]
    else:
        g = _json_call(config.PICTURE_MODEL, GRADE_PROMPT.format(question=q["question"], answer=q["answer"],
                                                                 reply=text[:2000]),
                       _GRADE_SCHEMA, usage, max_tokens=1024, temp=0)
        verdict = g["verdict"]
        icon = {"correct": "✅ **Correct!**", "partly": "🟡 **Partly right.**", "wrong": "❌ **Not quite.**"}[verdict]
        fb = f"{icon} {g['feedback']}\n\n**Model answer:** {q['answer']}"
    state["score"] += {"correct": 1, "partly": 0.5}.get(verdict, 0)
    state["results"].append({"topic": q["topic"], "verdict": verdict, "cite": q.get("cite", ""),
                             "link": q.get("link", "")})
    fb += _source_line(q)
    state["i"] += 1
    if state["i"] >= len(state["questions"]):
        return fb + "\n\n---\n\n" + finish(state)
    return fb + "\n\n---\n\n" + ask_text(state)


def where_reply(state: dict, usage=None) -> str:
    """Where the last answered question and the current one were taught: the source the
    question was made from plus the lecture video moments, with links to the right time."""
    picks = []
    if state["i"] > 0:
        picks.append((state["i"] - 1, "the last question"))
    if state["i"] < len(state["questions"]):
        picks.append((state["i"], "the current question"))
    parts = []
    for idx, label in picks:
        q = state["questions"][idx]
        lines = [f"**Question {idx + 1}** ({label}: *{q['topic']}*)"]
        if q.get("cite"):
            lines.append(f"- 📍 Made from: " + (f"[{q['cite']}]({q['link']})" if q.get("link") else q["cite"]))
        query = f"{q['topic']}. {q['question']}"
        hits = []
        if q.get("class"):
            hits = store.search(query, class_labels=[q["class"]], limit=4, usage=usage, sources="lectures")
        hits = [h for h in hits if h.get("source") in ("lecture", "slide")]
        if not hits:
            hits = [h for h in store.search(query, limit=4, usage=usage, sources="lectures")
                    if h.get("source") in ("lecture", "slide")]
        for h in hits[:2]:
            what = "slide shown" if h["source"] == "slide" else "lecture video"
            lines.append(f"- 🎬 [{h['class_label']} — {what} at {h['time']}]({h['link']})")
        if len(lines) == 1:
            lines.append("- No lecture video moment found for this one.")
        parts.append("\n".join(lines))
    msg = "**Where this was taught** (links open the video at that time):\n\n" + "\n\n".join(parts)
    if state["i"] < len(state["questions"]):
        msg += "\n\n---\n\n*Quiz still on.* " + ask_text(state)
    return msg


def _fmt_score(x: float) -> str:
    return str(int(x)) if x == int(x) else f"{x:.1f}"


def finish(state: dict, stopped: bool = False) -> str:
    state["active"] = False
    done = len(state["results"])
    if not done:
        return "Quiz stopped. Ask for another one any time, e.g. *quiz me on GTM*."
    total = len(state["questions"])
    msg = (f"🏁 **Quiz {'stopped' if stopped else 'finished'}: {_fmt_score(state['score'])} / {done}**"
           + (f" (answered {done} of {total})" if stopped and done < total else ""))
    pct = state["score"] / done
    msg += "\n\n" + ("Excellent, you know this well!" if pct >= 0.8 else
                     "Good effort, a few things to revise." if pct >= 0.5 else
                     "Worth revising this topic before moving on.")
    weak = [r for r in state["results"] if r["verdict"] != "correct"]
    if weak:
        msg += "\n\n**Revise these:**\n" + "\n".join(
            f"- {r['topic']}" + (f" — [{r['cite']}]({r['link']})" if r["link"] else
                                 f" — {r['cite']}" if r["cite"] else "") for r in weak)
    return msg + "\n\nWant another? Try *quiz me on {} hard* or pick a new topic.".format(state["topic"])
