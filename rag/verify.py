"""Fact-check an answer against the excerpts it was written from, and fix what's wrong.

A second Gemini call reads the answer next to the class material and lists every detail
the material contradicts or doesn't support: numbers, table values (row and column),
names, examples, who said what, class numbers and lecture times. Each comes back as an
exact piece of the answer plus its corrected text, and is swapped in.
"""
import json
import re

from google.genai import types

from . import config, llm

PROMPT = """You are checking a study assistant's answer for factual mistakes before a
student sees it. The answer must agree with the class material excerpts below.

Check every specific detail in the answer:
- numbers, percentages, prices, counts, and values read from tables: check the right row
  AND the right column (e.g. a High/Low grid: which row, which column);
- names of companies, products, people, frameworks and their parts or steps;
- which example illustrates which point, and what the example says;
- who said or taught it (teacher vs a student), and the class number;
- lecture times: a time like "Class 18 at 1:16:27" must fall inside (or within 1 minute of)
  the time range of an excerpt from that class. Cited excerpt numbers like [3] must point
  to an excerpt that supports the sentence.
- notes file names and page numbers.

Lecture transcripts are machine-made and often mishear names and terms ("Namayatri" for
Namma Yatri, "Hickey Act" for Yik Yak, "Moscow" for MoSCoW, "MetaRaven" for Meta Ray-Ban).
A correct, well-known name or term in the answer is NOT a mistake because the transcript
misspells it: never replace a correct term with a misheard one. Text read off the screen
or slides is usually spelled right; prefer it over the transcript for names.

Always report (and remove) any remark that the teacher mispronounced, misnamed, joked
about or misspelled something: those come from transcription slips, not from the class.

Report a detail ONLY when the excerpts clearly contradict it, or it is presented as class
material but nothing in the excerpts says it. Don't report: wording or style, details that
are correct, reasonable paraphrases or summaries, or text inside a section headed
"General answer (not from class)" or in a line starting "⚠️ Note (general knowledge, not
from class)" (general knowledge is allowed there).

For each problem:
- "wrong": the shortest piece of the answer that contains the mistake, copied EXACTLY
  character for character (including markdown like ** and citation marks like [3]), long
  enough to be unique in the answer.
- "right": that same piece with only the mistake corrected, keeping its formatting and
  citations. If the detail isn't in the material at all, rewrite the piece without it
  (it may be "" if the whole piece should go).
- "why": a short note, e.g. "Notes table: mutual funds have High conversion [7]".
If everything checks out, return {{"issues": []}}.

Student's request: {question}

Class material excerpts:
{excerpts}

Answer to check:
<answer>
{answer}
</answer>"""

_SCHEMA = {"type": "OBJECT", "properties": {"issues": {"type": "ARRAY", "items": {
    "type": "OBJECT", "properties": {"wrong": {"type": "STRING"}, "right": {"type": "STRING"},
                                     "why": {"type": "STRING"}},
    "required": ["wrong", "right", "why"]}}}, "required": ["issues"]}


@llm._retry
def _call(model: str, prompt: str):
    return llm.gen_client().models.generate_content(
        model=model, contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0, max_output_tokens=8192, response_mime_type="application/json",
            response_schema=_SCHEMA,
            thinking_config=types.ThinkingConfig(thinking_level=config.VERIFY_THINKING)
            if model.startswith("gemini-3") else types.ThinkingConfig(thinking_budget=2048)),
    )


def _replace(text: str, wrong: str, right: str) -> str | None:
    """text with wrong swapped for right; None if wrong can't be found."""
    if wrong and wrong in text:
        return text.replace(wrong, right, 1)
    # Same words but different spacing / line breaks.
    words = wrong.split()
    if not words:
        return None
    pattern = r"\s+".join(re.escape(w) for w in words)
    m = re.search(pattern, text)
    return text[:m.start()] + right + text[m.end():] if m else None


def check(answer: str, excerpts: str, question: str, usage=None) -> tuple[str, list[dict]]:
    """(corrected answer, fixes applied as [{"wrong", "right", "why"}]).
    On any error the answer is returned unchanged with no fixes."""
    if not answer.strip() or not excerpts.strip():
        return answer, []
    model = config.VERIFY_MODEL
    try:
        resp = _call(model, PROMPT.format(question=question[:2000], excerpts=excerpts, answer=answer))
    except Exception as e:  # the answer is still shown if the check fails
        print(f"fact-check failed: {e}")
        return answer, []
    if usage is not None:
        usage.add_generation(model, resp.usage_metadata)
    try:
        issues = json.loads(resp.text or "{}").get("issues", [])
    except json.JSONDecodeError:
        return answer, []
    fixed, applied = answer, []
    for it in issues:
        wrong, right = it.get("wrong", ""), it.get("right", "")
        if not wrong.strip() or wrong.strip() == right.strip():
            continue
        new = _replace(fixed, wrong, right)
        if new is not None:
            fixed = re.sub(r"\n{3,}", "\n\n", new)
            applied.append({"wrong": wrong, "right": right, "why": it.get("why", "")})
    # If the fixes would gut the answer, the excerpts were probably the wrong ones for it
    # (not that every sentence is wrong). Keep the answer and flag it instead of a blank reply.
    if applied and _words(fixed) < 0.4 * _words(answer):
        return answer, [dict(f, kept=True) for f in applied]
    return fixed, applied


def _words(text: str) -> int:
    return len(re.findall(r"\w+", text))


def unconfirmed(fixes: list[dict]) -> bool:
    """The check flagged most of the answer, so it was left as written."""
    return bool(fixes) and all(f.get("kept") for f in fixes)


def note(fixes: list[dict]) -> str:
    """One line for under the answer."""
    if unconfirmed(fixes):
        return "⚠️ Couldn't confirm most of this against the class material. Double-check it."
    if not fixes:
        return "✅ Fact-checked against the class material."
    return f"✅ Fact-checked: corrected {len(fixes)} detail{'s' if len(fixes) > 1 else ''} against the class material."


def check_hits(answer: str, hits: list[dict], question: str, usage=None) -> tuple[str, list[dict]]:
    """check() against the excerpts an answer was written from (with their links and times)."""
    from .answer import clean_links, format_context
    # The tip under an answer is the app's, not the model's: leave it out of the check.
    body, tip = split_tip(answer)
    fixed, fixes = check(body, format_context(hits, links=True) if hits else "", question, usage)
    return clean_links(fixed) + tip, fixes


_TIP = re.compile(r"\n\n\*Want just one\?")


def split_tip(answer: str) -> tuple[str, str]:
    """(answer, tip under it); the tip is "" when there is none."""
    m = _TIP.search(answer)
    return (answer[:m.start()], answer[m.start():]) if m else (answer, "")
