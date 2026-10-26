"""Works out what kind of request a message is: a class summary, a topic summary, a quiz,
or a normal question. Plain word rules, so it costs nothing and is instant."""
import re

_SUMMARY = re.compile(
    r"\b(summar\w*|summery|recap|overview|gist|tl;?dr|"
    r"what (?:all )?(?:happened|was (?:covered|taught|discussed)|did (?:\w+ ){0,2}(?:cover|teach|discuss|talk about))|"
    r"what (?:\w+ ){0,2}(?:sir )?(?:covered|taught))\b", re.I)
_QUIZ = re.compile(r"\b(quiz|test me|mcqs?|practice questions|ask me (?:some )?questions|"
                   r"question me|viva)\b", re.I)
_CLASS_NO = re.compile(r"\b(?:class|session|lecture|lec)\s*(?:no\.?|number|#)?\s*(\d{1,2})\b"
                       r"|\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:class|session|lecture)\b", re.I)
_MONTHS = {"sep": "sep", "sept": "sep", "september": "sep", "oct": "oct", "october": "oct",
           "nov": "nov", "november": "nov", "aug": "aug", "august": "aug"}
_STOP = set("the a an of on in about me my summary summarise summarize class workshop sunday what was "
            "happened please quiz test give".split())


def class_refs(text: str, labels: list[str]) -> list[str]:
    """Class / workshop labels the message points at ("class 9", "14th lecture", "PRD workshop")."""
    found = []
    for m in _CLASS_NO.finditer(text):
        no = int(m.group(1) or m.group(2))
        found += [lb for lb in labels if lb.startswith(f"Class {no} - ")]
    if re.search(r"workshop", text, re.I):
        words = {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP}
        words |= {_MONTHS[w] for w in words if w in _MONTHS}
        best, score = None, 0
        for lb in labels:
            if "workshop" not in lb.lower():
                continue
            lw = set(re.findall(r"[a-z0-9]+", lb.lower()))
            lw |= {_MONTHS[w] for w in lw if w in _MONTHS} | {re.sub(r"(st|nd|rd|th)$", "", w) for w in lw}
            s = len(words & lw)
            if s > score:
                best, score = lb, s
        if best:
            found.append(best)
    return list(dict.fromkeys(found))


def detect(text: str, labels: list[str]) -> dict:
    """{"mode": "quiz" | "class_summary" | "topic_summary" | "ask", "classes": [...]}"""
    refs = class_refs(text, labels)
    if _QUIZ.search(text):
        return {"mode": "quiz", "classes": refs}
    if _SUMMARY.search(text):
        return {"mode": "class_summary" if refs else "topic_summary", "classes": refs}
    return {"mode": "ask", "classes": refs}


def quiz_options(text: str) -> tuple[int, str]:
    """(number of questions, difficulty) asked for, defaulting to 5 and medium."""
    m = re.search(r"\b(\d{1,2})\s*(?:questions?|qs?|mcqs?)\b", text, re.I)
    n = min(max(int(m.group(1)), 1), 15) if m else 5
    level = next((lv for lv in ("easy", "hard") if re.search(rf"\b{lv}\b", text, re.I)), "medium")
    if re.search(r"\b(tough|difficult|advanced)\b", text, re.I):
        level = "hard"
    if re.search(r"\b(simple|basic)\b", text, re.I):
        level = "easy"
    return n, level
