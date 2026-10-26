"""Run the test questions in eval/questions.json against the RAG and grade them.

Run:  uv run python -m eval.run            # all questions
      uv run python -m eval.run zomato-mvp  # just these ids

Checks per question: the expected file was retrieved, the answer mentions the key terms,
class filters were respected, and off-topic questions say "wasn't covered". Questions with
"pictures" also check the slides shown: "expect" ("file" or "file:page") must be among
them, "forbid" must not, "none" means no picture at all, "max_copies" caps repeats of a slide.
The full answers go to eval/results.md so you can read them against good_answer.
"""
import json
import re
import sys
from pathlib import Path

from rag import images
from rag.answer import ask, search_query

HERE = Path(__file__).parent
NOT_COVERED = "wasn't covered in the class material"


def norm(text: str) -> str:
    # "4,000" = "4000", "18 to 24" = "18–24" = "18-24", "Product-Market" = "product market"
    text = re.sub(r"(?<=\d),(?=\d)", "", text.lower())
    text = re.sub(r"(?<=\d)\s+to\s+(?=\d)", " ", text)
    return " ".join(re.sub(r"[-–—*]", " ", text).split())


def _shows(pics: list[dict], want: str) -> bool:
    file, _, page = want.partition(":")
    return any(file.lower() in p["label"].lower() and (not page or p["label"].endswith(f"page {page}"))
               for p in pics)


def grade_pictures(q: dict, pics: list[dict]) -> list[str]:
    rule, problems = q["pictures"], []
    if rule.get("none") and pics:
        problems.append(f"showed {len(pics)} picture(s), expected none")
    for want in rule.get("expect", []):
        if not _shows(pics, want):
            problems.append(f"picture not shown: {want}")
    for bad in rule.get("forbid", []):
        if _shows(pics, bad):
            problems.append(f"wrong picture shown: {bad}")
    if "max_copies" in rule:
        prints = [images._fingerprint(p["name"]) for p in pics]
        if any(prints.count(x) > rule["max_copies"] for x in prints):
            problems.append("same picture shown more than once")
    return problems


def grade(q: dict, hits: list[dict], answer: str) -> list[str]:
    problems = []
    files = [h["file_name"] for h in hits]
    if q.get("source") and not any(s.lower() in f.lower() for s in q["source"] for f in files):
        problems.append(f"expected file not retrieved ({' / '.join(q['source'])})")
    missing = [m for m in q.get("mention", []) if norm(m) not in norm(answer)]
    if missing:
        problems.append(f"answer doesn't mention: {', '.join(missing)}")
    if q.get("class_labels"):
        stray = {h["class_label"] for h in hits} - set(q["class_labels"])
        if stray:
            problems.append(f"hits from outside the filter: {', '.join(stray)}")
    if q.get("not_covered") and NOT_COVERED not in answer.replace("’", "'"):
        problems.append("didn't say it wasn't covered")
    if q.get("covered") and NOT_COVERED in answer.replace("’", "'"):
        problems.append("wrongly said it wasn't covered")
    if q.get("shorter_than_last"):
        last = next(m["content"] for m in reversed(q["history"]) if m["role"] == "assistant")
        if len(answer) > len(last) / 2:
            problems.append(f"not much shorter ({len(answer)} vs {len(last)} chars before)")
    return problems


def main(ids: list[str]):
    questions = json.loads((HERE / "questions.json").read_text())
    if ids:
        questions = [q for q in questions if q["id"] in ids]
    report, passed = [], 0
    for q in questions:
        hits, stream = ask(q["question"], q.get("history", []), class_labels=q.get("class_labels"),
                           allow_general=q.get("allow_general", True))
        answer = "".join(stream)
        problems = grade(q, hits, answer)
        pics = []
        if q.get("pictures"):
            pics = images.find(search_query(q["question"], q.get("history", []))[0], q.get("class_labels"))
            problems += grade_pictures(q, pics)
        passed += not problems
        print(f"{'PASS' if not problems else 'FAIL'}  {q['id']:<20} {'; '.join(problems)}")
        sources = "\n".join(f"- {h['class_label']} — {h['file_name']}"
                            + (f", page {h['page']}" if h.get("page") else "") for h in hits)
        report.append(f"## {'✅' if not problems else '❌'} {q['id']} ({q['type']})\n\n"
                      f"**Q:** {q['question']}\n\n**Good answer:** {q['good_answer']}\n\n"
                      + (f"**Problems:** {'; '.join(problems)}\n\n" if problems else "")
                      + f"**Answer:**\n\n{answer}\n\n**Retrieved:**\n\n{sources}\n"
                      + ("\n**Pictures shown:**\n\n" + ("\n".join(f"- {p['label']} (fit {p['score']}/10)"
                                                               for p in pics) or "- none") + "\n"
                         if q.get("pictures") else ""))
    print(f"\n{passed}/{len(questions)} passed")
    (HERE / "results.md").write_text(f"# RAG test results: {passed}/{len(questions)} passed\n\n"
                                     + "\n---\n\n".join(report))
    print(f"Full answers: {HERE / 'results.md'}")


if __name__ == "__main__":
    main(sys.argv[1:])
