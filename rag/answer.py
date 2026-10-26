"""Build the grounded prompt and stream Gemini's answer."""
import re

from . import config, llm, store, topic

SYSTEM = """You are a study assistant for a product management course. Classes were taught
by Instructor A and by Instructor B; each class says who taught it.
You answer the student's questions using the class material excerpts provided. There are
four kinds of excerpt:
- Lecture video: what was said in class, as "[time] Speaker: words", sometimes followed by
  "[On screen]" text read off the shared slides/screen.
- Slide: one slide, diagram or screen shown in a lecture video at a time: a description of
  what it shows, its text, and what was said while it was up. Cite it as the slide shown in
  that class at that time (e.g. "the slide shown in Class 14 at 1:19:50 [2]").
- Class summary: study notes written from a whole lecture video.
- Notes: the PDFs, Docs and Sheets shared for the class. Classes are numbered by lecture
  video; the Drive notes folders use older numbers, shown as (Drive folder "Class 12 - …").
  Always use the class number from the excerpt header, and mention the Drive folder name
  when pointing the student to a notes file.

Rules:
- Base the answer on the excerpts. Cite them inline as [1], [2] matching the excerpt numbers.
- Notes pages may include tables marked "[Table … read each value by its row and column]".
  Take every value (High/Low, numbers, ticks) from the table's own row and column, never
  from the loose page text above it, where the columns are jumbled. Copy numbers exactly.
- When a point comes from a lecture, say which class and roughly when, and who said it
  (e.g. "In Class 14, Instructor B explained… [3]"). A student's question or opinion is not
  the teacher's teaching: say "a student asked…" and give the teacher's reply. Student
  answers or notes shown on screen or read out in class are the students' views, not what
  the teacher taught. The lecture transcripts are machine-made, so quietly fix obvious
  mis-heard words; never mention transcription slips or say the teacher mispronounced
  something.
- Everything you present as class material must be in the excerpts. Don't add your own
  definitions, reasons, numbers or examples to it (e.g. a textbook definition of a Kano
  category that the class didn't give). Extra general knowledge goes only in a section
  headed "General answer (not from class):".
- For "what is / what are X": open with a one- or two-line plain definition of X from the
  excerpts, then the details. For "how many types / kinds": give every grouping the
  excerpts show (e.g. kinds of AI and kinds of LLMs) and say how many there are in each.
- Class slides sometimes have slips (a tool under the wrong heading, a misspelt name).
  Keep what the class showed, and when you are sure from general knowledge that it is
  wrong, add right after it a short line starting "⚠️ Note (general knowledge, not from
  class):" with the correction, e.g. "Pika Labs is a video model, not an image model."
  Only for clear factual slips; never for opinions or the teacher's own framing.
- Prefer the exact frameworks, steps, examples, case studies, tools and hands-on exercises
  from the classes over generic textbook explanations, and say which class they came from.
- If the excerpts don't cover the question, say clearly "This wasn't covered in the class
  material I have", then (only if general knowledge is allowed) give a short general answer
  labelled "General answer (not from class):".
- If the excerpts mention the topic but not the specific detail asked (e.g. they name a
  framework but don't explain its name, origin or author), first say what the class does
  show, with citations, then say that specific detail wasn't explained in class.
- When general knowledge is allowed, never stop at "wasn't covered": always follow it with
  the general answer. A reply that is only "This wasn't covered…" is not acceptable then.
- If the student asks you to rephrase the previous answer (simpler, shorter, more detail,
  in Hindi/Hinglish, with an example, etc.), rewrite your previous answer from the
  conversation that way. That is a style request, not a new topic, so never reply
  "wasn't covered" to it. Keep the citations from the previous answer.
- When asked for simpler / easy / simple English: write for someone with no PM background
  whose English is basic. Common everyday words, short sentences (under 15 words), no
  jargon (if a class term is needed, explain it in a few words), one everyday analogy.
- When a new question asks for a gist, pointers only, in short or in brief: at most about
  8 short bullets (roughly 150 words), no long sections, tables or "Where to revise" list.
- When asked for shorter: about half the length of the previous answer or less. Keep
  only the main idea and the key points; drop extra examples and side sections.
- When asked to summarise your previous answer ("summarise this", "summary please"): a
  short recap, at most a third of its length: the key points as bullets, no new sections
  or material.
- Each repeated "simpler", "shorter" or "summarise" request must be clearly simpler and
  shorter than the answer before it. Never repeat the previous answer as it was.
- When the student asks where something was taught, or for a link, timestamp or lecture
  video: list each place as a markdown link using the link given in the excerpt header,
  e.g. "[Class 20 lecture video at 12:22](link)" or "[Some.pdf, page 3](link)". Put lecture
  video moments first. Only use links that appear in the excerpts, copied exactly; never
  make up a link or put anything but a full https:// address inside (…).
- Lecture times: every time you give must fall inside the time range of the excerpt you
  cite, where the thing is actually said. In a class summary excerpt, use the time of the
  section that covers it, not the time of a nearby student question. A "Slide" excerpt
  only shows what was on screen: say "shown on a slide in Class N at T", and say sir
  explained or discussed something only when a lecture-video excerpt has him saying it.
- The app separately shows class slides/diagrams that match the question below your
  answer, when there are any. When asked to show a diagram, slide, chart or framework,
  explain it from the excerpts and say that matching slides from class appear below if
  available. Don't name or number specific pictures. Never say you can't show, display or
  render images.
- Be clear and practical. Use headings/bullets for multi-part answers. Match the student's
  language (English or Hinglish)."""


def where(h: dict) -> str:
    """'Lecture video (Instructor B), 47:26–49:10' / 'Some.pdf, page 3'."""
    if h.get("source") == "lecture":
        return f"{h['file_name']}, {h['time']}"
    if h.get("source") == "slide":
        return f"slide shown in the lecture video at {h['time']}"
    out = h["file_name"] + (f", page {h['page']}" if h.get("page") else "")
    if h.get("notes_folder") and h["notes_folder"] != h["class_label"]:
        out += f" (Drive folder \"{h['notes_folder']}\")"
    return out


def format_context(hits: list[dict], links: bool = False) -> str:
    """links: add each excerpt's link (video at that time / Drive file) to its header."""
    return "\n\n---\n\n".join(
        f"[{i}] {h['class_label']} — {where(h)}"
        + (f" — link: {h['link']}" if links and h.get("link") else "") + f"\n{h['text']}"
        for i, h in enumerate(hits, 1))


TOPIC_SUMMARY_TASK = """The student asked: "{question}"
Write a study summary of everything the classes taught about this topic, from the excerpts.
- Organise it by sub-topic with short headings, not by excerpt.
- For each point say which class it came from (and who taught it), citing [n].
- Include the frameworks, steps, definitions, examples and tools that were taught, and
  anything the teacher stressed or warned about.
- End with "Where to revise": the classes, lecture times and notes pages to go back to."""


COURSE_OVERVIEW_TASK = """The student asked: "{question}"
The excerpts are the summaries of every class in the course so far, in order.
Answer with the core concepts of the course as a whole:
- Group the classes into 4-7 themes (e.g. research, defining and building, launch and
  growth, AI products). Under each theme list the key concepts and frameworks, one short
  line each, saying which class taught it and citing [n].
- Only use what the summaries say; do not add concepts that are not in them.
- End with "Where to start": 3-5 classes to revise first, and why, in one line each."""

# "What are the core concepts I should understand?" names no topic: about the whole course.
_OVERVIEW_WORDS = set("""what whats what's are is the core key main important most essential basic basics
fundamental fundamentals concept concepts topic topics thing things idea ideas lesson lessons takeaway
takeaways skill skills should i need to must know learn understand cover covered taught study revise
remember from this course program programme so far all overall in of a an me my please pls give list
tell which do does we have been be""".split())
_OVERVIEW_NEED = re.compile(r"\b(core|key|main|important|essential|fundamentals?|basics|takeaways?|"
                            r"concepts?|topics?|lessons?|skills?)\b", re.I)
_WHOLE_COURSE = re.compile(r"\b(course|program|programme|so far|overall|all (the )?classes)\b", re.I)


_MAP_LINES = re.compile(r"^(?:📍 \*\*Taught mainly in:\*\*|\*\*Also comes up in:\*\*).*\n*", re.M)


def is_course_overview(question: str, history: list[dict]) -> bool:
    """A request for the course's core concepts: only general words, and either nothing
    was asked before it or it names the course ("main topics of the course")."""
    words = re.findall(r"[a-z0-9']+", question.lower())
    if not words or not all(w in _OVERVIEW_WORDS for w in words) or not _OVERVIEW_NEED.search(question):
        return False
    return bool(_WHOLE_COURSE.search(question)) or not any(m["role"] == "user" for m in history)


# The course overview is the same for everyone until a class is added, so the first
# fact-checked one is kept and reused: {class labels: (answer, hits)}.
_OVERVIEWS: dict = {}


def _overview_key() -> tuple:
    return tuple(p["class_label"] for p in store.all_summaries())


def cached_overview() -> tuple[str, list[dict]] | None:
    return _OVERVIEWS.get(_overview_key())


def remember_overview(answer: str, hits: list[dict]):
    _OVERVIEWS.clear()  # only the latest course matters
    _OVERVIEWS[_overview_key()] = (answer, hits)


def _outline(full_text: str) -> str:
    """A class summary cut to its opening paragraph and its section headings."""
    intro, _, rest = full_text.partition("**What was covered**")
    heads = re.findall(r"^- (?:\[\[[^\]]*\]\]\([^)]*\) )?\*\*(.+?)\*\*", rest, re.M)
    intro = "\n".join(l for l in intro.splitlines() if l.strip() and not l.startswith(("###", "**Taught")))
    return intro.strip() + ("\nTopics: " + "; ".join(heads) if heads else "")


def nothing_to_restyle(question: str, history: list[dict]) -> str | None:
    """'Summarise this' / 'make it simpler' with no earlier answer: say what to ask instead."""
    if any(m["role"] == "assistant" for m in history) or not is_restyle(question):
        return None
    return ("There's nothing above to summarise yet. Ask a question first, or try:\n"
            "- *summarise class 12*: one class\n- *summarise stakeholder management*: one topic across classes\n"
            "- *what are the core concepts of the course?*: the whole course")


# A follow-up made only of these words asks to restyle the last answer, not a new topic.
# Prefixes so typos and forms still match: "shotter", "simplier", "easier", "points".
_STYLE = re.compile(r"^(tl|simp|easy|easi|esay|short|shot|shrt|brief|concise|summar|tldr|eli5|"
                    r"layman|plain|basic|clear|hindi|hinglish|english|example|elaborat|detail|"
                    r"long|deep|bullet|point|step|line|sentence|word|again|rephrase|reword|rewrite|"
                    # "explain like I'm five / a 5 year old / a kid / a beginner / for dummies"
                    r"five$|5$|kid|child|beginner|dumm|noob|newbie)")
_FILLER = set("""make it this that more much very even bit little a an the pls please plz kindly
can could would you u explain explanation answer in and or to with of me for give write say put
terms way language some now too so one version is be are try keep just only form using use like
i im i'm am my kid child 5 year old five beginner someone new non technical tech bro yaar thoda
aur mein me karo kar do samjhao samjha bata batao it's its lot also dr""".split())


# Asking for the link / timestamp / video of what was just discussed.
_WHERE = re.compile(r"\b(links?|time ?stamps?|timings?|videos?|where|urls?|sources?|references?|watch)\b", re.I)
_WHERE_WORDS = set("""link links timestamp timestamps time stamp timing video videos lecture lectures
class session where which what url source sources reference references watch find location minute
recording taught discussed covered said was were did""".split())


def asks_where(question: str) -> bool:
    return bool(_WHERE.search(question))


def is_where_followup(question: str) -> bool:
    """'give me the lecture video, timestamp and link for it' — no topic of its own."""
    words = re.findall(r"[a-z0-9']+", question.lower())
    return asks_where(question) and all(w in _FILLER or w in _WHERE_WORDS for w in words)


# "Summarise this", "make that shorter", "tl;dr of the above" point back at the last answer,
# even when the message also names its topic ("discuss partnerships and summarise this").
_SHORTEN = re.compile(r"\b(summar\w*|summery|sumar\w*|recap|gist|tl;?dr|shorten|shorter|condense|"
                      r"brief|in short|key points)\b", re.I)
_BACK = re.compile(r"\b(this|that|above|it|these|those|your (?:last |previous )?(?:answer|reply)|"
                   r"the (?:last |previous )?(?:answer|reply)|what you (?:said|wrote))\b", re.I)
_NEW_SCOPE = re.compile(r"\b(class|classes|lecture|session|workshop|course|chapter|module|"
                        r"week|everything|all)\b", re.I)


_NUM_WORDS = r"\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen"
# "like I'm 12", "a 10-year-old": an age alone asks for a simpler version.
_AGE = re.compile(rf"\blike (?:i'?m|i am) (?:{_NUM_WORDS})(?:[- ]?(?:years?|yrs?)[- ]?old)?\b|\b(?:{_NUM_WORDS})[- ]?(?:years?|yrs?)[- ]?old\b",
                  re.I)
# "in 3 points", "in 100 words": a length, not "point 2" of the last answer.
_LENGTH = re.compile(rf"\b(?:{_NUM_WORDS}) (points?|lines?|words?|sentences?|bullets?|steps?|paras?|paragraphs?)\b",
                     re.I)


def is_restyle(question: str) -> bool:
    # An age reads as "kid", a length as its unit word ("points"), before the word check.
    styled = _LENGTH.sub(r" \1 ", _AGE.sub(" kid ", question.lower()))
    words = re.findall(r"[a-z0-9']+", styled)
    if (any(_STYLE.match(w) for w in words)
            and all(w in _FILLER or _STYLE.match(w) for w in words)):
        return True
    # "summarise this class" / "recap of the whole course" ask for something new.
    return bool(_SHORTEN.search(question) and _BACK.search(question) and not _NEW_SCOPE.search(question))


# Words that point back at the last topic: "how is it different", "the second one".
_BACK_REF = set("""it its it's this that these those they them their above same previous
former latter""".split())
_WHICH_ONE = re.compile(r"\b(first|second|third|fourth|fifth|last|other|next|that|this) ones?\b", re.I)
# Question words with no topic of their own.
_GENERIC = _FILLER | _WHERE_WORDS | set("""what whats what's how why when who whom does do did
is are was were mean means meaning define definition tell about more detail details example
examples type types kind kinds different difference differ compare comparison between vs versus
similar use used uses using work works happen happens list many much there here ok okay thanks
thank yes no then also need needed important importance difference benefit benefits advantage
advantages disadvantage disadvantages pros cons should can will role purpose""".split())
_CONTINUE = re.compile(r"^\s*(and|also|but|so|then|what about|how about|why not|same for)\b", re.I)


def is_followup(question: str) -> bool:
    """True when the question needs the earlier ones to make sense: it has no topic word,
    starts like a continuation ("and…", "what about…"), or points back ("it", "the second
    one") before naming any topic. "What is figma and how do PMs use it" names its topic
    first, so it stands alone."""
    if _CONTINUE.search(question) or _WHICH_ONE.search(question):
        return True
    for w in re.findall(r"[a-z0-9']+", question.lower()):
        if w in _BACK_REF:
            return True
        if w not in _GENERIC and len(w) > 1:
            return False
    return True  # nothing but generic words: "why?", "explain more"


def search_query(question: str, history: list[dict]) -> tuple[str, bool]:
    """(text to search the class material with, whether it's a restyle request)."""
    last_answer = next((m["content"] for m in reversed(history) if m["role"] == "assistant"), "")
    restyle = bool(last_answer) and is_restyle(question)
    user_turns = [m["content"] for m in history if m["role"] == "user"]
    if restyle or (last_answer and is_where_followup(question)):
        # Search with the question that started the topic, skipping earlier follow-ups.
        i = next((i for i in range(len(user_turns) - 1, -1, -1)
                  if not is_restyle(user_turns[i]) and not is_where_followup(user_turns[i])), None)
        if i is None:
            return question, restyle
        # A vague start ("and what is it") gets the questions before it, as below.
        start = user_turns[i]
        for q in reversed(user_turns[:i]):
            if len(start) >= 80:
                break
            start = f"{q}\n{start}"
        return start, restyle
    # Follow-ups like "explain the second one" carry no topic, so add earlier questions
    # until there's enough text to find it. A question with its own topic ("what is
    # figma") is searched alone: adding the last topic mixed the two.
    search_text = question
    if not is_followup(question):
        return search_text, False
    for q in reversed(user_turns):
        if len(search_text) >= 80:
            break
        search_text = f"{q}\n{search_text}"
    return search_text, False


_key = topic.key

# Questions about counts, prices or calculations: the cheaper model got these wrong more often.
_NUMBERS = re.compile(r"\d|%|₹|\$|\bhow (many|much)\b|\b(cost\w*|pric\w*|calculat\w*|numbers?|"
                      r"tokens?|percent\w*|ltv|cac|tam|sam|som|revenue|budget)\b", re.I)


def pick_model(question: str, where: bool) -> str:
    """The answer model: the cheaper one (if set) unless the question asks where something
    was taught or is about numbers; the cheaper model made more mistakes on those."""
    if not config.ANSWER_MODEL_LITE or where or _NUMBERS.search(question):
        return config.ANSWER_MODEL
    return config.ANSWER_MODEL_LITE


def _search_both(search_text: str, question: str, restyle: bool, class_labels, k: int, usage,
                 sources: str) -> list[dict]:
    """Search with the student's words and with the same question in the course's words,
    merged best first. Students say "graded" where the notes say "evaluation criteria";
    one search alone misses those pages. k: how many chunks each search returns."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(store.search, search_text, class_labels=class_labels, limit=k, usage=usage,
                            sources=sources)
        rewritten = "" if restyle else llm.rewrite_query(search_text[-1500:], usage)
        second = (store.search(rewritten, class_labels=class_labels, limit=k, usage=usage, sources=sources)
                  if rewritten and rewritten.lower() != search_text.lower() else [])
        a = first.result()
    return topic.merge([a, second] if second else [a])


_NOT_COVERED = re.compile(r"(wasn.?t|was not|isn.?t|not) (been )?(covered|taught|discussed)", re.I)


def _with_tip(stream, tip: str):
    """The answer, then the tip under it, unless the answer opens with "this wasn't covered"."""
    start = ""
    for piece in stream:
        if len(start) < 200:
            start += piece
        yield piece
    if not _NOT_COVERED.search(start[:200]):
        yield "\n\n" + tip


def ask(question: str, history: list[dict], class_labels: list[str] | None = None,
        allow_general: bool = True, k: int = 8, usage=None, sources: str = "all",
        topic_summary: bool = False):
    """Returns (hits, token stream). Tokens used are added to usage (a rag.usage.Usage).
    topic_summary: write a cross-class summary of the topic instead of a plain answer."""
    last_answer = next((m["content"] for m in reversed(history) if m["role"] == "assistant"), "")
    search_text, restyle = search_query(question, history)
    if topic_summary:
        restyle, k = False, max(k, 20)
    where_followup = not restyle and bool(last_answer) and is_where_followup(question)
    overview = not restyle and not class_labels and is_course_overview(question, history)
    plan = None
    if overview:
        hits = [{**p, "text": _outline(p["full_text"])} for p in store.all_summaries()]
    elif restyle or where_followup:
        hits = topic.select(_search_both(search_text, question, restyle, class_labels, k, usage, sources),
                            None, k, search_text, class_labels, sources, usage)
    else:
        # Wide search, find the class(es) that teach the topic, then pick the excerpts.
        cands = _search_both(search_text, question, restyle, class_labels, topic.WIDE, usage, sources)
        plan = topic.route(search_text[-1500:], cands, usage, search=lambda q: _search_both(
            q, q, True, class_labels, topic.WIDE, usage, sources))
        hits = topic.select(cands, plan, k, search_text, class_labels, sources, usage)
    note, tip = topic.prompt_note(plan, hits), topic.footer(plan, hits)
    hits = [{kk: v for kk, v in h.items() if not kk.startswith("_")} for h in hits]
    if asks_where(question) and sources != "notes":
        # "Where was X taught?" needs the lecture moments, so add a few more of them.
        seen = {_key(h) for h in hits}
        extra = store.search(search_text, class_labels=class_labels, limit=6, usage=usage, sources="lectures")
        hits += [h for h in extra if _key(h) not in seen][:4]

    # Without the app's own "📍 Taught mainly in" / "Also comes up in" lines: the model
    # copied them into its answer, and the app adds them again.
    convo = "\n".join(f"{m['role'].upper()}: {_MAP_LINES.sub('', m['content'])[:4000]}" for m in history[-6:])
    if restyle:
        task = (f"The student wants your previous answer rewritten: \"{question}\"\n"
                "Rewrite the previous answer (below) that way, on the same topic. This is not a new "
                "question, so do not say it wasn't covered. Do not mention that you are rewriting.\n\n"
                f"Your previous answer:\n{last_answer[:8000]}")
    elif where_followup:
        task = (f"The student asks: \"{question}\"\nThey want to know where the topic of your previous "
                "answer was taught. List the lecture video moments (class, time, link) and the notes "
                "files (page, link) from the excerpts that cover it, each with one line on what is "
                "there. Lecture videos first.")
    elif overview:
        task = COURSE_OVERVIEW_TASK.format(question=question)
    elif topic_summary:
        task = TOPIC_SUMMARY_TASK.format(question=question)
    else:
        task = f"Student's question: {question}"
        if allow_general:
            # Kept next to the question: from the system prompt alone the model sometimes
            # stops at "wasn't covered" without the general answer.
            task += ("\n\nIf the excerpts don't fully answer this (any part of it), you must end "
                     "with a section headed \"General answer (not from class):\" that answers it "
                     "from general knowledge.")
    prompt = (
        (f"Conversation so far:\n{convo}\n\n" if convo else "")
        + f"Class material excerpts:\n\n{format_context(hits, links=True) if hits else '(no matching excerpts)'}\n\n"
        + (f"{note}\n\n" if note else "")
        + f"General knowledge allowed: {'yes' if allow_general else 'no'}\n\n"
        + task
    )
    model = pick_model(question, where_followup or asks_where(question))
    stream = llm.answer_stream(SYSTEM, prompt, usage, model=model)
    if tip:
        stream = _with_tip(stream, tip)
    return hits, stream


# A markdown link whose target is not a web address (one level of brackets allowed inside).
_BAD_LINK = re.compile(r"\[([^\[\]]+)\]\((?!https?://)[^()\n]*(?:\([^()\n]*\)[^()\n]*)*\)")


def clean_links(text: str) -> str:
    """Markdown links whose target isn't a web address become plain text (the model
    sometimes puts a source label where the URL should be)."""
    return _BAD_LINK.sub(lambda m: m.group(1), text)
