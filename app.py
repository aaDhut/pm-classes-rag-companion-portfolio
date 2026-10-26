"""Chat with the PM classes.   Run:  uv run streamlit run server.py  (adds the Gemini voice route)"""
import copy
import hmac
import json
import secrets
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime

import streamlit as st

import answer_actions
import dictation
import styles
from rag import chats, config, images, llm, modes, quiz, store, verify, voice
from rag.answer import (ask, cached_overview, clean_links, is_course_overview, nothing_to_restyle,
                        remember_overview, search_query)
from rag.topic import main_classes
from rag.usage import TOKENS_HELP, Usage, describe, describe_voice, fmt_money

st.set_page_config(page_title="PM Classes Assistant", page_icon="📘", layout="centered")


@st.cache_resource
def background() -> ThreadPoolExecutor:
    """Finds an answer's pictures while it is written, and fact-checks it after."""
    return ThreadPoolExecutor(8)

@st.cache_resource
def theme_resumes() -> dict:
    """One-time tickets that carry a session across the reload a theme switch needs:
    ticket -> (chat_id, expiry time)."""
    return {}


# Back from a theme switch: stay logged in and reopen the same chat.
if ticket := st.query_params.pop("resume", None):
    chat_id, expires = theme_resumes().pop(ticket, (None, 0))
    if chat_id and time.time() < expires:
        st.session_state.authed = True
        st.session_state.chat_id = chat_id
        st.session_state.messages = chats.load(chat_id)

# Password gate for the public Cloud Run URL. Local runs without APP_PASSWORD skip it.
if config.APP_PASSWORD and not st.session_state.get("authed"):
    st.title("📘 PM Classes Assistant")
    with st.form("login"):
        pw = st.text_input("Password", type="password")
        if st.form_submit_button("Enter"):
            if hmac.compare_digest(pw.encode(), config.APP_PASSWORD.encode()):
                st.session_state.authed = True
                st.rerun()
            st.error("Wrong password.")
    st.stop()

# Light/dark is each browser's own choice (following the device until someone picks one).
# The sidebar toggle saves the pick where Streamlit's ⋮ → Settings → Theme keeps it, in this
# browser only, so it never changes the theme for anyone else.
dark = (st.context.theme.type or "light") == "dark"
styles.apply(dark)


def switch_theme():
    """Streamlit only reads the saved theme when the page loads, so save it and reload,
    with a one-time ticket that brings back the login and the open chat."""
    ticket = secrets.token_urlsafe(16)
    resumes = theme_resumes()
    for t, (_, exp) in list(resumes.items()):
        if exp < time.time():
            resumes.pop(t, None)
    resumes[ticket] = (st.session_state.chat_id, time.time() + 120)
    st.session_state.theme_switch = ("Dark" if st.session_state.dark_mode else "Light", ticket)


def apply_theme_switch():
    theme, ticket = st.session_state.pop("theme_switch")
    # The iframe's sandbox blocks it from reloading the page, so the code runs as a
    # script added to the page itself.
    code = (f"localStorage.setItem(`stActiveTheme-${{location.pathname}}-v2`, {json.dumps(json.dumps(theme))});"
            f"location.replace(location.pathname + '?resume=' + {json.dumps(ticket)});")
    st.iframe(f"""<script>
const s = window.parent.document.createElement("script");
s.textContent = {json.dumps(code)};
window.parent.document.body.appendChild(s);
</script>""", height=1)


@st.cache_data(ttl="6h")  # new classes only come with an ingest run
def classes() -> list[str]:
    return store.indexed_classes()


if "chat_id" not in st.session_state:
    st.session_state.chat_id = chats.new_id()
    st.session_state.messages = []


def saved_quiz(messages: list[dict]) -> dict | None:
    """The quiz still running in this chat, if any (it lives on the latest answer)."""
    last = next((m for m in reversed(messages) if m["role"] == "assistant"), None)
    return last["quiz"] if last and last.get("quiz", {}).get("active") else None


def open_chat(chat_id: str):
    st.session_state.chat_id = chat_id
    st.session_state.messages = chats.load(chat_id)


def start_new_chat():
    st.session_state.chat_id = chats.new_id()
    st.session_state.messages = []


def day_label(iso: str) -> str:
    d = datetime.fromisoformat(iso).date()
    delta = (date.today() - d).days
    return "Today" if delta == 0 else "Yesterday" if delta == 1 else d.strftime("%d %b %Y")


with st.sidebar:
    st.header("📘 PM Classes")
    st.button("New chat", icon=":material/add:", key="new_chat", use_container_width=True, on_click=start_new_chat)
    st.toggle("🌙 Dark mode", value=dark, key="dark_mode", on_change=switch_theme)
    if "theme_switch" in st.session_state:
        apply_theme_switch()

    with st.expander("Search settings", expanded=False):
        picked = st.multiselect("Limit to classes or workshops", classes(), placeholder="All classes")
        sources = {"Lectures + notes": "all", "Lecture videos only": "lectures", "Notes only": "notes"}[
            st.radio("Search in", ["Lectures + notes", "Lecture videos only", "Notes only"], horizontal=True,
                     help="Lecture videos = what was said in class (transcripts and class summaries). "
                          "Notes = the PDFs, Docs and Sheets in the course folder.")]
        allow_general = st.toggle("Allow general answer if not covered", value=False)
        dict_lang = st.selectbox(
            "🎤 Dictation language", list(dictation.LANGUAGES),
            help="Language the mic listens for. 'Hindi / Hinglish' writes Hindi words in "
                 "Devanagari; 'English (India)' handles Indian accents and English PM terms best.")
        k = st.slider(
            "Passages to read per answer", 4, 16, 8,
            help="How many slides/passages from the class material are fetched and given to the AI "
                 "before it answers. 6-8 suits specific questions. Use 12-16 for broad ones like "
                 "'summarise Class 9' or 'compare the MVP types across classes'. More passages mean "
                 "more coverage but slower answers and a bit more off-topic material.")

    dictation.render(dictation.LANGUAGES[dict_lang])
    st.subheader("History")
    # Filled at the end of the run, and also as soon as a question is sent (before its
    # answer is written), so the open chat shows up in the list straight away.
    history_box = st.empty()


def render_history(answering: bool = False):
    """Chats grouped by day, each day a fold-out list. The open chat is always listed:
    as a "New chat" card until its first question is saved. answering=True is the early
    draw while an answer is still being written; it gets its own widget keys because the
    end of the same run draws the list again."""
    tag = "a" if answering else ""
    cur = st.session_state.chat_id
    history = chats.list_chats()
    if not any(c["id"] == cur for c in history):
        with st.container(border=True, key=f"hist_{tag}new"):
            st.button("New chat", key=f"open_{tag}new", use_container_width=True, type="primary")
            st.caption("Ask a question to start")
    elif not history:
        st.caption("Your chats will appear here.")
    days: dict[str, list[dict]] = {}
    for c in history:
        days.setdefault(day_label(c["updated"]), []).append(c)
    for label, group in days.items():
        has_cur = any(c["id"] == cur for c in group)
        # Today and the day holding the open chat start unfolded; older days stay folded.
        with st.expander(f"{label} · {len(group)}", expanded=label == "Today" or has_cur):
            for c in group:
                current = c["id"] == cur
                # One card per chat (styled in styles.py); the open chat's card is highlighted.
                with st.container(border=True, key=f"hist_{tag}{c['id']}"):
                    cols = st.columns([0.85, 0.15], vertical_alignment="center")
                    cols[0].button(c["title"], key=f"open_{tag}{c['id']}", use_container_width=True,
                                   type="primary" if current else "secondary",
                                   on_click=open_chat, args=(c["id"],))
                    deleted = cols[1].button("", icon=":material/delete:", key=f"del_{tag}{c['id']}",
                                             help="Delete this chat")
                    # Kept to one line so the card stays short; token counts show inside the chat.
                    if current and answering:
                        st.caption("⏳ Answering…")
                    else:
                        q = c.get("questions", 0)
                        st.caption(f"{fmt_money(c['cost'], c['cost_inr']) if c.get('cost') is not None else '—'}"
                                   f" · {q} question{'s' if q != 1 else ''}")
                if deleted:
                    chats.delete(c["id"])
                    if current:
                        start_new_chat()
                    st.rerun()


def show_check(fixes: list[dict]):
    """The fact-check line under an answer, with what it corrected."""
    if not fixes:
        st.caption(verify.note(fixes))
        return
    with st.expander(verify.note(fixes)):
        if verify.unconfirmed(fixes):
            for f in fixes:
                st.markdown(f"- {f['wrong'].strip()}  \n  <small>{f['why']}</small>", unsafe_allow_html=True)
            return
        for f in fixes:
            st.markdown(f"- ~~{f['wrong'].strip()}~~ → {f['right'].strip() or '*(removed)*'}  \n"
                        f"  <small>{f['why']}</small>", unsafe_allow_html=True)


def show_sources(hits: list[dict]):
    with st.expander(f"Sources ({len(hits)})"):
        for i, h in enumerate(hits, 1):
            if h.get("source") == "lecture":
                name = f"🎬 [{h['file_name']} at {h['time']}]({h['link']})"
            elif h.get("source") == "slide":
                name = f"🖼️ [Slide shown in the lecture video at {h['time']}]({h['link']})"
            else:
                place = f" · page {h['page']}" if h.get("page") else ""
                if h.get("notes_folder") and h["notes_folder"] != h["class_label"]:
                    place += f" · Drive folder: {h['notes_folder']}"
                name = (f"[{h['file_name']}]({h['link']})" if h.get("link") else h["file_name"]) + place
            st.markdown(f"**[{i}] {h['class_label']}** — {name}")
            st.caption(h["text"][:400] + ("…" if len(h["text"]) > 400 else ""))


st.title("Ask about your PM classes")

def checks() -> dict:
    """Fact-checks still running for this session: (chat_id, answer index) -> job."""
    return st.session_state.setdefault("checks", {})


def apply_checks():
    """Put finished fact-checks into their answers, here and in the saved chat."""
    for key, job in list(checks().items()):
        if not job["future"].done():
            continue
        del checks()[key]
        chat_id, i = key
        try:
            fixed, fixes = job["future"].result()
        except Exception as e:  # the answer stays as written
            print(f"fact-check failed: {e}")
            fixed, fixes = job["original"], None
        current = chat_id == st.session_state.chat_id
        msgs = st.session_state.messages if current else chats.load(chat_id)
        if i >= len(msgs) or not msgs[i]["content"].startswith(job["original"]):
            continue  # chat deleted or changed meanwhile
        m = msgs[i]
        m.pop("checking", None)
        if fixes is not None:
            # Keep anything added after the answer (the "quiz is paused" note).
            m["content"] = fixed + m["content"][len(job["original"]):]
            m["checked"] = fixes
        m["usage"] = job["usage"].to_dict()
        chats.save(chat_id, msgs)


@st.fragment(run_every=1.5)
def check_status(key):
    """'Fact-checking…' under an answer; redraws the page once the check is done.
    These timed runs wait for a running answer instead of interrupting it."""
    job = checks().get(key)
    if job is None or job["future"].done():
        st.rerun()
    st.caption("⏳ Fact-checking against the class material…")


apply_checks()

if st.session_state.messages:
    # Voice costs are written to the saved chat by the /voice route; pick them up.
    for m, saved in zip(st.session_state.messages, chats.cached(st.session_state.chat_id)):
        if "tts" in saved and saved["content"] == m["content"]:
            m["tts"] = saved["tts"]
    t = chats.totals(st.session_state.messages)
    if t["cost"] is not None:
        st.caption(f"This chat so far: {fmt_money(t['cost'], t['cost_inr'])} · {t['tokens']:,} tokens"
                   f" · {t['questions']} question{'s' if t['questions'] != 1 else ''}",
                   help=TOKENS_HELP)
else:
    st.markdown("Try: *What steps did Instructor A give for writing a PRD?* · *Summarise class 14* · "
                "*Summarise everything about pricing* · *Quiz me on GTM* · "
                "*What did Instructor B say about story points?*")

def answer_footer(i: int, m: dict):
    """Play/Copy buttons and cost lines under answer number i."""
    cid = st.session_state.chat_id
    label, title = "🔊 Play", "Listen to the answer"
    if (m.get("tts") or {}).get("complete"):  # saved MP3: replays cost nothing
        label, title = "🔊 Play · free", "Already read aloud once, so replaying is free."
    elif est := voice.estimate(m["content"]):
        label = f"🔊 Play · ~₹{est[1]:.1f}"
        title = (f"Reading this answer aloud costs about ₹{est[1]:.1f} (${est[0]:.3f}). "
                 "The whole answer is voiced once you click, even if you stop early. "
                 "Replays after that are free.")
    answer_actions.render(m["content"], dark=dark, play_label=label, play_title=title,
                          voice_url=f"/voice/{cid}/{i}?sig={voice.sign(cid, i)}")
    if m.get("usage"):
        st.caption(describe(m["usage"]), help=TOKENS_HELP)
    if m.get("tts"):
        st.caption(describe_voice(m["tts"]), help=TOKENS_HELP)


for i, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("checking") and (st.session_state.chat_id, i) in checks():
            check_status((st.session_state.chat_id, i))
        elif "checked" in m:
            show_check(m["checked"])
        images.render(m.get("images", []))
        if m.get("hits"):
            show_sources(m["hits"])
        if m["role"] == "assistant":
            answer_footer(i, m)


def read_input(value, usage: Usage) -> str | None:
    """Typed text, or the dictated question turned into text."""
    if value is None:
        return None
    text = (value.text if hasattr(value, "text") else str(value)).strip()
    audio = getattr(value, "audio", None)
    if text or audio is None:
        return text or None
    with st.spinner("Listening…"):
        text = llm.transcribe(audio.getvalue(), audio.type or "audio/wav", usage)
    if not text:
        st.warning("Couldn't hear a question in that recording. Please try again.")
    return text or None


usage = Usage()
question = read_input(st.chat_input("Ask anything from the classes… or tap 🎤 to speak",
                                    accept_audio=True), usage)
def summary_reply(labels: list[str]) -> str | None:
    """The stored lecture summaries of these classes; None if none of them has one."""
    found = [text for lb in labels if (text := store.class_summary(lb))]
    return "\n\n---\n\n".join(found) if found else None


if question:
    history_msgs = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    # The chat without quiz turns, for a side question asked during a quiz: "summarise
    # this" there means the last answer, not the quiz question.
    no_quiz = [m for i, m in enumerate(st.session_state.messages)
               if "quiz" not in m and not (i + 1 < len(st.session_state.messages)
                                           and "quiz" in st.session_state.messages[i + 1])]
    chat_msgs = [{"role": m["role"], "content": m["content"]} for m in no_quiz]
    st.session_state.messages.append({"role": "user", "content": question})
    # Save the question now, not after the answer, so the chat is in History right away
    # and isn't lost if the answer is cut off.
    chats.save(st.session_state.chat_id, st.session_state.messages)
    with history_box.container():
        render_history(answering=True)
    with st.chat_message("user"):
        st.markdown(question)
    # Sending another question (or clicking anything) stops this run. Keep what was
    # written, so the chat and the next follow-up still have this turn.
    answered, partial, check_job = False, [], None

    def collect(stream):
        for piece in stream:
            partial.append(piece)
            yield piece

    try:
        with st.chat_message("assistant"):
            # Marks this bubble as "being written" so its avatar cycles monkeys (see styles.py).
            live = st.empty()
            live.html('<span class="monkey-live"></span>')
            running_quiz = copy.deepcopy(saved_quiz(st.session_state.messages[:-1]))
            intent = modes.detect(question, classes())
            # During a quiz: an answer, a stop, "where was this taught?", or another question
            # (answered normally while the quiz waits).
            quiz_kind = quiz.kind(running_quiz, question) if running_quiz else None
            if quiz_kind == "aside":
                history_msgs = chat_msgs
            msg = None
            if quiz_kind == "stop":
                msg = {"role": "assistant", "content": quiz.finish(running_quiz, stopped=True), "quiz": running_quiz}
            elif not running_quiz and intent["mode"] == "quiz" and quiz.is_stop(question):
                msg = {"role": "assistant", "content": "No quiz is running. Ask for one any time, e.g. *quiz me on GTM*."}
            elif intent["mode"] == "quiz":
                n, level = modes.quiz_options(question)
                with st.spinner("Making your quiz from the class material…"):
                    state, text = quiz.start(question, intent["classes"] or picked, n, level, usage,
                                             prev_topic=quiz.topic_from_history(st.session_state.messages[:-1]))
                msg = {"role": "assistant", "content": text, **({"quiz": state} if state else {})}
            elif quiz_kind == "answer":
                with st.spinner("Checking your answer…"):
                    text = quiz.reply(running_quiz, question, usage)
                msg = {"role": "assistant", "content": text, "quiz": running_quiz}
            elif quiz_kind == "where":
                with st.spinner("Finding where this was taught…"):
                    text = quiz.where_reply(running_quiz, usage)
                msg = {"role": "assistant", "content": text, "quiz": running_quiz}
            elif intent["mode"] == "class_summary" and (text := summary_reply(intent["classes"])):
                msg = {"role": "assistant", "content": text}
            elif text := nothing_to_restyle(question, history_msgs):
                msg = {"role": "assistant", "content": text}
            elif (not intent["classes"] and not picked and is_course_overview(question, history_msgs)
                  and (saved := cached_overview())):
                # Already written and fact-checked for an earlier student: free.
                msg = {"role": "assistant", "content": saved[0], "hits": saved[1], "checked": []}
            if msg is not None:
                st.markdown(msg["content"])
                if msg.get("hits"):
                    show_sources(msg["hits"])
                msg["usage"] = usage.to_dict()
            else:
                topic, restyle = search_query(question, history_msgs)
                # "Summarise this" shortens the last answer; it isn't a new topic summary.
                summary_mode = intent["mode"] in ("class_summary", "topic_summary") and not restyle
                filt = intent["classes"] if intent["mode"] == "class_summary" else picked
                last_pics = next((m.get("images", []) for m in reversed(st.session_state.messages)
                                  if m["role"] == "assistant"), [])
                with st.spinner("Searching class material…"):
                    hits, stream = ask(question, history_msgs, class_labels=filt or None,
                                       allow_general=allow_general, k=k, usage=usage, sources=sources,
                                       topic_summary=summary_mode)
                # Pictures from the classes that teach the topic (found by ask), not from a class
                # that only shares a word with the question. A "make it shorter/simpler" request
                # keeps the pictures of the answer it rewrites.
                # A course overview spans every class, so no one class's pictures fit it.
                overview = not filt and is_course_overview(question, history_msgs)
                if overview:
                    last_pics = []
                pics_job = None if restyle or overview else background().submit(
                    images.find, topic, main_classes(hits) or filt or None, usage)
                box = st.empty()
                with box.container():
                    written = st.write_stream(collect(stream))
                answer = clean_links(written)
                if answer != written:
                    box.markdown(answer)
                # The answer is saved now and fact-checked in the background, so the next
                # question can be asked straight away (see apply_checks).
                check_status_box = st.container()
                pics = last_pics if pics_job is None else pics_job.result()
                images.render(pics)
                if hits:
                    show_sources(hits)
                msg = {"role": "assistant", "content": answer, "hits": hits,
                       "images": pics, "usage": usage.to_dict(), "checking": True}
                check_job = (answer, hits)
            if quiz_kind == "aside" and running_quiz.get("active"):
                # Keep the quiz going after answering the side question.
                note = (f"\n\n---\n\n*Your quiz is paused on question {running_quiz['i'] + 1} of "
                        f"{len(running_quiz['questions'])}. Reply A, B, C or D to carry on, or type **stop quiz**.*")
                st.markdown(note)
                msg["content"] += note
                msg["quiz"] = running_quiz
            live.empty()
            st.session_state.messages.append(msg)
            answered = True
            chats.save(st.session_state.chat_id, st.session_state.messages)
            idx = len(st.session_state.messages) - 1
            if check_job:
                key = (st.session_state.chat_id, idx)
                future = background().submit(verify.check_hits, *check_job, question, usage)
                if overview:
                    hits_now = check_job[1]
                    future.add_done_callback(
                        lambda f: f.exception() is None and remember_overview(f.result()[0], hits_now))
                checks()[key] = {"original": check_job[0], "usage": usage, "future": future}
                with check_status_box:
                    check_status(key)
            answer_footer(idx, msg)
    except BaseException as e:
        if not answered:
            text = "".join(partial).strip()
            why = ("This answer was stopped before it finished." if not isinstance(e, Exception)
                   else "Something went wrong while answering. Please ask again.")
            st.session_state.messages.append({"role": "assistant",
                                              "content": (text + "\n\n" if text else "") + f"*({why})*"})
            chats.save(st.session_state.chat_id, st.session_state.messages)
        raise

with history_box.container():
    render_history()
