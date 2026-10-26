"""Live dictation for the chat box.

Adds a mic button inside Streamlit's chat input that uses the browser's speech recognition
(Chrome, Edge, Safari): the words appear in the box while you speak, the button pulses red
with "Listening…", and you can edit before pressing Enter. Free, no API call.

Where the browser has no speech recognition (e.g. Firefox), Streamlit's own mic stays
visible and the recording is transcribed by Gemini after you stop (see app.read_input).

The script is installed once on the app page itself (the iframe is same-origin), so it
keeps working across reruns; each run only updates the chosen language."""
import json

import streamlit as st

LANGUAGES = {"English (India)": "en-IN", "Hindi / Hinglish": "hi-IN", "English (US)": "en-US"}

_SCRIPT = r"""
(() => {
  const W = window, D = document;
  const SR = W.SpeechRecognition || W.webkitSpeechRecognition;
  if (!SR || W.__pmDictation) return;
  W.__pmDictation = true;

  const css = D.createElement("style");
  css.textContent = `
    body.pm-dict [data-testid="stChatInputMicButton"] { display: none !important; }
    #pm-mic { border: none; background: transparent; cursor: pointer; border-radius: 50%;
      width: 34px; height: 34px; display: inline-flex; align-items: center; justify-content: center;
      color: inherit; opacity: .75; flex: none; align-self: flex-end; margin-bottom: 2px; }
    #pm-mic:hover { opacity: 1; background: rgba(128,128,128,.15); }
    #pm-mic.on { opacity: 1; color: #fff; background: #e5484d; animation: pm-pulse 1.2s infinite; }
    @keyframes pm-pulse { 0% { box-shadow: 0 0 0 0 rgba(229,72,77,.55); }
                          70% { box-shadow: 0 0 0 10px rgba(229,72,77,0); }
                          100% { box-shadow: 0 0 0 0 rgba(229,72,77,0); } }
    [data-testid="stChatInput"].pm-listening { outline: 2px solid #e5484d; border-radius: 12px; }
    #pm-dict-status { position: fixed; z-index: 1000; font: 13px "Source Sans Pro", sans-serif;
      background: #e5484d; color: #fff; padding: 4px 12px; border-radius: 14px; display: none;
      box-shadow: 0 2px 8px rgba(0,0,0,.2); }
    #pm-dict-status.err { background: #6b6f76; }
    #pm-dict-status .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
      background: #fff; margin-right: 6px; animation: pm-blink 1s infinite; }
    @keyframes pm-blink { 50% { opacity: .2; } }`;
  D.head.appendChild(css);
  D.body.classList.add("pm-dict");

  const MIC = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor"
    stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="2" width="6"
    height="12" rx="3"/><path d="M5 10v1a7 7 0 0 0 14 0v-1"/><line x1="12" y1="18" x2="12"
    y2="22"/></svg>`;
  const STOP = `<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><rect x="5"
    y="5" width="14" height="14" rx="2"/></svg>`;

  const status = D.createElement("div");
  status.id = "pm-dict-status";
  D.body.appendChild(status);

  const box = () => D.querySelector('[data-testid="stChatInput"]');
  const area = () => D.querySelector('[data-testid="stChatInputTextArea"]');
  const setter = Object.getOwnPropertyDescriptor(W.HTMLTextAreaElement.prototype, "value").set;
  function write(text) {
    const ta = area(); if (!ta) return;
    setter.call(ta, text);
    ta.dispatchEvent(new Event("input", { bubbles: true }));  // let React see the new value
    ta.scrollTop = ta.scrollHeight;
  }

  function showStatus(html, err) {
    const b = box(); if (!b) return;
    const r = b.getBoundingClientRect();
    status.innerHTML = html; status.className = err ? "err" : "";
    status.style.display = "block";
    status.style.left = (r.left + 8) + "px";
    status.style.top = (r.top - 34) + "px";
  }
  const hideStatus = () => { status.style.display = "none"; };

  let rec = null, listening = false, base = "", finals = "", silence = null, oldPh = "";

  function stop() { if (rec) { try { rec.stop(); } catch (e) {} } }

  function start() {
    const ta = area(); if (!ta) return;
    rec = new SR();
    rec.lang = W.__pmDictLang || "en-IN";
    rec.continuous = true; rec.interimResults = true;
    base = ta.value ? ta.value.replace(/\s*$/, " ") : "";
    finals = "";
    rec.onstart = () => {
      listening = true; render();
      oldPh = ta.placeholder; ta.placeholder = "Listening… speak now";
      showStatus('<span class="dot"></span>Listening… click ■ to stop');
    };
    rec.onresult = (ev) => {
      let interim = "";
      for (let i = ev.resultIndex; i < ev.results.length; i++) {
        const t = ev.results[i][0].transcript;
        if (ev.results[i].isFinal) finals += t; else interim += t;
      }
      write((base + finals + interim).replace(/\s+/g, " ").trimStart());
      clearTimeout(silence);
      silence = setTimeout(stop, 4000);  // stop after 4 s of silence
    };
    rec.onerror = (ev) => {
      const msg = { "not-allowed": "Microphone blocked. Allow it in the address bar.",
                    "service-not-allowed": "Microphone blocked. Allow it in the address bar.",
                    "no-speech": "Didn't hear anything. Try again.",
                    "audio-capture": "No microphone found.",
                    "network": "Speech service unreachable. Use Chrome or Edge." }[ev.error];
      if (msg) { showStatus(msg, true); setTimeout(hideStatus, 3500); }
    };
    rec.onend = () => {
      listening = false; clearTimeout(silence); render();
      const t = area(); if (t) { t.placeholder = oldPh || t.placeholder; t.focus(); }
      if (status.className !== "err") hideStatus();
      rec = null;
    };
    try { rec.start(); } catch (e) {}
  }

  let btn = null;
  function render() {
    if (!btn) return;
    btn.classList.toggle("on", listening);
    btn.innerHTML = listening ? STOP : MIC;
    btn.title = listening ? "Stop dictation" : "Dictate your question";
    const b = box(); if (b) b.classList.toggle("pm-listening", listening);
  }

  // Streamlit re-renders the chat input; put the button back whenever it goes missing.
  function mount() {
    const send = D.querySelector('[data-testid="stChatInputSubmitButton"]');
    if (!send || D.getElementById("pm-mic")) return;
    btn = D.createElement("button");
    btn.id = "pm-mic"; btn.type = "button";
    btn.onclick = (e) => { e.preventDefault(); listening ? stop() : start(); };
    send.parentElement.insertBefore(btn, send);
    render();
  }
  new MutationObserver(mount).observe(D.body, { childList: true, subtree: true });
  mount();

  // Sending the question ends dictation, so late results don't refill the box.
  D.addEventListener("keydown", (e) => {
    if (listening && e.key === "Enter" && !e.shiftKey && e.target === area()) stop();
  }, true);
  D.addEventListener("click", (e) => {
    if (listening && e.target.closest && e.target.closest('[data-testid="stChatInputSubmitButton"]')) stop();
  }, true);
})();
"""


def render(lang: str = "en-IN"):
    """Install live dictation (once per page) and set its language."""
    html = f"""<script>
    try {{
      const P = window.parent;
      P.__pmDictLang = {json.dumps(lang)};
      if (!P.__pmDictation) {{
        const s = P.document.createElement("script");
        s.textContent = {json.dumps(_SCRIPT).replace("<", "\\u003c")};
        P.document.head.appendChild(s);
      }}
    }} catch (e) {{}}
    </script>"""
    st.iframe(html, height=1)
