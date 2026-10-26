"""Play (read aloud) and Copy buttons under an answer.

Play streams Gemini's voice from the /voice route (see server.py): raw audio arrives in
chunks and is scheduled with the Web Audio API, so it starts in about a second. Answers
played before come back as a saved MP3. If the route isn't there (app.py run directly) or
fails, it falls back to the browser's built-in voice.

The buttons live in a small same-origin iframe; the current player is kept on the parent
page so starting one answer stops any other."""
import json
from html import escape as html_escape

import streamlit as st

_HTML = """
<div class="bar">
  <button id="play" title="__PLAY_TITLE__">__PLAY_LABEL__</button>
  <button id="stop" title="Stop reading. Parts not made yet are not charged." hidden>■ Stop</button>
  <button id="copy" title="Copy the whole answer">📋 Copy</button>
  <span id="note"></span>
</div>
<style>
  body { margin: 0; font-family: "Source Sans Pro", sans-serif; }
  .bar { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
  button {
    font: inherit; font-size: 13px; padding: 3px 10px; cursor: pointer; border-radius: 6px;
    border: 1px solid __BORDER__; background: transparent; color: __FG__;
  }
  button[hidden] { display: none; }
  button:hover { border-color: __ACCENT__; color: __ACCENT__; }
  #note { font-size: 12px; color: __MUTED__; }
</style>
<script>
const TEXT = __TEXT__;
const URL_ = __URL__;
const PLAY_LABEL = __PLAY_LABEL_JS__;
const play = document.getElementById("play");
const note = document.getElementById("note");
const stopBtn = document.getElementById("stop");
let top_ = window;
try { void window.parent.document; top_ = window.parent; } catch (e) {}

// ---- Gemini voice (streamed) ----
function makePlayer() {
  const ctx = new (top_.AudioContext || top_.webkitAudioContext)({ sampleRate: 24000 });
  const p = { ctx, nextTime: 0, ended: false, sources: 0, aborted: false, onend: null,
              controller: new AbortController(), segs: [], total: 0, cur: 0 };
  p.schedule = (buf) => {
    const src = ctx.createBufferSource();
    src.buffer = buf; src.connect(ctx.destination);
    const t = Math.max(p.nextTime, ctx.currentTime + 0.05);
    src.start(t); p.nextTime = t + buf.duration; p.sources++;
    p.segs.push({ t, dur: buf.duration, off: p.total }); p.total += buf.duration;
    src.onended = () => { p.sources--; if (p.ended && p.sources === 0 && p.onend) p.onend(); };
  };
  // Seconds of the answer heard so far (skips any gaps while audio was still arriving).
  p.position = () => {
    const ct = ctx.currentTime;
    while (p.cur + 1 < p.segs.length && p.segs[p.cur + 1].t <= ct) p.cur++;
    const s = p.segs[p.cur];
    if (!s || ct < s.t) return 0;
    return s.off + Math.min(ct - s.t, s.dur);
  };
  p.stop = () => { p.aborted = true; p.controller.abort(); try { ctx.close(); } catch (e) {} };
  return p;
}

async function streamVoice(p) {
  const res = await fetch(URL_, { signal: p.controller.signal });
  if (!res.ok) throw new Error("voice " + res.status);
  if ((res.headers.get("content-type") || "").includes("audio/mpeg")) {
    p.schedule(await p.ctx.decodeAudioData(await res.arrayBuffer()));
    p.ended = true; return;
  }
  const reader = res.body.getReader();
  let leftover = null, got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    let bytes = value;
    if (leftover) { bytes = new Uint8Array(leftover.length + value.length);
                    bytes.set(leftover); bytes.set(value, leftover.length); leftover = null; }
    if (bytes.length % 2) { leftover = bytes.slice(-1); bytes = bytes.slice(0, -1); }
    if (!bytes.length) continue;
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.length);
    const n = bytes.length / 2, buf = p.ctx.createBuffer(1, n, 24000), ch = buf.getChannelData(0);
    for (let i = 0; i < n; i++) ch[i] = view.getInt16(i * 2, true) / 32768;
    p.schedule(buf); got += n;
  }
  if (!got) throw new Error("no audio");
  p.ended = true;
  if (p.sources === 0 && p.onend) p.onend();
}

// ---- Browser voice (fallback) ----
function plain(md) {
  return md
    .replace(/```[\\s\\S]*?```/g, " ")
    .replace(/\\[(\\d+(,\\s*\\d+)*)\\]/g, "")
    .replace(/\\[([^\\]]+)\\]\\([^)]*\\)/g, "$1")
    .replace(/^#+\\s*/gm, "")
    .replace(/^\\s*[-*+]\\s+/gm, "")
    .replace(/^\\s*\\d+[.)]\\s+/gm, "")
    .replace(/[*_`>|#~]/g, "")
    .replace(/\\n{2,}/g, ".\\n")
    .replace(/\\s+/g, " ")
    .trim();
}
function browserVoice(onend, onprogress) {
  const synth = top_.speechSynthesis || window.speechSynthesis;
  const vs = synth.getVoices();
  const v = vs.find(x => x.lang === "en-IN") || vs.find(x => x.lang.startsWith("en")) || null;
  const pieces = plain(TEXT).match(/[^.!?\\n]+[.!?]*/g) || [];
  const total = pieces.reduce((a, s) => a + s.length, 0) || 1;
  let before = 0;
  synth.cancel();
  pieces.forEach((s, i) => {
    const u = new SpeechSynthesisUtterance(s.trim()), at = before;
    before += s.length;
    if (v) { u.voice = v; u.lang = v.lang; }
    u.onstart = () => onprogress(at / total);
    u.onboundary = (e) => onprogress((at + e.charIndex) / total);
    if (i === pieces.length - 1) u.onend = onend;
    synth.speak(u);
  });
  return { stop: () => synth.cancel(), pause: () => synth.pause(), resume: () => synth.resume() };
}

// ---- Live highlight of the words being spoken ----
// Gemini gives no word timings, so position in the audio is mapped to words by length,
// with extra time for pauses at punctuation and paragraph ends. Uses the CSS Custom
// Highlight API on the parent page: nothing in Streamlit's DOM is changed.
const SPEAK_RATE = 15.5;  // length units per second, used until the full length is known
function makeHighlighter() {
  const doc = top_.document;
  if (!top_.CSS || !top_.CSS.highlights || !top_.Highlight || !window.frameElement) return null;
  let style = doc.getElementById("pm-speak-style");
  if (!style) { style = doc.createElement("style"); style.id = "pm-speak-style"; doc.head.appendChild(style); }
  style.textContent = `
    ::highlight(pm-sentence) { background-color: __HL_SENT__; }
    ::highlight(pm-word) { background-color: __HL_WORD__; color: __HL_WORD_FG__;
      text-decoration: underline 2px __HL_LINE__; text-underline-offset: 3px; }`;
  const msg = window.frameElement.closest('[data-testid="stChatMessage"]');
  const findRoot = () => msg && msg.querySelector('[data-testid="stMarkdownContainer"]');
  const BLOCK = "p, li, h1, h2, h3, h4, h5, h6, td, th, blockquote";
  let words = [], cum = [], W = 0, last = -1, lastUserScroll = 0;
  const onUser = () => { lastUserScroll = Date.now(); };
  doc.addEventListener("wheel", onUser, { passive: true });
  doc.addEventListener("touchmove", onUser, { passive: true });

  function collect() {
    const root = findRoot();
    words = []; cum = []; W = 0; last = -1;
    if (!root) return;
    const walker = doc.createTreeWalker(root, NodeFilter.SHOW_TEXT, { acceptNode: (n) =>
      !n.nodeValue.trim() || n.parentElement.closest("pre, .katex-mathml, svg, button")
        ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT });
    for (let n; (n = walker.nextNode());) {
      const block = n.parentElement.closest(BLOCK) || root;
      for (const m of n.nodeValue.matchAll(/\\S+/g)) {
        const w = m[0];
        if (!/[\\p{L}\\p{N}]/u.test(w) || (/[\\[\\]]/.test(w) && /^[\\[\\]\\d,.;:]+$/.test(w))) continue;
        let wt = w.length + 1;
        if (/[.!?]["'”’)\\]]*$/.test(w)) wt += 8; else if (/[,;:]["'”’)]*$/.test(w)) wt += 4;
        words.push({ node: n, start: m.index, end: m.index + w.length, block, wt,
                     stop: /[.!?:]["'”’)\\]]*$/.test(w) });
      }
    }
    for (let i = 0; i < words.length; i++) {
      const next = words[i + 1];
      if (!next || next.block !== words[i].block) { words[i].wt += 10; words[i].stop = true; }
      W += words[i].wt; cum.push(W);
    }
    // Sentence spans for the soft background.
    let s = 0;
    for (let i = 0; i < words.length; i++) {
      words[i].sent = s;
      if (words[i].stop) { for (let j = s; j <= i; j++) words[j].sentEnd = i; s = i + 1; }
    }
  }
  collect();
  if (!words.length) return null;

  function range(a, b) {
    const r = doc.createRange();
    r.setStart(words[a].node, words[a].start); r.setEnd(words[b].node, words[b].end);
    return r;
  }
  return {
    weight: () => W,
    show(frac) {
      if (!words[0].node.isConnected) { collect(); if (!words.length) return; }
      const target = Math.max(0, Math.min(1, frac)) * W;
      let lo = 0, hi = words.length - 1;
      while (lo < hi) { const mid = (lo + hi) >> 1; if (cum[mid] > target) hi = mid; else lo = mid + 1; }
      if (lo === last) return;
      last = lo;
      const w = words[lo], wr = range(lo, lo);
      top_.CSS.highlights.set("pm-word", new top_.Highlight(wr));
      top_.CSS.highlights.set("pm-sentence", new top_.Highlight(range(w.sent, w.sentEnd)));
      // Follow along, unless the reader scrolled away in the last few seconds.
      const rect = wr.getBoundingClientRect();
      if (Date.now() - lastUserScroll > 4000 && (rect.top < 70 || rect.bottom > top_.innerHeight - 140))
        w.node.parentElement.scrollIntoView({ block: "center", behavior: "smooth" });
    },
    clear() {
      top_.CSS.highlights.delete("pm-word"); top_.CSS.highlights.delete("pm-sentence");
      doc.removeEventListener("wheel", onUser); doc.removeEventListener("touchmove", onUser);
    },
  };
}

// ---- Button ----
let player = null, state = "idle", hl = null;  // state: idle | loading | playing | paused
function setState(s) {
  state = s;
  play.textContent = { idle: PLAY_LABEL, loading: "⏳ Loading…", playing: "⏸ Pause",
                       paused: "▶ Resume" }[s];
  stopBtn.hidden = s === "idle";
}
stopBtn.onclick = () => { if (top_.__pmVoice) { try { top_.__pmVoice.stop(); } catch (e) {} } };
function finish() { player = null; setState("idle"); if (hl) { hl.clear(); hl = null; } }

play.onclick = async () => {
  if (state === "playing") { player.pause(); setState("paused"); return; }
  if (state === "paused") { player.resume(); setState("playing"); return; }
  if (state === "loading") return;
  if (top_.__pmVoice) { try { top_.__pmVoice.stop(); } catch (e) {} }
  setState("loading"); note.textContent = "";
  const p = makePlayer();
  const mine = { stop: () => { p.stop(); finish(); } };
  player = { pause: () => p.ctx.suspend(), resume: () => p.ctx.resume() };
  top_.__pmVoice = mine;
  p.onend = () => { if (top_.__pmVoice === mine) top_.__pmVoice = null; p.stop(); finish(); };
  try { hl = makeHighlighter(); } catch (e) { hl = null; }
  const myHl = hl;
  if (myHl) {
    const tick = () => {
      if (p.aborted || hl !== myHl) return;
      if (p.sources > 0 || p.ended) {
        const total = p.ended ? p.total : Math.max(p.total, myHl.weight() / SPEAK_RATE);
        if (total > 0) myHl.show(p.position() / total);
      }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }
  const started = setInterval(() => {
    if (p.sources > 0 && state === "loading") { setState("playing"); clearInterval(started); }
  }, 100);
  try {
    if (!URL_) throw new Error("no voice route");
    await streamVoice(p);
  } catch (e) {
    clearInterval(started);
    if (p.aborted) return;
    p.stop();
    if (p.sources > 0) { finish(); return; }  // played part of it; don't restart in another voice
    note.textContent = "Using basic voice";
    if (!hl) { try { hl = makeHighlighter(); } catch (e2) { hl = null; } }
    const b = browserVoice(() => finish(), (f) => { if (hl) hl.show(f); });
    player = b; top_.__pmVoice = { stop: () => { b.stop(); finish(); } };
    setState("playing");
  }
};

document.getElementById("copy").onclick = async (ev) => {
  const btn = ev.currentTarget;
  try {
    await navigator.clipboard.writeText(TEXT);
  } catch (e) {
    const ta = document.createElement("textarea");
    ta.value = TEXT; document.body.appendChild(ta); ta.select();
    document.execCommand("copy"); ta.remove();
  }
  btn.textContent = "✓ Copied";
  setTimeout(() => btn.textContent = "📋 Copy", 1500);
};
</script>
"""


def _js_string(text: str) -> str:
    # The answer is model output: escape every "<" so it can't close the <script> tag.
    return json.dumps(text).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def render(text: str, voice_url: str = "", dark: bool = False,
           play_label: str = "🔊 Play", play_title: str = "Listen to the answer"):
    html = (_HTML
            .replace("__PLAY_LABEL_JS__", _js_string(play_label))
            .replace("__PLAY_LABEL__", html_escape(play_label))
            .replace("__PLAY_TITLE__", html_escape(play_title, quote=True))
            .replace("__TEXT__", _js_string(text))
            .replace("__URL__", _js_string(voice_url))
            .replace("__FG__", "#f0e3c6" if dark else "#1b211a")
            .replace("__BORDER__", "#4d6141" if dark else "#d2be94")
            .replace("__MUTED__", "#a9b39a" if dark else "#6b6a55")
            .replace("__ACCENT__", "#9cc26f" if dark else "#1b211a")
            .replace("__HL_SENT__", "rgba(156, 194, 111, 0.12)" if dark else "rgba(139, 174, 102, 0.22)")
            .replace("__HL_WORD__", "rgba(156, 194, 111, 0.35)" if dark else "rgba(27, 33, 26, 0.16)")
            .replace("__HL_WORD_FG__", "#ffffff" if dark else "#1b211a")
            .replace("__HL_LINE__", "#9cc26f" if dark else "#1b211a"))
    st.iframe(html, height=34)
