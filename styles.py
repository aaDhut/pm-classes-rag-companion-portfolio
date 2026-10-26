"""Look of the app beyond what Streamlit's theme settings cover (colors live in
.streamlit/config.toml): plain history rows with the open chat marked, a chat bubble
for your questions, a floating question box and soft shadows."""
import random

import streamlit as st

_LIGHT = {
    "text": "#1B211A", "acc": "#1B211A", "on_acc": "#F8F0E0", "card": "#EBD5AB",
    "tint": "rgba(27, 33, 26, 0.14)", "tint_hover": "rgba(27, 33, 26, 0.07)",
    "line": "rgba(27, 33, 26, 0.45)",
    "sh_xs": "0 1px 3px rgba(27, 33, 26, 0.10)",
    "sh_sm": "0 2px 6px rgba(27, 33, 26, 0.14), 0 1px 2px rgba(27, 33, 26, 0.08)",
    "sh_md": "0 6px 18px rgba(27, 33, 26, 0.12), 0 2px 4px rgba(27, 33, 26, 0.06)",
    "sh_side": "3px 0 18px rgba(27, 33, 26, 0.18)",
    "del": "#A3321F", "del_bg": "rgba(163, 50, 31, 0.12)",
    "row": "rgba(248, 240, 224, 0.55)", "row_on": "#F8F0E0", "row_line": "rgba(27, 33, 26, 0.22)",
    "sh_row": "0 2px 6px rgba(27, 33, 26, 0.16), 0 1px 2px rgba(27, 33, 26, 0.10)",
    "sh_row_on": "0 8px 20px rgba(27, 33, 26, 0.24), 0 2px 5px rgba(27, 33, 26, 0.14)",
    "bubble_text": "#1B211A", "exp_bg": "transparent",
    "ans_bg": "#E3E3E3", "ans_line": "#CCCCCC", "side_acc": "#1B211A",
}
# Dark mirrors light's three layers: dark page, sage sidebar with lighter chat cards (the open
# one is a brighter sage with a forest-black edge), and sand for your questions.
_DARK = {
    "text": "#EFE7D4", "acc": "#A8C98A", "on_acc": "#161B15", "card": "#E3C98F",
    "tint": "rgba(168, 201, 138, 0.18)", "tint_hover": "rgba(168, 201, 138, 0.10)",
    "line": "rgba(168, 201, 138, 0.55)",
    "sh_xs": "0 1px 3px rgba(0, 0, 0, 0.35)",
    "sh_sm": "0 2px 8px rgba(0, 0, 0, 0.45), 0 1px 2px rgba(0, 0, 0, 0.30)",
    "sh_md": "0 8px 24px rgba(0, 0, 0, 0.50), 0 0 0 1px rgba(168, 201, 138, 0.10)",
    "sh_side": "3px 0 20px rgba(0, 0, 0, 0.50)",
    "del": "#F2A08B", "del_bg": "rgba(242, 160, 139, 0.18)",
    "row": "rgba(239, 231, 212, 0.13)", "row_on": "#6E9058", "row_line": "rgba(239, 231, 212, 0.20)",
    "sh_row": "0 2px 6px rgba(0, 0, 0, 0.22), 0 1px 2px rgba(0, 0, 0, 0.14)",
    "sh_row_on": "0 8px 22px rgba(0, 0, 0, 0.45), 0 0 0 1px rgba(27, 33, 26, 0.55)",
    "bubble_text": "#161B15", "exp_bg": "#1E241C",
    "ans_bg": "#2C2C2C", "ans_line": "rgba(255, 255, 255, 0.12)", "side_acc": "#1B211A",
}

_CSS = """<style>
/* Sidebar sits slightly above the page. */
[data-testid="stSidebar"] { box-shadow: __sh_side__; }

/* New chat: outlined button that lifts a little on hover. */
.st-key-new_chat button { box-shadow: __sh_xs__; transition: box-shadow .15s, transform .15s; }
.st-key-new_chat button:hover { box-shadow: __sh_sm__; transform: translateY(-1px); }

/* Sidebar text: larger and bolder so it reads well on the green. */
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label,
[data-testid="stSidebar"] summary span { font-size: 1rem; font-weight: 600; color: __text__; }
[data-testid="stSidebar"] h2 { font-weight: 800; }
[data-testid="stSidebar"] h3 { font-weight: 800; font-size: 1.25rem; }
/* Day labels (Today, Yesterday…) */
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
  font-size: 0.9rem; font-weight: 700; color: __text__; opacity: .95;
}

/* History: one compact card per chat; the open one is solid, raised and has an accent bar. */
[class*="st-key-hist_"] {
  background: __row__; border: 1px solid __row_line__; border-left: 4px solid transparent;
  border-radius: 10px; box-shadow: __sh_row__; padding: 0.3rem 0.35rem 0.4rem 0.5rem;
  gap: 0; transition: box-shadow .15s, transform .15s, background .15s;
}
[class*="st-key-hist_"] [data-testid="stHorizontalBlock"] { gap: 0.25rem; }
[class*="st-key-hist_"]:hover { box-shadow: __sh_row_on__; transform: translateY(-1px); }
[class*="st-key-hist_"]:has([data-testid="stBaseButton-primary"]) {
  background: __row_on__; border-left-color: __side_acc__; box-shadow: __sh_row_on__;
}
[class*="st-key-hist_"] [data-testid="stCaptionContainer"] { padding-left: 0.4rem; margin-top: -0.15rem; }
[class*="st-key-hist_"] [data-testid="stCaptionContainer"] p {
  font-size: 0.82rem; font-weight: 600; opacity: .85; white-space: nowrap;
  overflow: hidden; text-overflow: ellipsis;
}
[class*="st-key-open_"] button {
  background: transparent; border: none; box-shadow: none; color: __text__;
  justify-content: flex-start; padding: 0.1rem 0.4rem; min-height: 2rem;
}
[class*="st-key-open_"] button > div { justify-content: flex-start; width: 100%; }
[class*="st-key-open_"] button p { text-align: left; font-weight: 700; font-size: 1rem; }
[class*="st-key-open_"] button:hover,
[class*="st-key-open_"] button[data-testid="stBaseButton-primary"] {
  background: transparent; color: __text__;
}

/* Delete: a solid trash icon in the text color that turns red on hover. */
[class*="st-key-del_"] button {
  padding: 0; min-height: 2rem; width: 100%; display: flex; align-items: center;
  justify-content: center; background: transparent; border: none; box-shadow: none;
  color: __text__; border-radius: 8px;
}
[class*="st-key-del_"] button [data-testid="stIconMaterial"] { font-size: 1.35rem; }
[class*="st-key-del_"] button:hover { background: __del_bg__; color: __del__; }

/* Answers sit in a plain light-gray bubble; your questions are right-aligned sand bubbles. */
[data-testid="stChatMessage"] { background: transparent; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
  background: __ans_bg__; border: 1px solid __ans_line__; max-width: 92%;
  border-radius: 16px; padding: 0.75rem 1rem; box-shadow: __sh_sm__;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  background: __card__; width: fit-content; max-width: 80%; margin-left: auto;
  border-radius: 16px; padding: 0.6rem 1rem; box-shadow: __sh_sm__;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) * { color: __bubble_text__; }
[data-testid="stChatMessageAvatarUser"] { display: none; }
[data-testid="stChatMessageAvatarAssistant"] {
  background: transparent !important; border: none !important; box-shadow: none !important;
  width: 2.9rem; height: 2.9rem; display: flex; align-items: center; justify-content: center;
  position: relative;
}
/* Monkey picture instead of the robot icon. They're images (static/monkeys/, all scaled to the
   same head size), so every device shows the Apple-style monkey. While an answer is written the
   avatar flips through all 11 in an order shuffled on each run (__MONKEY_FRAMES__). */
[data-testid="stChatMessageAvatarAssistant"] > * { display: none; }
[data-testid="stChatMessageAvatarAssistant"]::before {
  content: ""; width: 2.9rem; height: 2.9rem;
  background: url("./app/static/monkeys/e1.png") center / contain no-repeat;
}
/* Loads every monkey up front so the first flip doesn't blink. */
[data-testid="stChatMessageAvatarAssistant"]::after {
  content: ""; position: absolute; width: 1px; height: 1px; opacity: 0.01; pointer-events: none;
  background: __MONKEY_ALL__; background-size: 1px 1px;
}
[data-testid="stChatMessage"]:has(.monkey-live) [data-testid="stChatMessageAvatarAssistant"]::before {
  animation: monkeys __MONKEY_SECS__s steps(1, end) infinite;
}
@keyframes monkeys { __MONKEY_FRAMES__ }

/* Question box floats above the page with an outline. */
[data-testid="stChatInput"] > div {
  border: 1px solid __line__; border-radius: 12px; box-shadow: __sh_md__;
}
[data-testid="stChatInput"] > div:focus-within { border-color: __acc__; }
[data-testid="stChatInputSubmitButton"]:not(:disabled) { background: __acc__; color: __on_acc__; }

/* Expanders (sources, fact-check, settings): rounded cards with a faint shadow. */
[data-testid="stExpander"] details { border-radius: 10px; box-shadow: __sh_xs__; background: __exp_bg__; }

/* Filled buttons (password form etc.) get a soft shadow. */
[data-testid^="stBaseButton-primary"]:not([class*="st-key-open_"] *) { box-shadow: __sh_sm__; }
__EXTRA__
</style>"""

# Dark mode only: a dark knob on switches that are on, and dark text on selected tags.
_DARK_EXTRA = """
/* Switches that are on: a cream knob on the forest-black sidebar track, a dark knob on the sage page track. */
label:has(input[role="switch"]:checked) > span + div > div { background-color: #161B15; }
[data-testid="stSidebar"] label:has(input[role="switch"]:checked) > span + div > div { background-color: #F8F0E0; }
/* Selected-class tags sit on the forest-black sidebar accent: cream text keeps them readable. */
[data-testid="stMultiSelectTagsContainer"] [role="group"] > span, [data-testid="stMultiSelectTagsContainer"] [role="group"] > span * { color: #F3ECDA !important; font-weight: 700; }

/* Sidebar: a soft top-to-bottom sage gradient instead of one flat green. */
[data-testid="stSidebar"] > div:first-child { background: linear-gradient(180deg, #557A45 0%, #4A6A3C 45%, #3E5A33 100%); }
[data-testid="stSidebar"] h2 { padding-bottom: 0.45rem; border-bottom: 2px solid rgba(235, 213, 171, 0.45); }
/* Search settings card: a slightly darker inset panel so its controls have a clear frame. */
[data-testid="stSidebar"] [data-testid="stExpander"] details {
  background: rgba(22, 27, 21, 0.22); border-color: rgba(239, 231, 212, 0.25);
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover span { color: #EBD5AB; }
/* New chat: a lighter sage pill that clearly reads as a button. */
.st-key-new_chat button { background: rgba(239, 231, 212, 0.16); border-color: rgba(239, 231, 212, 0.35); }
.st-key-new_chat button:hover { background: rgba(239, 231, 212, 0.24); border-color: #EBD5AB; color: #F8F0E0; }
/* Dropdowns: deep-forest fill with a sage edge, sand edge on hover and focus. */
[data-testid="stSidebar"] [data-baseweb="select"] > div {
  border: 1px solid #86A86A; border-radius: 10px; box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.35);
  transition: border-color .15s;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div:hover,
[data-testid="stSidebar"] [data-baseweb="select"] > div:focus-within { border-color: #EBD5AB; }
[data-testid="stSidebar"] [data-baseweb="select"] svg { color: #EBD5AB; }

/* Outlined buttons on the page (Play, Copy...) get a sage edge so they don't sink into the dark. */
[data-testid="stMain"] [data-testid="stBaseButton-secondary"] { border-color: rgba(168, 201, 138, 0.50); background: #1E241C; }
[data-testid="stMain"] [data-testid="stBaseButton-secondary"]:hover { border-color: #A8C98A; color: #A8C98A; }
"""

# Light mode only: New chat and the sidebar folders (Search settings, day groups) get a
# slightly lighter green than the sidebar so they read as buttons.
_LIGHT_EXTRA = """
.st-key-new_chat button,
[data-testid="stSidebar"] [data-testid="stExpander"] details {
  background: rgba(255, 255, 255, 0.18); border-color: rgba(27, 33, 26, 0.25);
}
.st-key-new_chat button:hover,
[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover { background: rgba(255, 255, 255, 0.12); }
"""


_MONKEYS = [f"e{i}" for i in range(1, 5)] + [f"m{i}" for i in range(1, 8)]
_MONKEY_STEP = 0.4  # seconds each monkey stays up while an answer is written


def _monkey_css(css: str) -> str:
    url = lambda n: f'url("./app/static/monkeys/{n}.png")'
    order = random.sample(_MONKEYS, len(_MONKEYS))
    frames = " ".join(f"{i * 100 / len(order):.2f}% {{ background-image: {url(n)}; }}"
                      for i, n in enumerate(order))
    return (css.replace("__MONKEY_FRAMES__", frames)
               .replace("__MONKEY_SECS__", f"{_MONKEY_STEP * len(order):.1f}")
               .replace("__MONKEY_ALL__", ", ".join(url(n) for n in _MONKEYS)))


def apply(dark: bool):
    css = _monkey_css(_CSS.replace("__EXTRA__", _DARK_EXTRA if dark else _LIGHT_EXTRA))
    for name, value in (_DARK if dark else _LIGHT).items():
        css = css.replace(f"__{name}__", value)
    st.markdown(css, unsafe_allow_html=True)
